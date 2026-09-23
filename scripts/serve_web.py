"""Web 前端：在浏览器里下任务、看 Agent 一步步操作手机。

## 为什么现在才做

前面所有工作都是命令行——跑完打一屏文字就没了。问题是：

1. **别人看不到你的 Agent。** 简历上、演示时，命令行输出说服力很弱。
2. **调试试错很难。** 模型卡住的时候，你想同时看到「它看到的屏幕」
   和「它说的话」，命令行只能二选一。

这个服务把两者并排放在一个页面里，边跑边看。

## 零新依赖

只用标准库 `http.server`。理由和 `serve_models.py` 一样：
**一个要演示给人看的组件，如果还要先 `pip install` 一堆东西，很多人就走不到看见它跑起来那一步。**

## 接口

    GET  /                    -> web/index.html
    GET  /api/tasks           -> 可用任务列表
    POST /api/run             -> {"task": "...", "max_steps": 12} 开始跑
    GET  /api/stream/<run_id> -> SSE：逐步推送 思考 / 动作 / 信号 / 截图
    GET  /api/shot/<run_id>/<n> -> 第 n 步的截图 PNG

## 为什么用 SSE 而不是 WebSocket

只需要**服务器单向推**，不需要双向。SSE 是浏览器原生支持的
（`EventSource`），服务端就是普通 HTTP 响应，不用握手、不用额外库。
WebSocket 要处理帧协议，纯标准库写起来麻烦得多——收益为零。
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from android_tasks import check_network, cleanup_device  # noqa: E402
from task_sets import collect_tasks  # noqa: E402
from guicascade.agent import Agent  # noqa: E402
from guicascade.envs.android import AndroidEnv, find_adb, load_apps  # noqa: E402
from guicascade.registry import build  # noqa: E402
from guicascade.tools import FinishTool, NoteTool, Toolkit  # noqa: E402
from guicascade.types import Trajectory  # noqa: E402

__all__ = ["main"]


# --------------------------------------------------------------------------
# 流式埋点
# --------------------------------------------------------------------------


@dataclass
class StreamTracer:
    """把每一步推进一个队列，供 SSE 取走。

    形状和 `guicascade.trace.Tracer` 一致（`log_step` / `log_trajectory`），
    所以能直接顶替它塞进 `Agent`——**不用为前端改主循环一行代码**。

    截图存到内存里的字典，不落盘：一次演示最多几十张图，
    落盘还要想清理逻辑，不划算。
    """

    run_id: str
    shots: dict[int, bytes] = field(default_factory=dict)
    q: "queue.Queue[dict[str, Any]]" = field(default_factory=queue.Queue)

    def log_step(self, task: str, step) -> None:
        d = step.to_dict()
        idx = int(d.get("step", 0))

        # 截图：有就存下来，前端按 index 取
        img = getattr(step.observation, "image", None)
        if img:
            self.shots[idx] = img

        # 给模型看的屏幕文本也带上——**这是排查问题最有用的一栏**：
        # 模型看到什么、说了什么，并排看才知道它为什么那样决策。
        screen = (getattr(step.observation, "text", "") or "")[:4000]

        self.q.put({
            "type": "step",
            "index": idx,
            "model": d.get("model", ""),
            "escalated": bool(d.get("escalated")),
            "reason": d.get("reason", ""),
            "action": d.get("action", ""),
            "signals": {k: v for k, v in d.items()
                        if k in ("stuck", "milestone") and isinstance(v, (int, float))},
            "latency_model_s": d.get("latency_model_s", 0.0),
            "latency_env_s": d.get("latency_env_s", 0.0),
            "format_retries": d.get("format_retries", 0),
            "screen": screen,
            "has_shot": idx in self.shots,
        })

    def log_trajectory(self, trajectory: Trajectory) -> None:
        self.q.put({
            "type": "done",
            "success": trajectory.success,
            "abort_reason": trajectory.meta.get("abort_reason", ""),
            "summary": trajectory.summary(),
        })


# --------------------------------------------------------------------------
# 运行管理
# --------------------------------------------------------------------------


@dataclass
class Run:
    run_id: str
    tracer: StreamTracer
    thread: threading.Thread
    error: str = ""


_RUNS: dict[str, Run] = {}
_LOCK = threading.Lock()
"""一次只允许跑一个任务。

模拟器是**独占资源**：两个任务同时抢 adb，点击会互相打断，
轨迹全废。与其做排队，不如直接拒绝——这个工具是给人盯着看的，
本来就该一次跑一个。
"""


def _run_task(run_id: str, cfg_path: str, instruction: str, task_name: str,
              max_steps: int, capture_image: bool) -> None:
    """在后台线程里跑完一个任务。异常都转成事件推给前端，不往外抛。

    `task_name` 为空表示**自由对话**——用户直接说一句要求，模型照着做，
    没有任务定义、也没有程序化判分（无从判起：判分函数是每个任务自带的一小段
    代码，自由输入没有对应的那段）。

    这时候前端只报"模型自己说完成了没有"，并明确标出**这不等于成功**——
    否则用户会以为那条绿勾是程序化验证过的。
    """
    tracer = _RUNS[run_id].tracer
    adb = find_adb()

    def emit_error(msg: str) -> None:
        tracer.q.put({"type": "error", "message": msg})

    try:
        tasks = collect_tasks(adb, "emulator-5554")
        task = tasks.get(task_name) if task_name else None

        # 环境自检：网络坏了的话浏览器类任务必挂，而且**不会报错**，
        # 只会表现为"模型不会"。堵在这里，别让它污染演示。
        # ⚠️ 网络自检**只对联网任务硬拦**，其余任务降级成提示。
        #
        # 踩过：闸门接成硬拦之后，网络一坏，连"设闹钟"这种根本不需要网的
        # 任务也跑不了。自检的目的是**防止把环境故障记成模型不会**，
        # 不是禁止一切。
        net_ok, net_why = check_network(adb, "emulator-5554")
        needs_net = bool(task and task.package and
                         any(k in task.package for k in ("chrome", "browser")))
        if not net_ok and needs_net:
            emit_error(f"这个任务需要联网，但环境自检不通过：{net_why}")
            tracer.q.put({"type": "done", "success": False, "summary": {}})
            return
        if not net_ok:
            emit_error(f"网络不可用（{net_why}）。本任务不依赖网络，继续跑——"
                       "但浏览器类任务现在会失败，别算成模型的问题。")

        import yaml

        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        policy = build("policy", cfg["policy"])

        # 前置状态：AndroidWorld 的任务自带 initialize_task（建联系人、
        # 铺文件、设设备时间……），我们的任务没有这步。**必须在建 env 之后、
        # 跑 agent 之前做**——它是任务的一部分，不是环境的一部分。
        if task is not None and task.setup is not None:
            try:
                task.setup()
            except Exception as e:  # noqa: BLE001
                emit_error(f"任务前置状态建立失败：{type(e).__name__}: {e}")

        env = AndroidEnv(
            serial="emulator-5554",
            adb=adb,
            task_package=task.package if task else "",
            apps=load_apps(),
            # 演示时开截图：一图胜千言。代价是每步多 ~1.9s。
            capture_image=capture_image,
        )
        toolkit = Toolkit().add(NoteTool()).add(FinishTool())
        agent = Agent(env, policy, toolkit=toolkit, tracer=tracer, max_steps=max_steps)

        tracer.q.put({"type": "start", "instruction": instruction,
                      "task": task_name, "max_steps": max_steps})
        agent.run(instruction)

        # 判分：读设备真实状态，不是问模型。
        #
        # ⚠️ 自由对话（task is None）**没有判分**，而且要明确告诉前端这一点：
        #   程序化判分是每个任务自带的一小段代码，自由输入没有对应的那段。
        #   这时候只能说"模型自己认为完成了"，**那不等于成功**。
        #   前端必须把两者画得不一样，否则用户会把模型的自我申报当成验证过的结论。
        if task is not None:
            time.sleep(1.5)   # 等界面稳定
            try:
                success = bool(task.check()) if task.check else False
            except Exception as e:  # noqa: BLE001
                success = False
                emit_error(f"判分函数抛异常：{type(e).__name__}: {e}")
            if task.teardown is not None:
                try:
                    task.teardown()
                except Exception:  # noqa: BLE001
                    pass
            tracer.q.put({"type": "verdict", "success": success,
                          "check": task_name, "verified": True})
        else:
            tracer.q.put({"type": "verdict", "success": None,
                          "check": "", "verified": False,
                          "note": "自由指令没有程序化判分，下面显示的是模型自己"
                                  "认为完成了，不等于验证通过。"})
    except Exception as e:  # noqa: BLE001 - 前台工具，任何异常都要变成看得见的消息
        emit_error(f"{type(e).__name__}: {e}")
        tracer.q.put({"type": "traceback", "text": traceback.format_exc()[-1500:]})
        tracer.q.put({"type": "done", "success": False, "summary": {}})
    finally:
        tracer.q.put({"type": "__close__"})


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # 别把日志刷到 stdout 上
        pass

    # ---- 工具 ----

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # ---- GET ----

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            f = ROOT / "web" / "index.html"
            if not f.exists():
                self._send(404, b"web/index.html not found", "text/plain; charset=utf-8")
                return
            self._send(200, f.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/api/tasks":
            try:
                tasks = collect_tasks(find_adb(), "emulator-5554")
                self._json({"tasks": [
                    {"name": t.key, "label": t.name, "source": t.source,
                     "instruction": t.instruction, "steps_hint": list(t.steps_hint)}
                    for t in tasks.values() if t.source != "error"
                ]})
            except Exception as e:  # noqa: BLE001
                self._json({"tasks": [], "error": f"{type(e).__name__}: {e}"})
            return

        if path.startswith("/api/shot/"):
            parts = path.split("/")
            run_id, idx = parts[3], int(parts[4])
            run = _RUNS.get(run_id)
            img = run.tracer.shots.get(idx) if run else None
            if not img:
                self._send(404, b"no shot", "text/plain")
                return
            self._send(200, img, "image/png")
            return

        if path.startswith("/api/stream/"):
            self._stream(path.rsplit("/", 1)[-1])
            return

        self._send(404, b"not found", "text/plain")

    def _stream(self, run_id: str) -> None:
        run = _RUNS.get(run_id)
        if run is None:
            self._send(404, b"unknown run", "text/plain")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        while True:
            try:
                ev = run.tracer.q.get(timeout=60)
            except queue.Empty:
                # 心跳：没有它，中间的代理/浏览器会以为连接死了把流掐掉
                try:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                continue

            if ev.get("type") == "__close__":
                try:
                    self.wfile.write(b"event: close\ndata: {}\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            try:
                payload = json.dumps(ev, ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return   # 前端关了页面，正常退出

    # ---- POST ----

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        # 两个入口，同一套执行逻辑：
        #   /api/run   跑评测任务（指令从任务定义取，有程序化判分）
        #   /api/chat  自由对话（用户说啥就是啥，没有判分）
        if path not in ("/api/run", "/api/chat"):
            self._send(404, b"not found", "text/plain")
            return

        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "请求体不是合法 JSON"}, 400)
            return

        with _LOCK:
            busy = [r for r in _RUNS.values() if r.thread.is_alive()]
            if busy:
                self._json({"error": "已经有一个任务在跑了。"
                                     "模拟器是独占资源，同时跑两个会互相打断。"}, 409)
                return

            task_name = body.get("task", "")
            message = (body.get("message") or "").strip()
            max_steps = int(body.get("max_steps", 12))
            cfg_path = body.get("config", "configs/android_cascade_repeat.yaml")
            capture = bool(body.get("capture_image", True))

            run_id = uuid.uuid4().hex[:12]
            tracer = StreamTracer(run_id=run_id)

            if message:
                # 自由对话：用户直接下要求，没有任务定义、没有判分
                instruction = message
                task_name = ""
            else:
                # 跑评测任务：**指令必须从任务定义里取**，
                # 不能让前端自由发挥——指令和判分是成对的，
                # 改一个字，判分判的就不是同一件事了。
                try:
                    tasks = collect_tasks(find_adb(), "emulator-5554")
                except Exception as e:  # noqa: BLE001
                    self._json({"error": f"读不到任务定义：{e}"}, 500)
                    return
                if task_name not in tasks:
                    self._json({"error": f"没有名为 {task_name!r} 的任务"}, 400)
                    return
                instruction = tasks[task_name].instruction

            t = threading.Thread(
                target=_run_task,
                args=(run_id, cfg_path, instruction, task_name, max_steps, capture),
                daemon=True,
            )
            _RUNS[run_id] = Run(run_id=run_id, tracer=tracer, thread=t)
            t.start()

        self._json({"run_id": run_id, "instruction": instruction})


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  Agent 前端已启动：  http://{args.host}:{args.port}\n")
    print("   Ctrl+C 停止\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
