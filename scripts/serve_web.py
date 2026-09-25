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


def _shrink(png: bytes, width: int = 420) -> bytes:
    """把截图缩小再存盘。

    ## 为什么必须缩

    设备是 1080×2400，原图 PNG 一张 **~1.4MB**。前端显示宽度只有 104px
    （点开放大也远用不到原图）。不缩的话：

        跑一夜 ≈ 2009 步/形态 × 3 形态 × 1.4MB ≈ 8.4 GB

    这不只是占地方——**写盘本身会把跑测拖慢**，而且磁盘满了会让整轮
    任务静默失败。缩到 420px 宽之后约 40KB，同样的量级降到 250MB。

    用 JPEG 不用 PNG：截图是大色块界面，JPEG 在同样观感下小一个量级。
    缩放失败就原样返回——**降级而不是抛异常**，一张图不该让整轮跑挂掉。
    """
    try:
        import io

        from PIL import Image

        im = Image.open(io.BytesIO(png)).convert("RGB")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)),
                           Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=72, optimize=True)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return png


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
    events: list = field(default_factory=list)
    verdict: dict | None = None
    stop_requested: bool = False
    """前端点了「停止」。下一拍就中断这一轮。

    ⚠️ 用**抛异常**实现而不是加个 if：`Agent` 的主循环里对异常的处理
    已经写好了（判负、关环境、记原因），复用它比自己再穿一条停止信号
    干净得多，也不用为了"能停"去改主循环的签名。"""
    """这一轮的**全部事件副本**，给写记录用。

    ⚠️ 早先写记录时是去 `q` 里 `get_nowait()` 抽干的，**那是偷 SSE 的事件**——
    记完账，前端那条流就再也收不到东西了。队列是"发出去"的通道，
    要留底就该自己留一份，不该消费别人的队列。"""

    def log_step(self, task: str, step) -> None:
        if self.stop_requested:
            raise RuntimeError("用户中止")
        d = step.to_dict()
        idx = int(d.get("step", 0))

        # 截图：**内存一份 + 磁盘一份**。
        #
        # 内存那份给正在看的人（快），磁盘那份给明天来翻记录的人（留得住）。
        # 只留内存的话，服务器一重启，历史记录里所有截图都变成裂图，
        # 而 DOM 文本还在——看起来像"截图功能坏了"，实际上是没存。
        img = getattr(step.observation, "image", None)
        if img:
            self.shots[idx] = img
            try:
                # ⚠️ 变量名**不能叫 `d`**：上面 `d = step.to_dict()` 已经用了，
                # 覆盖掉之后下面 `d.get("model")` 就变成对 Path 调 .get，
                # 每一步都抛 AttributeError，**整轮任务 0 步就死**。
                # 症状是"所有任务都跑不起来、只有 20 秒"，很难联想到是
                # 一个局部变量名。加截图功能时就是这么把自己绊倒的。
                shot_dir = _SHOT_DIR / self.run_id
                shot_dir.mkdir(parents=True, exist_ok=True)
                (shot_dir / f"{idx}.jpg").write_bytes(_shrink(img))
            except OSError:
                pass      # 写盘失败不该把这一轮跑挂掉，内存那份还在

        # 给模型看的屏幕文本也带上——**这是排查问题最有用的一栏**：
        # 模型看到什么、说了什么，并排看才知道它为什么那样决策。
        screen = (getattr(step.observation, "text", "") or "")[:4000]

        ev = {
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
        }
        self.events.append(ev)
        self.q.put(ev)

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
    started: float = 0.0
    done: bool = False
    """线程真正退出时由它自己在 finally 里置 True。

    ⚠️ 不要用 `thread.is_alive()` 判"跑完没有"。线程可能因为底层 adb 调用挂死
    而**永远活着**，那样服务就永久占死了——实测踩过：后面每个 /api/run
    都被 409 挡掉，60 轮跑测全废。让线程自己报"我结束了"才准。
    """


_RUNS: dict[str, Run] = {}
_LOCK = threading.Lock()
_CURRENT: list = [None]
"""当前在跑的那一个。`None` 表示空闲。

**显式记录，而不是靠线程存活推断**——见 /api/run 里的说明。"""
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
        tasks = _tasks_cached(adb, "emulator-5554")
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
        print(f"[run {run_id}] 开始（{task_name or '自由对话'}）")
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
                # ⚠️ 判分必须带超时。AndroidWorld 的判分里有 adb 调用，
                # 设备卡住时它会**一直挂着**——线程不结束，`_RUNS` 里就永远
                # 有个"活着"的线程，后面每个 /api/run 都被 409 挡掉。
                # 实测踩过：20 个任务里 19 个"起不来"，全卡在这。
                success = _call_with_timeout(task.check, 120) if task.check else False
            except Exception as e:  # noqa: BLE001
                success = False
                emit_error(f"判分函数抛异常：{type(e).__name__}: {e}")
            if task.teardown is not None:
                _call_with_timeout(task.teardown, 60)

            v = {"type": "verdict", "success": success,
                 "check": task_name, "verified": True}
            tracer.verdict = v
            tracer.events.append(v)
            tracer.q.put(v)
        else:
            v = {"type": "verdict", "success": None, "check": "", "verified": False,
                 "note": "自由指令没有程序化判分，下面显示的是模型自己"
                         "认为完成了，不等于验证通过。"}
            tracer.verdict = v
            tracer.events.append(v)
            tracer.q.put(v)
    except Exception as e:  # noqa: BLE001 - 前台工具，任何异常都要变成看得见的消息
        emit_error(f"{type(e).__name__}: {e}")
        tracer.q.put({"type": "traceback", "text": traceback.format_exc()[-1500:]})
        tracer.q.put({"type": "done", "success": False, "summary": {}})
    finally:
        # 记录落到**服务端**，不是浏览器 localStorage。
        #
        # 理由：批量跑一轮要一两个小时，用 CLI 驱动（浏览器关掉也能跑）。
        # 记录只存浏览器的话，CLI 跑出来的结果前端看不见，两边就成了两套数据。
        # 存服务端则**两边看的是同一份**，谁跑的都能在页面上看到。
        #
        # ⚠️ **必须放在 finally 里，不能只放在 except 里。**
        #
        # 早先它挂在 `except` 分支下面，于是**只有崩掉的任务才会被记录**，
        # 正常跑完的（不管成功还是失败）全部不落盘。这个 bug 极其安静：
        # 单跑一个任务时你会去看终端输出，感觉「记录功能是好的」；
        # 只有整批跑完、回头翻记录，才会发现**只有异常的那几条**。
        # 一夜跑下来前端一片空白，而日志里什么错都没有。
        _append_record(cfg_path, task_name, instruction, tracer)

        # 释放锁：**必须在 finally 里**，否则异常路径会把服务永久锁死
        if _CURRENT[0] and _CURRENT[0].get("run_id") == run_id:
            _CURRENT[0] = None
        tracer.q.put({"type": "__close__"})



def _call_with_timeout(fn, seconds: float):
    """跑 `fn()`，超时就放弃并返回 False。

    为什么不用 signal/async：这是在**后台线程**里跑的，signal 只对主线程有效。
    判分卡死的后果不是"这个任务失败"，而是**整个服务再也接不了新任务**
    （线程一直活着 -> 忙判断一直为真 -> 后面全部 409）。所以宁可放弃这一次判分。
    """
    box: dict = {}

    def run():
        try:
            box["v"] = fn()
        except Exception as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        print(f"[timeout] 判分/清理超过 {seconds}s，放弃")
        return False
    if "e" in box:
        raise box["e"]
    return box.get("v", False)



# --------------------------------------------------------------------------
# 任务表缓存
# --------------------------------------------------------------------------

_TASK_CACHE: dict = {"at": 0.0, "tasks": {}}
_TASK_TTL = 300.0


def _tasks_cached(adb: str, serial: str) -> dict:
    """任务表**必须缓存**。

    每次 `collect_tasks()` 都要 import 整个 android_world 包（加载 20 个任务类
    + 编译好的 proto），实测要 30~60 秒。

    踩过的坑：`/api/run` 为了校验任务名，每个请求都重新加载一次。
    批量驱动那边 curl 的 `--max-time` 是 60 秒，于是**请求超时 -> 拿不到
    run_id -> 被当成"服务忙" -> 无限重试**。表现是 20 个任务全部"起不来"，
    而服务端其实是好的、日志里一条错误都没有。

    **一个只读的、30 秒才能算出来的表，没有任何理由每个请求重算一次。**
    """
    now = time.time()
    if not _TASK_CACHE["tasks"] or (now - _TASK_CACHE["at"]) > _TASK_TTL:
        _TASK_CACHE["tasks"] = collect_tasks(adb, serial)
        _TASK_CACHE["at"] = now
    return _TASK_CACHE["tasks"]


# --------------------------------------------------------------------------
# 测试记录（服务端）
# --------------------------------------------------------------------------

_RECORDS = ROOT / "results" / "web_records.jsonl"
_SHOT_DIR = ROOT / "results" / "shots"
"""每一步的截图落盘位置：`results/shots/<run_id>/<步号>.png`。

放 `results/` 下是因为**跑测的产物都该在一处**，找的时候不用满盘翻。
一张约 50KB，一轮 59 任务 × 平均 20 步 ≈ 60MB，三形态不到 200MB，可以接受。"""
"""一行一轮，跑完即落盘。**追加写**，所以中途挂了也不丢已完成的。"""

_CONFIG_KEY = {
    "android_small": "small",
    "android_large": "large",
    "android_cascade_repeat": "cascade",
    "android_cascade": "cascade",
}


def _config_key(path: str) -> str:
    """从配置文件名推出配置标识。前端按它分三栏。"""
    stem = Path(path).stem
    return _CONFIG_KEY.get(stem, stem)


def _append_record(cfg_path: str, task_name: str, instruction: str, tracer) -> None:
    """把这一轮的步骤事件和判分结果写到 records 文件。

    存**原始事件**而不是渲染好的 HTML：前端本来就有一套 `stepHtml()`，
    让它自己渲染，两边样式才不会各写一份、各错各的。
    """
    try:
        evs = list(tracer.events)        # 副本，**不动队列**
        verdict = tracer.verdict
        steps = [e for e in evs if e.get("type") == "step"]
        rec = {
            "task": task_name or "(自由对话)",
            "instruction": instruction,
            "config": _config_key(cfg_path),
            "ok": (verdict or {}).get("success"),
            "verified": (verdict or {}).get("verified", False),
            "steps": len(steps),
            "escalated": sum(1 for e in steps if e.get("escalated")),
            "at": time.time(),
            "errors": [e.get("message", "")[:200] for e in evs if e.get("type") == "error"][:3],
            "events": steps,
        }
        _RECORDS.parent.mkdir(parents=True, exist_ok=True)
        with _RECORDS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + chr(10))
    except Exception as e:  # noqa: BLE001 - 记不上不该影响跑测
        print(f"[records] 写记录失败：{type(e).__name__}: {e}")


def _read_records() -> list[dict]:
    if not _RECORDS.exists():
        return []
    out = []
    for line in _RECORDS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out[-400:]      # 只回最近 400 条，页面用不着更老的


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

        if path == "/api/records":
            self._json({"records": _read_records()})
            return

        if path == "/api/tasks":
            try:
                tasks = _tasks_cached(find_adb(), "emulator-5554")
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

            # 先看内存，再回落到磁盘。
            #
            # ⚠️ **必须落盘，不能只放内存。** 截图原先只存在 `run.tracer.shots`
            # 里，服务器一重启就全没了——前端翻历史记录时**图全是裂的**，
            # 而 DOM 文本还在，看起来像"截图功能坏了"。
            #
            # 这个项目里"看到模型当时看到的画面"是排查失败原因最有用的一栏，
            # 丢了它，失败就只能靠猜。**跑一夜的图必须留得住。**
            # 内存里那份是原图 PNG（这一轮正在看的），磁盘那份是缩过的 JPEG。
            # 两处格式不同，**content-type 要跟着变**，否则浏览器可能按错格式解。
            img = run.tracer.shots.get(idx) if run else None
            if img:
                self._send(200, img, "image/png")
                return
            # 两种后缀都找：`.jpg` 是压缩后的（现在的默认），`.png` 是早期
            # 没压缩时留下的原图。**老数据也要能看**——不然改一次格式，
            # 之前跑的所有记录截图全变裂图，而记录本身还在，很难解释。
            for name, ctype in ((f"{idx}.jpg", "image/jpeg"),
                                (f"{idx}.png", "image/png")):
                p = _SHOT_DIR / run_id / name
                if p.is_file():
                    self._send(200, p.read_bytes(), ctype)
                    return
            self._send(404, b"no shot", "text/plain")
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
                # ⚠️ **必须显式关连接。** 这一行是踩了很久才补上的。
                #
                # 上面的 `Connection: keep-alive` + HTTP/1.1 意味着：
                # handler 返回后 socket **不关**，服务器准备在同一根连接上
                # 接下一个请求。对普通接口这是对的，但对 SSE 是致命的——
                # 客户端等的是 EOF，而 EOF 永远不会来。
                #
                # 症状极具误导性：任务其实 34 秒就跑完了，事件也全发到了，
                # 但 `curl --max-time 900` 会**每一轮都耗满 900 秒**。
                # 75 个任务算下来是 19 小时，而不是 2 小时。
                # 光看总时长，你会以为是"任务太慢"或"模型太慢"，
                # 而真相是流没关——**这两种情况的修法完全相反**。
                self.close_connection = True
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
        if path == "/api/stop":
            stopped = False
            with _LOCK:
                cur = _CURRENT[0]
                if cur is not None:
                    run = _RUNS.get(cur["run_id"])
                    if run is not None:
                        run.tracer.stop_requested = True
                        stopped = True
                    _CURRENT[0] = None      # 立刻放锁，不等线程收尾
            self._json({"stopped": stopped})
            return

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
            # ⚠️ **不用"线程活没活"判断忙闲。**
            #
            # 踩过两次：底层 adb 挂死时线程永远活着，服务就**永久卡住**，
            # 后面每一个 /api/run 都被 409 挡掉（实测 20 个任务 19 个"起不来"）。
            #
            # 改成显式的时间戳锁：谁开的、什么时候开的，一目了然；
            # 超过 RUN_TIMEOUT 直接强制释放——**跑测工具的可恢复性
            # 比"绝不误杀"更重要**，误杀的代价是重跑一个任务，
            # 卡死的代价是整轮实验作废。
            RUN_TIMEOUT = 60 * 5
            now = time.time()
            cur = _CURRENT[0]
            if cur is not None and (now - cur["started"]) > RUN_TIMEOUT:
                print(f"[lock] 上一个任务 {cur['run_id']} 已跑 "
                      f"{(now-cur['started'])/60:.1f} 分钟，强制释放")
                _CURRENT[0] = None
                cur = None
            if cur is not None:
                self._json({"error": "已经有一个任务在跑了（自 "
                                     f"{(now-cur['started'])/60:.1f} 分钟前开始）。"
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
                    tasks = _tasks_cached(find_adb(), "emulator-5554")
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
            _RUNS[run_id] = Run(run_id=run_id, tracer=tracer, thread=t,
                                started=time.time())
            _CURRENT[0] = {"run_id": run_id, "started": time.time()}
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
