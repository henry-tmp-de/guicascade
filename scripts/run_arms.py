"""三形态全量测试的驱动（纯 Python，替代 run_suite.sh）。

## 为什么不再用 shell 版

`run_suite.sh` 是 curl + 内嵌 Python 拼接出来的，出过一次**非常难查的假故障**：

    第一步 curl POST 明明成功了（服务端已经占了锁、任务真的在跑），
    但内嵌那段 Python 解析没把 run_id 传回来，脚本就当成"没起来"，
    于是重试 -> 这次真的撞 409 -> 再重试……十轮之后记一条 `start failed`。

症状是"服务忙"，真因在**解析那一层**，而它还被 `2>/dev/null` 吞掉了错误。
一个任务的成败要靠"两个进程 + 字符串拼接"对暗号，这种结构本身就容易假故障。

纯 Python 一个进程搞定：POST、读流、判分、落盘都在同一处，
**出错有 traceback，不会变成静默的空字符串**。

## 用法

    python scripts/run_arms.py                      # 两组 × 三配置，全跑
    python scripts/run_arms.py --group androidworld
    python scripts/run_arms.py --configs configs/android_small.yaml

结果：
    results/suite_<组>_<配置>.jsonl   一行一个任务，跑完即落盘
    后端 results/web_records.jsonl    由服务端自己写
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8765"


def _opener():
    """显式绕开代理：本机回环走代理会 502（踩过）。"""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get_json(path: str, timeout: float = 300) -> dict:
    with _opener().open(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def start_run(task: str, cfg: str, max_steps: int) -> str | None:
    """起一个任务。返回 run_id；服务忙返回 None；其他错误抛出来。

    ⚠️ 服务端一次只接一个任务（模拟器是独占资源）。**"忙"和"出错"要分开**：
    忙是可以等的，出错（配置路径写错、任务名不存在）等一晚上也不会好。
    早先 shell 版把两者都算成"忙"，于是一个配置写错会耗掉整轮的重试时间。
    """
    # ⚠️ `capture_image` 必须是 True。早先为省时间传了 False，
    # 后来在前端上只看得到 DOM 树、看不到屏幕截图——**排查 GUI agent
    # 最想看的就是"它当时看到什么"**，少了截图，失败原因基本靠猜。
    # 代价是每步多约 1.4 秒（一次 screencap），按 1384 步/形态算约多 30 分钟，
    # 换来的是可解释性，值。
    body = json.dumps({"task": task, "config": cfg,
                       "max_steps": max_steps, "capture_image": True}).encode()
    req = urllib.request.Request(BASE + "/api/run", data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with _opener().open(req, timeout=120) as r:
            d = json.loads(r.read().decode("utf-8"))
        if d.get("error"):
            raise RuntimeError(f"服务端拒绝：{d['error']}")
        return d.get("run_id") or None
    except urllib.error.HTTPError as e:
        if e.code == 409:
            return None                       # 忙，等一会儿再来
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}")


def stream(run_id: str, timeout: float = 1800) -> dict:
    """接 SSE 直到服务端关闭。流现在会正常关（见 serve_web 里 close_connection 的说明）。"""
    out = {"steps": 0, "escalated": 0, "success": None, "errors": [],
           "actions": [], "saw_close": False}
    with _opener().open(f"{BASE}/api/stream/{run_id}", timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            try:
                ev = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            if t == "step":
                out["steps"] += 1
                if ev.get("escalated"):
                    out["escalated"] += 1
                out["actions"].append(str(ev.get("action", ""))[:60])
            elif t == "verdict":
                out["success"] = ev.get("success")
            elif t == "error":
                out["errors"].append(str(ev.get("message", ""))[:200])
            elif t is None:
                out["saw_close"] = True       # `event: close` 的 data 是 {}
    return out


def run_one(task: str, cfg: str, max_steps: int, *, wait_busy: float = 15.0,
            max_wait: float = 900.0) -> dict:
    """跑一个任务，忙就等。返回这一轮的结果。"""
    t0 = time.time()
    run_id = None
    while time.time() - t0 < max_wait:
        run_id = start_run(task, cfg, max_steps)
        if run_id:
            break
        time.sleep(wait_busy)
    if not run_id:
        return {"name": task, "ok": None, "note": "服务一直忙，超过等待上限"}

    s = stream(run_id)
    s["seconds"] = round(time.time() - t0, 1)
    s["name"] = task
    s["max_steps"] = max_steps      # 落盘记下这一轮给了多少步，否则事后无法判断
    s["ok"] = s.pop("success")      # 一条轨迹是"被截断"还是"跑完了"
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="both",
                    choices=["both", "androidworld", "ours"])
    ap.add_argument("--configs", default="")
    # ⚠️ 步数默认**按官方给**（每个任务 `int(10*complexity)`，14~78 步），
    # 而不是一律一个数。
    #
    # 踩过：早先一律给 12，结果 57/59 个任务撞上限被截断，"成功率 0/59"
    # 主要说明的是**没让它跑完**。一律给 78 同样不对——官方给简单任务 14 步，
    # 全给 78 会让简单任务白等，而且跑不完（59×3×78×8.5s ≈ 32 小时）。
    # 按官方逐个给是唯一既对齐口径、又跑得完的做法（合计 3.3 小时/形态）。
    ap.add_argument("--max-steps", type=int, default=0,
                    help="强制统一的步数上限；不填则用每个任务的官方预算")
    ap.add_argument("--limit", type=int, default=0,
                    help="每组只跑前 N 个任务（冒烟测试用）")
    ap.add_argument("--tasks", default="",
                    help="只跑名字含这些串的任务，逗号分隔")
    ap.add_argument("--exact", action="store_true",
                    help="--tasks 按完整任务名精确匹配，而不是子串匹配")
    args = ap.parse_args()

    configs = ([c.strip() for c in args.configs.split(",") if c.strip()]
               or ["configs/android_small.yaml",
                   "configs/android_large.yaml",
                   "configs/android_cascade_repeat.yaml"])
    groups = (["androidworld", "ours"] if args.group == "both" else [args.group])

    print("=" * 78)
    print(f"  三形态全量测试   开始 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  步数上限 {args.max_steps}")
    print("=" * 78, flush=True)

    tasks_all = get_json("/api/tasks").get("tasks", [])

    for g in groups:
        mine = [t for t in tasks_all if t.get("source") == g]
        if args.tasks:
            keys = [s.strip() for s in args.tasks.split(",") if s.strip()]
            if args.exact:
                # **默认的子串匹配很容易多带任务**，实测踩过：给
                # 'MarkorCreateNote' 会连 MarkorCreateNoteAndSms /
                # MarkorCreateNoteFromClipboard 一起跑，给
                # 'RecipeAddMultipleRecipes' 会带进三个变体——
                # 结果是"选了 14 个跑出 20 个"，时间多花一倍还不自知。
                # 想精确指定一批就用 --exact。带不带 `aw:` 前缀都认。
                want = {k if k.startswith(("aw:", "ours:")) else f"aw:{k}" for k in keys}
                want |= {k.split(":", 1)[-1] for k in want}
                mine = [t for t in mine
                        if t["name"] in want or t["name"].split(":", 1)[-1] in want]
            else:
                mine = [t for t in mine if any(k in t["name"] for k in keys)]
        if args.limit:
            mine = mine[: args.limit]
        if not mine:
            print(f"  ⚠️ 没有 source={g} 的任务，跳过")
            continue
        for cfg in configs:
            key = Path(cfg).stem.replace("android_", "")
            out_path = ROOT / "results" / f"suite_{g}_{key}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("", encoding="utf-8")

            print(f"\n{'─'*78}")
            print(f"  ▶ {g} × {Path(cfg).name}   {len(mine)} 个任务")
            print(f"    写入 {out_path.name}")
            print(f"{'─'*78}", flush=True)

            for i, t in enumerate(mine, 1):
                name = t["name"]
                # 步数：命令行给了就一律用它，否则用这个任务的官方预算
                cap = args.max_steps or int((t.get("steps_hint") or [0, 20])[-1] or 20)
                print(f"  [{i:>2}/{len(mine)}] {name:<46}", end="", flush=True)
                try:
                    r = run_one(name, cfg, cap)
                except Exception as e:  # noqa: BLE001
                    r = {"name": name, "ok": None, "steps": 0,
                         "note": f"{type(e).__name__}: {e}"[:200]}
                    print(f" ⚠️ {type(e).__name__}: {str(e)[:60]}")
                else:
                    mark = "✅" if r["ok"] else ("❌" if r["ok"] is False else "⚠️")
                    print(f" {mark} {r.get('steps',0):>2}步 "
                          f"强模型 {r.get('escalated',0)}/{r.get('steps',0)} "
                          f"{r.get('seconds',0):>6.0f}s", flush=True)

                # 一行一个，跑完即落盘：中途挂了也不丢已完成的
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            done = [x for x in rows if x.get("ok") is not None]
            ok = [x for x in done if x["ok"]]
            steps = sum(x.get("steps", 0) for x in rows)
            esc = sum(x.get("escalated", 0) for x in rows)
            print(f"\n  {g} × {key} 汇总：成功 {len(ok)}/{len(done)}"
                  f"（{len(rows)-len(done)} 个没跑出结果）"
                  f"  总步数 {steps}·强模型占比 "
                  f"{round(esc/steps*100) if steps else 0}%", flush=True)

    print(f"\n{'='*78}")
    print(f"  全部跑完 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
