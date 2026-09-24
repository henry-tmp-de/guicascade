"""批量跑一组任务，输出汇总。驱动的是**和前端同一套 HTTP 接口**。

## 为什么要单独有一个 CLI

前端是给人看的：点一下跑一个，边跑边看。但"把 20 个任务全跑一遍"这件事
需要**无人值守跑一小时**——浏览器关掉就断了。

这个脚本走同一个 `/api/run` + `/api/stream`，所以：

- 结果和前端**必然一致**（同一个后端、同一套判分）
- 两边的统计口径不会漂

## 两组分开跑，汇总也分开报

`--group ours` 跑我们自己的 6 个，`--group androidworld` 跑那 20 个。

**不合并总数**是有意的：两边判分的可信度不一样（AndroidWorld 的判分是别人
在真机上反复调过的，我们的翻过四次车）。合成一个数会掩盖差异。

## 用法

    python scripts/run_suite.py --group androidworld
    python scripts/run_suite.py --group ours --config configs/android_large.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

__all__ = ["main"]

BASE = "http://127.0.0.1:8765"


def post(path: str, payload: dict, timeout: float = 60) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    # 显式绕过代理：本机回环地址走代理会 502（踩过）
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path: str, timeout: float = 120) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def stream_steps(run_id: str) -> dict:
    """接 SSE 直到收到 __close__，返回这一轮的汇总。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    out = {"steps": 0, "escalated": 0, "success": None, "verified": False,
           "errors": [], "summary": {}, "actions": []}
    with opener.open(f"{BASE}/api/stream/{run_id}", timeout=1800) as r:
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
                out["verified"] = bool(ev.get("verified"))
            elif t == "error":
                out["errors"].append(ev.get("message", "")[:160])
            elif t == "done":
                out["summary"] = ev.get("summary") or {}
            elif t == "__close__":
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["ours", "androidworld"])
    ap.add_argument("--config", default="configs/android_cascade_repeat.yaml")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--out", default="", help="结果写到这里（JSON）")
    args = ap.parse_args()

    try:
        tasks = get("/api/tasks").get("tasks", [])
    except Exception as e:  # noqa: BLE001
        print(f"❌ 连不上前端服务（{BASE}）：{e}")
        print("   先在另一个终端跑：python scripts/serve_web.py")
        return 2

    mine = [t for t in tasks if t.get("source") == args.group]
    if not mine:
        print(f"❌ 没有 source={args.group} 的任务")
        return 2

    title = "我们的任务" if args.group == "ours" else "AndroidWorld"
    print(f"\n{'='*92}")
    print(f"  {title} · {len(mine)} 个任务")
    print(f"  配置 {args.config} · 步数上限 {args.max_steps}")
    print(f"{'='*92}\n")

    results = []
    t_all = time.time()

    for i, t in enumerate(mine, 1):
        name, label = t["name"], t.get("label", t["name"])
        print(f"  [{i:>2}/{len(mine)}] {label:<34} {t['instruction'][:44]}")
        t0 = time.time()
        try:
            r = post("/api/run", {"task": name, "config": args.config,
                                  "max_steps": args.max_steps, "capture_image": False})
            if r.get("error"):
                print(f"          ⚠️ 起不来：{r['error'][:90]}")
                results.append({"name": name, "label": label, "ok": None,
                                "note": r["error"][:200]})
                continue
            s = stream_steps(r["run_id"])
        except Exception as e:  # noqa: BLE001
            print(f"          ⚠️ {type(e).__name__}: {str(e)[:90]}")
            results.append({"name": name, "label": label, "ok": None,
                            "note": f"{type(e).__name__}: {e}"[:200]})
            continue

        dt = time.time() - t0
        ok = s["success"]
        mark = "✅" if ok else ("❌" if ok is False else "⚠️")
        esc = f"  强模型 {s['escalated']}/{s['steps']}" if s["steps"] else ""
        print(f"          {mark} {s['steps']:>2} 步{esc}  {dt:.0f}s")
        for e in s["errors"][:2]:
            print(f"          · {e}")
        results.append({
            "name": name, "label": label, "ok": ok, "steps": s["steps"],
            "escalated": s["escalated"], "seconds": round(dt, 1),
            "errors": s["errors"][:3], "actions": s["actions"],
        })

    # ---- 汇总 ----
    done = [r for r in results if r.get("ok") is not None]
    ok = [r for r in done if r["ok"]]
    steps = sum(r.get("steps", 0) for r in results)
    esc = sum(r.get("escalated", 0) for r in results)

    print(f"\n{'='*92}")
    print(f"  {title} 汇总")
    print(f"{'='*92}")
    print(f"  完成 {len(done)}/{len(mine)}    成功 {len(ok)}/{len(done) if done else 0}"
          f"    成功率 {round(len(ok)/len(done)*100) if done else 0}%")
    if steps:
        print(f"  总步数 {steps}    平均 {steps/max(len(results),1):.1f} 步/任务"
              f"    强模型占比 {round(esc/steps*100)}%")
    if len(done) < len(mine):
        print(f"  ⚠️ 有 {len(mine)-len(done)} 个任务没跑出结果（环境问题，不算模型失败）")
    print(f"  总耗时 {(time.time()-t_all)/60:.1f} 分钟")

    failed = [r["label"] for r in done if not r["ok"]]
    if failed:
        print(f"\n  失败的任务：")
        for f in failed:
            print(f"    · {f}")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"group": args.group, "config": args.config,
                        "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n  明细已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
