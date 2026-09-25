"""跑一个任务，给每个 SSE 事件打上到达时间。用来定位"时间花在哪一段"。

## 为什么需要它

跑测速的时候遇到过一个很误导的现象：

    curl --max-time 900 ... > 结果
    → 每个任务都耗满 900 秒

看起来像"任务要跑 15 分钟"。但直接跑 agent 主循环，同样 3 步只要 28 秒。
两边的差就是**包装层**——可光看总时长分不出是模型慢、环境慢、
还是流没关。

把到达时间打出来，一眼就能看出是"事件慢慢来"还是"事件早来完了、
连接却挂着"。**这两种情况的修法完全相反**，光看总时长永远分不清。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8765"


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def main() -> int:
    task = sys.argv[1] if len(sys.argv) > 1 else "aw:MarkorCreateNote"
    cfg = sys.argv[2] if len(sys.argv) > 2 else "configs/android_small.yaml"
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    req = urllib.request.Request(
        BASE + "/api/run",
        data=json.dumps({"task": task, "config": cfg,
                         "max_steps": steps, "capture_image": False}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with _opener().open(req, timeout=60) as r:
        run_id = json.loads(r.read()).get("run_id", "")
    print(f"run_id={run_id}  task={task}  steps<={steps}\n")

    t0 = time.time()
    last = t0
    with _opener().open(f"{BASE}/api/stream/{run_id}", timeout=900) as r:
        for raw in r:
            now = time.time()
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            try:
                d = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            t = d.get("type")
            gap = now - last
            last = now
            if t == "step":
                print(f"  +{now-t0:6.1f}s  (间隔 {gap:5.1f}s)  step{d.get('index')}  "
                      f"{str(d.get('action'))[:52]}  模型{d.get('latency_model_s',0):.1f}s "
                      f"环境{d.get('latency_env_s',0):.1f}s")
            else:
                print(f"  +{now-t0:6.1f}s  (间隔 {gap:5.1f}s)  [{t}]")
    print(f"\n  流结束，总耗时 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
