"""批量跑任务、收集轨迹，存成监控器能直接吃的格式。

## 为什么要收集

1. **给监控器当测试集**。之前所有评测都建立在别人的数据上（AgentNet、
   EvoCUAFeedback），而且那些数据里几乎没有真正的卡死轨迹。自己跑出来的
   轨迹是**这个项目自己的分布**——小模型在安卓上真的会卡，这是免费的、
   带真实失败标签的数据。

2. **以后要自己训监控器时的训练集**。149M 基座 + 几百条轨迹就够，
   不需要公开数据集。

## 输出格式

刻意存成与参考实现一致的 `traj.jsonl`（`step_num` / `action` / `response`），
这样监控器评测脚本不用改就能读：

    results/<run>/<task>/traj.jsonl
    results/<run>/<task>/result.txt      0 或 1
    results/<run>/summary.jsonl          每任务一行

另外**自动标出疑似卡住的步**（连续重复动作），存进 summary——
省得事后靠人眼从几千行里找。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guicascade.agent import Agent  # noqa: E402
from guicascade.envs.android import AndroidEnv, find_adb, load_apps  # noqa: E402
from guicascade.registry import build  # noqa: E402
from guicascade.tools import FinishTool, NoteTool, Toolkit  # noqa: E402

from android_tasks import build_tasks, check_network, cleanup_device  # noqa: E402


def max_repeat_run(actions: list[str]) -> int:
    """连续相同动作的最大长度。这是"卡住"的独立判据。"""
    best = cur = 0
    prev = None
    for a in actions:
        cur = cur + 1 if a == prev else 1
        prev = a
        best = max(best, cur)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks", default="", help="逗号分隔；留空=全部")
    ap.add_argument("--repeat", type=int, default=1, help="每个任务跑几次（模型有随机性）")
    ap.add_argument("--serial", default="emulator-5554")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--out", default="results/collect")
    ap.add_argument("--no-image", action="store_true")
    args = ap.parse_args()

    import yaml

    adb = find_adb()
    # 网络自检要在**任何任务开跑之前**做。这条失败链上没有一步会报错：
    # 网络坏了 -> 任务做不成 -> 判分记成"模型不会"。堵在这里，别让它进数据。
    ok_net, why = check_network(adb, args.serial)
    if not ok_net:
        print("")
        print("❌ 环境自检不通过：" + why)
        print("   网络不通的话，浏览器那类任务必挂，这种成绩不能算数。")
        print("   先修网络再跑。常见修法：")
        print("     adb shell svc wifi disable && adb shell svc wifi enable")
        print("     adb root && adb shell setprop net.dns1 8.8.8.8")
        print("")
        return 2
    print("")
    print("✅ 环境自检通过：" + why)
    print("")

    all_tasks = build_tasks(adb, args.serial)
    names = [n.strip() for n in args.tasks.split(",") if n.strip()] or list(all_tasks)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    run_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"配置 : {args.config}")
    print(f"任务 : {len(names)} 个 x {args.repeat} 次")
    print(f"输出 : {run_dir}")
    print("=" * 76)

    summaries = []
    for name in names:
        task = all_tasks[name]
        for rep in range(args.repeat):
            tag = f"{name}" + (f"_r{rep}" if args.repeat > 1 else "")
            policy = build("policy", cfg["policy"])       # 每个 episode 重建，状态归零
            env = AndroidEnv(serial=args.serial, adb=adb, task_package=task.package,
                             apps=load_apps(), capture_image=not args.no_image)
            toolkit = Toolkit().add(NoteTool()).add(FinishTool())
            agent = Agent(env, policy, toolkit=toolkit, max_steps=args.max_steps)

            t0 = time.time()
            traj = agent.run(task.instruction)
            wall = time.time() - t0
            time.sleep(1.5)                                # 等界面稳定再判分
            success = bool(task.check())

            # 存成监控器能直接读的格式
            d = run_dir / tag
            d.mkdir(parents=True, exist_ok=True)
            with (d / "traj.jsonl").open("w", encoding="utf-8") as f:
                for s in traj.steps:
                    f.write(json.dumps({
                        "step_num": s.index + 1,
                        "action": str(s.decision.action),
                        "response": s.decision.reason,
                        "model": s.decision.model,
                        "escalated": s.escalated,
                        **{k: round(v, 4) for k, v in s.decision.signals.items()},
                    }, ensure_ascii=False) + "\n")
            (d / "result.txt").write_text("1" if success else "0", encoding="utf-8")

            actions = [str(s.decision.action) for s in traj.steps]
            longest = max_repeat_run(actions)
            summary = {
                "task": name, "repeat": rep, "success": success,
                "steps": len(traj.steps), "wall_s": round(wall, 1),
                "model_s": round(traj.model_latency_s, 1),
                "env_s": round(traj.env_latency_s, 1),
                "escalation_rate": round(traj.escalation_rate, 3),
                "longest_repeat": longest,
                # 连续 >=3 步同一动作 = 疑似卡住（与监控器评测里的判据一致）
                "stuck": longest >= 3,
                "top_action": Counter(actions).most_common(1)[0][0] if actions else "",
            }
            summaries.append(summary)

            # 每个任务跑完立刻擦痕迹。**不能等整批跑完再清**——下一个任务
            # 一开始就要从干净状态出发，否则会读到上一个任务留下的数据。
            # 踩过：飞行模式任务把网断了，三个任务之后 Chrome 那条必然失败。
            cleanup_device(adb, args.serial)

            flag = "✅" if success else "❌"
            stuck = f"  卡住({longest}连)" if summary["stuck"] else ""
            print(f"  {flag} {tag:<22} {len(traj.steps):>2}步  "
                  f"{wall:>5.1f}s  模型{traj.model_latency_s:>5.1f}s  "
                  f"环境{traj.env_latency_s:>5.1f}s{stuck}")

    with (run_dir / "summary.jsonl").open("w", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_ok = sum(s["success"] for s in summaries)
    n_stuck = sum(s["stuck"] for s in summaries)
    print("=" * 76)
    print(f"  成功 {n_ok}/{len(summaries)}   卡住 {n_stuck}/{len(summaries)}")
    print(f"  汇总: {run_dir / 'summary.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
