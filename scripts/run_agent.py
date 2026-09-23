"""运行入口：从配置构造一个完整的 Agent 并跑任务。

    python scripts/run_agent.py --config configs/single_small.yaml \
                                --task "把闹钟设到明早 7 点" \
                                --trace results/small.jsonl

三组对照只差一个 `--config`：

    --config configs/single_large.yaml   上界：全程大模型
    --config configs/single_small.yaml   下界：全程小模型
    --config configs/cascade.yaml        本项目：级联

**三者走的是同一段代码、同一个主循环、同一份日志格式**，所以结果表直接可比。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接 `python scripts/run_agent.py` 而不用先 pip install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guicascade.agent import Agent  # noqa: E402
from guicascade.registry import build  # noqa: E402
from guicascade.tools import FinishTool, NoteTool, Toolkit  # noqa: E402
from guicascade.trace import Tracer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="跑一个 GUI 任务")
    ap.add_argument("--config", required=True, help="策略配置文件")
    ap.add_argument("--task", required=True, help="任务描述")
    ap.add_argument("--env", default="android", help="环境名（registry 里登记的）")
    ap.add_argument("--env-config", default="{}", help="环境的 JSON 配置")
    ap.add_argument("--serial", default="emulator-5554", help="安卓设备序列号")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--trace", default="", help="轨迹输出路径（JSONL）")
    ap.add_argument("--no-tools", action="store_true", help="不挂通用工具，只留界面动作")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    policy = build("policy", cfg["policy"])

    env_cfg = dict(__import__("json").loads(args.env_config))
    if args.env == "android":
        env_cfg.setdefault("serial", args.serial)
    env = build("env", {"type": args.env, **env_cfg})

    # 通用工具：不改变屏幕，和界面动作分开
    toolkit = Toolkit() if args.no_tools else Toolkit().add(NoteTool()).add(FinishTool())

    tracer = Tracer(args.trace) if args.trace else None
    agent = Agent(env, policy, toolkit=toolkit, tracer=tracer, max_steps=args.max_steps)

    print(f"任务 : {args.task}")
    print(f"配置 : {args.config}")
    print(f"环境 : {args.env}")
    print("-" * 68)

    traj = agent.run(args.task)

    print("-" * 68)
    for s in traj.steps:
        sig = " ".join(f"{k}={v:.2f}" for k, v in s.decision.signals.items())
        flag = "⬆" if s.escalated else " "
        print(f"  {s.index:>3}{flag} [{s.model:<5}] {str(s.decision.action)[:44]:<46} {sig}")

    print("-" * 68)
    summary = traj.summary()
    print(f"  步数        : {summary['steps']}")
    print(f"  成功        : {traj.reward}")
    print(f"  强模型占比  : {summary['escalation_rate']:.1%}")
    print(f"  模型侧耗时  : {summary['latency_model_s']:.2f}s   ← 级联真正省掉的")
    print(f"  环境侧耗时  : {summary['latency_env_s']:.2f}s   ← 一分都省不掉")
    if args.trace:
        print(f"  轨迹已写入  : {args.trace}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
