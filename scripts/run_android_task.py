"""在真实安卓模拟器上跑一个任务，并做程序化判分。

## 为什么判分不放在 Environment 里

`env.success()` 在这个项目里恒返回 None。原因见 `envs/android.py` 的说明：
**判分属于「任务」，不属于「环境」**。一个安卓环境不该知道"设置页打开算不算成功"。

所以判分写在这里——每个任务自带一段检查逻辑。这样同一个安卓环境能服务
任意任务集（AndroidWorld 的 116 个任务、自己编的小任务），而不需要为每个
benchmark 改环境代码。

判分用 `dumpsys` 读设备真实状态（不是让模型自己说"我完成了"）。
**程序化判分不会因为裁判模型的随机性污染实验**，这是选安卓环境的理由之一。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guicascade.agent import Agent  # noqa: E402
from guicascade.envs.android import AndroidEnv, find_adb  # noqa: E402
from guicascade.registry import build  # noqa: E402
from guicascade.tools import FinishTool, NoteTool, Toolkit  # noqa: E402
from guicascade.trace import Tracer  # noqa: E402


# --------------------------------------------------------------------------
# 任务定义：一个任务 = 一句指令 + 一段程序化判分
# --------------------------------------------------------------------------


@dataclass
class Task:
    name: str
    instruction: str
    check: Callable[[str], bool]
    package: str = ""


def _dumpsys(serial: str, adb: str) -> str:
    out = subprocess.run(
        [adb, "-s", serial, "shell", "dumpsys", "window", "displays"],
        capture_output=True, timeout=30,
    ).stdout.decode("utf-8", "replace")
    if "mCurrentFocus" not in out:
        out += subprocess.run(
            [adb, "-s", serial, "shell", "dumpsys", "activity", "activities"],
            capture_output=True, timeout=30,
        ).stdout.decode("utf-8", "replace")
    return out


def _foreground_package(serial: str, adb: str) -> str:
    text = _dumpsys(serial, adb)
    m = re.search(r"mCurrentFocus=Window\{[^}]*?\s([\w\.]+)/", text)
    if not m:
        m = re.search(r"mFocusedApp.*?\s([\w\.]+)/", text)
    return m.group(1) if m else ""


TASKS: dict[str, Callable[[str, str], Task]] = {
    "open_settings": lambda serial, adb: Task(
        name="open_settings",
        instruction="打开系统设置应用（Settings）。",
        check=lambda _: _foreground_package(serial, adb) == "com.android.settings",
        package="com.android.settings",
    ),
    "open_contacts": lambda serial, adb: Task(
        name="open_contacts",
        instruction="打开联系人应用（Contacts）。",
        check=lambda _: "contacts" in _foreground_package(serial, adb),
        package="com.android.contacts",
    ),
    "open_clock": lambda serial, adb: Task(
        name="open_clock",
        instruction="打开时钟应用（Clock）。",
        check=lambda _: "clock" in _foreground_package(serial, adb) or
                        "deskclock" in _foreground_package(serial, adb),
        package="com.android.deskclock",
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default="open_settings", choices=sorted(TASKS))
    ap.add_argument("--serial", default="emulator-5554")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--trace", default="")
    ap.add_argument("--no-image", action="store_true", help="不抓截图（快 ~1.9s/步）")
    args = ap.parse_args()

    import yaml

    adb = find_adb()
    task = TASKS[args.task](args.serial, adb)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    policy = build("policy", cfg["policy"])

    env = AndroidEnv(
        serial=args.serial,
        adb=adb,
        task_package=task.package,
        capture_image=not args.no_image,
    )
    toolkit = Toolkit().add(NoteTool()).add(FinishTool())
    tracer = Tracer(args.trace) if args.trace else None

    print(f"任务   : {task.name}")
    print(f"指令   : {task.instruction}")
    print(f"配置   : {args.config}")
    print(f"截图   : {'关（纯文本模式）' if args.no_image else '开'}")
    print("=" * 74)

    agent = Agent(env, policy, toolkit=toolkit, tracer=tracer, max_steps=args.max_steps)
    t0 = time.time()
    traj = agent.run(task.instruction)
    wall = time.time() - t0

    print("=" * 74)
    for s in traj.steps:
        sig = " ".join(f"{k}={v:.2f}" for k, v in s.decision.signals.items())
        flag = "⬆" if s.escalated else " "
        ok = "" if (s.result is None or s.result.ok) else "  ✗执行失败"
        print(f"  {s.index:>2}{flag} [{s.model:<5}] {str(s.decision.action)[:40]:<42} {sig}{ok}")

    print("=" * 74)
    time.sleep(1.5)   # 等界面稳定再判分
    success = task.check("")
    print(f"  程序化判分 : {'✅ 成功' if success else '❌ 失败'}")
    print(f"  当前前台包 : {_foreground_package(args.serial, adb) or '(读不到)'}")
    print(f"  步数       : {len(traj.steps)}")
    print(f"  强模型占比 : {traj.escalation_rate:.1%}")
    print(f"  模型侧耗时 : {traj.model_latency_s:.2f}s   ← 级联能省的")
    print(f"  环境侧耗时 : {traj.env_latency_s:.2f}s   ← 省不掉的")
    print(f"  墙钟总耗时 : {wall:.1f}s")
    if args.trace:
        print(f"  轨迹       : {args.trace}")

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
