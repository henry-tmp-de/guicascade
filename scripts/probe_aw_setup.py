"""探针：AndroidWorld 的任务前置状态（initialize_task）在我们设备上能不能建起来。

## 为什么要有这个

AndroidWorld 的任务分两层状态：

    ① 装 app          —— 已经做了（scripts/setup_androidworld_env.py）
    ② app 内部的数据  —— 由 initialize_task() 从**快照**还原
                          /data/data/android_world/snapshots/<app>/

第 ② 层我们没有。**麻烦的是它失败是静默的**：只打一行 warning，
任务照跑，Agent 去找一个设备上根本不存在的东西，必然失败——
然后这笔账会记到模型头上。这是最坏的一种 bug：**分数错了，但没人知道**。

所以这个探针不问"模型行不行"，只问：

    A. initialize_task 会不会报错？
    B. 建完之后，is_successful 是不是 False？（起点就该是 False，
       如果是 True 说明任务被"白送"了，判分没意义）

跑法：

    python scripts/probe_aw_setup.py                    # 探前 8 个
    python scripts/probe_aw_setup.py --n 30             # 多探几个
    python scripts/probe_aw_setup.py --only Markor      # 只看名字含 Markor 的
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from guicascade.envs.android import find_adb  # noqa: E402

AW_PATH = Path("D:/学习/code/_aw")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="emulator-5554")
    ap.add_argument("--n", type=int, default=8, help="探多少个（0 = 全部）")
    ap.add_argument("--only", default="", help="只探名字含这个串的")
    ap.add_argument("--aw-path", default=str(AW_PATH))
    args = ap.parse_args()

    aw_path = Path(args.aw_path)
    if str(aw_path) not in sys.path:
        sys.path.insert(0, str(aw_path))

    from android_world import registry  # noqa: PLC0415
    from aw_bridge import AndroidWorldBridge  # noqa: PLC0415
    from android_world.env import json_action  # noqa: F401, PLC0415

    adb = find_adb()
    reg = registry.TaskRegistry().get_registry(registry.TaskRegistry.ANDROID_FAMILY)
    bridge = AndroidWorldBridge(adb=adb, serial=args.serial, aw_path=aw_path)

    names = sorted(reg)
    if args.only:
        names = [n for n in names if args.only.lower() in n.lower()]
    if args.n:
        names = names[: args.n]

    print(f"\n{'='*78}")
    print(f"  探针：initialize_task 在我们设备上能不能建起前置状态")
    print(f"  设备 {args.serial} · {len(names)} 个任务")
    print(f"{'='*78}\n")

    # 记录 initialize_task 打的 warning —— 静默失败就藏在这里面
    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.WARNING)

    rows = []
    for i, name in enumerate(names, 1):
        cls = reg[name]
        try:
            params = cls.generate_random_params()
            task = cls(params)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i:>2}] {name:<42} ⚠️ 实例化失败 {type(e).__name__}")
            rows.append((name, "init-err", str(e)[:60]))
            continue

        log_buf.truncate(0)
        log_buf.seek(0)

        # ---- A. initialize_task 会不会炸 ----
        setup_err = ""
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                task.initialize_task(bridge)
        except Exception as e:  # noqa: BLE001
            setup_err = f"{type(e).__name__}: {e}"

        warns = [ln for ln in log_buf.getvalue().splitlines() if ln.strip()]

        # ---- B. 起点判分该是 False ----
        at_start = None
        check_err = ""
        if not setup_err:
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    at_start = bool(round(task.is_successful(bridge)))
            except Exception as e:  # noqa: BLE001
                check_err = f"{type(e).__name__}: {e}"

        # ---- 结论 ----
        #
        # ⚠️ 判据要分清主次：**"起点判分是不是 False"比"有没有 warning"重要得多**。
        # 快照缺失对每个任务都会 warning（它无脑给自己 app_names 里的每个 app
        # 都试着还原一次），但其中大部分任务根本不依赖预置数据。
        # 第一版把 warning 排在前面，结果 9 个任务全被标成"有 warning"，
        # 看不到真正该看的东西。**排序错了，等于没测。**
        if setup_err:
            verdict, note = "❌ setup 炸", setup_err
        elif check_err:
            verdict, note = "⚠️ 判分炸", check_err
        elif at_start:
            verdict, note = "🎁 起点即满分", "判分白送，无意义"
        elif warns:
            verdict, note = "✅ 可跑·缺快照", warns[0][:80]
        else:
            verdict, note = "✅ 可跑", ""

        print(f"  [{i:>2}] {name:<42} {verdict}")
        if note:
            print(f"        {note}")
        rows.append((name, verdict, note))

    print(f"\n{'='*78}")
    from collections import Counter
    c = Counter(v for _, v, _ in rows)
    for k, v in c.most_common():
        print(f"  {k:<20} {v}")
    print(f"{'='*78}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
