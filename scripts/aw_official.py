"""AndroidWorld 官方环境：连上去、建前置状态、按需跑任务。

## 为什么要有这个文件（而不是继续用 aw_bridge）

我们先用 `aw_bridge.py` 把 AndroidWorld 的判分接到了自己的 adb 上，跑通了一部分。
但探针测出来的结果很明确：

    91 个任务   43 个可跑   41 个 setup 炸

而那 41 个**没有一个是桥的问题**，全是同一件事：

    22×  <app>/databases does not exist     app 从没初始化
     9×  Target text "NEXT" not found       app 引导页没走过
     4×  no such table: PlaylistEntity      app 数据库没建
     ...

也就是说，自建路的天花板不在"判分接得对不对"，而在**前置状态建不出来**。
AndroidWorld 的前置状态由它自己的 `setup_apps(env)` 生成、存成**快照**：

    /data/data/android_world/snapshots/<包名>/

这套东西依赖它的 `AndroidWorldController`（gRPC + accessibility forwarder）。
绕不过去，所以这个文件就是**把官方那套正经跑起来**。

## 两条路并存，不是替换

`aw_bridge.py` 留着——它证明了"判分可以脱离官方的执行层"，而且不依赖
`-grpc` 启动模拟器，跑得快。**两个环境按成功率取用**：

    官方环境 (本文件)   —— 要跑 AndroidWorld 任务时用，前置状态齐全
    自建环境 (android.py) —— 我们自己的任务、以及不需要快照的 AW 任务

## 前提

模拟器必须**带 `-grpc 8554` 启动**，否则连不上：

    ANDROID_AVD_HOME=D:\\tools\\android-avd \\
    emulator.exe -avd AndroidWorldAvd -no-snapshot -no-audio \\
                 -no-boot-anim -gpu host -memory 4096 -cores 4 \\
                 -port 5554 -grpc 8554

看一眼有没有起来：

    python scripts/aw_official.py --check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guicascade.envs.android import find_adb  # noqa: E402

AW_PATH = Path("D:/学习/code/_aw")
CONSOLE_PORT = 5554
GRPC_PORT = 8554


def _load_aw(aw_path: Path):
    if str(aw_path) not in sys.path:
        sys.path.insert(0, str(aw_path))


def connect(aw_path: Path = AW_PATH, *, emulator_setup: bool = False,
            console_port: int = CONSOLE_PORT, grpc_port: int = GRPC_PORT,
            adb: str | None = None):
    """连上官方环境。`emulator_setup=True` 会顺便把 app 前置状态建好。

    ⚠️ **`emulator_setup` 只在第一次用**。官方文档写得很明确：

        This must be done once and only once before running Android World.
        After an emulator is setup, this flag should always be False.

    它会把每个 app 清空重装、跑一遍引导页、再把 `/data/data/<包>` 存成快照。
    跑一次要很久（十几个 app），而且**中途重复跑会把已有快照覆盖掉**。
    """
    _load_aw(aw_path)
    from android_world.env import env_launcher  # noqa: PLC0415

    return env_launcher.load_and_setup_env(
        console_port=console_port,
        adb_path=adb or find_adb(),
        emulator_setup=emulator_setup,
        freeze_datetime=True,      # 冻住时间到 2023-10，任务参数才可复现
        grpc_port=grpc_port,
    )


def setup_apps_skipping(skip: set[str], aw_path: Path = AW_PATH,
                        console_port: int = CONSOLE_PORT,
                        grpc_port: int = GRPC_PORT) -> None:
    """建前置状态，但**跳过指定的 app**。

    ## 为什么需要"跳过"这个能力

    `setup_apps()` 是**一条直线**：24 个 app 按字母序排，中间任何一个抛异常，
    整条就断，**后面排队的全部不跑**。第一次实测崩在 Joplin（第 11 个，
    本机 Python 的 sqlite3 没编 FTS4），于是 Markor / Broccoli / Calendar
    这些任务最多的 app 一个都没轮到——**代价远大于崩掉的那一个**。

    官方其实留了口子：`setup_apps(env, app_list=...)` 就收自定义列表。
    所以不用改它的代码，把有问题的那个从列表里摘掉重跑即可。
    **已建好的快照不会重跑**（`setup_app` 开头会自己判断），所以可以安全续跑。
    """
    _load_aw(aw_path)
    from android_world.env.setup_device import apps as aw_apps  # noqa: PLC0415
    from android_world.env.setup_device import setup as aw_setup  # noqa: PLC0415

    env = connect(aw_path, emulator_setup=False,
                  console_port=console_port, grpc_port=grpc_port)

    todo = tuple(a for a in aw_setup._APPS if a.__name__ not in skip)  # noqa: SLF001
    skipped = [a.__name__ for a in aw_setup._APPS if a.__name__ in skip]  # noqa: SLF001

    print(f"\n  共 {len(aw_setup._APPS)} 个 app，本次跑 {len(todo)} 个")  # noqa: SLF001
    if skipped:
        print(f"  跳过：{', '.join(skipped)}")
    print()

    aw_setup.setup_apps(env, app_list=todo)
    _ = aw_apps  # 保持 import 便于将来按需取 app 类


def setup_only_missing(aw_path: Path = AW_PATH,
                       console_port: int = CONSOLE_PORT,
                       grpc_port: int = GRPC_PORT) -> None:
    """**只补还没有快照的 app**，已经建好的跳过。

    ## 为什么要这个

    `setup_apps()` 每次都会把列表里的 app **从头重跑一遍**（清空 → 启动 →
    点引导页 → 存快照）。第一次实测里，前 10 个已经建好了，但从头跑意味着
    再花十几分钟做无谓功——而**挂起风险是按 app 数累积的**：每多跑一个，
    就多一次"卡在那个 app 的权限弹窗上"的机会。

    实测就吃过这个亏：第二轮跑到 OsmAnd 时整个进程挂死在
    `AppManageExternalStorageActivity` 上，二十多分钟没动静，
    而它前面 13 个快照**早就建好躺在设备上了**。

    快照是**盘上的持久状态**，不是进程内的。所以"还缺哪些"这个问题
    可以直接问设备，不用问进程——这样每轮只做真正没做过的部分，
    挂掉重来的代价是常数，不是越滚越大。
    """
    _load_aw(aw_path)
    from android_world.env import adb_utils  # noqa: PLC0415
    from android_world.env.setup_device import setup as aw_setup  # noqa: PLC0415

    env = connect(aw_path, emulator_setup=False,
                  console_port=console_port, grpc_port=grpc_port)
    ctl = env.controller

    have = _device_snapshot_packages()
    print(f"\n  设备上已有 {len(have)} 个快照")

    todo = []
    for cls in aw_setup._APPS:  # noqa: SLF001
        try:
            act = adb_utils.get_adb_activity(cls.app_name)
            pkg = adb_utils.extract_package_name(act) if act else None
        except Exception:  # noqa: BLE001
            pkg = None
        if pkg and pkg in have:
            continue
        todo.append(cls.__name__)

    if not todo:
        print("  ✅ 全部齐了，不用再跑\n")
        return
    print(f"  还缺 {len(todo)} 个：{', '.join(todo)}\n")

    # 用官方的 app_list 参数，一次只跑缺的那些
    wanted = tuple(c for c in aw_setup._APPS if c.__name__ in set(todo))  # noqa: SLF001
    aw_setup.setup_apps(env, app_list=wanted)
    _ = ctl


def setup_selected(class_names: list[str], aw_path: Path = AW_PATH,
                   console_port: int = CONSOLE_PORT,
                   grpc_port: int = GRPC_PORT) -> None:
    """只跑指定的几个 app 类。

    ## 为什么要把粒度降到"单个 app"

    连续两次实测里，setup 都在某个 app 上**挂死**（卡在系统权限弹窗，
    gRPC 调用不返回）。挂死的代价不是"少做一个 app"，而是**整批停住**——
    `setup_apps` 是一条直线，前面做好的白等，后面排队的全不跑。

    把粒度降到单个 app、外面套一层超时，挂死就只损失这一个：
    进程被砍掉，下一个照跑。**无人值守的批量任务里，
    "一个坏掉不拖累其他"比"跑得快"重要得多。**
    """
    _load_aw(aw_path)
    from android_world.env import adb_utils  # noqa: PLC0415
    from android_world.env.setup_device import setup as aw_setup  # noqa: PLC0415

    have = _device_snapshot_packages()
    picked = [c for c in aw_setup._APPS if c.__name__ in set(class_names)]  # noqa: SLF001

    # 已经有快照的跳过 —— 这样外部驱动脚本可以**无脑重跑**，
    # 不用自己维护"跑到哪了"。中断重来的代价是常数。
    wanted = tuple(c for c in picked if _pkg_of(c, adb_utils) not in have)
    skipped = [c.__name__ for c in picked if _pkg_of(c, adb_utils) in have]

    if skipped:
        print(f"  已有快照，跳过：{', '.join(skipped)}")
    if not wanted:
        print("  这些 app 都已经有快照了，不用跑")
        return

    env = connect(aw_path, emulator_setup=False,
                  console_port=console_port, grpc_port=grpc_port)
    print(f"\n  跑 {len(wanted)} 个：{[c.__name__ for c in wanted]}\n")
    aw_setup.setup_apps(env, app_list=wanted)


def _pkg_of(cls, adb_utils) -> str | None:
    try:
        act = adb_utils.get_adb_activity(cls.app_name)
        return adb_utils.extract_package_name(act) if act else None
    except Exception:  # noqa: BLE001
        return None


def _device_snapshot_packages() -> set[str]:
    """直接问设备：快照目录下有哪些包。查不到返回空集。"""
    import subprocess  # noqa: PLC0415

    try:
        p = subprocess.run(
            [find_adb(), "-s", f"emulator-{CONSOLE_PORT}", "shell",
             "ls", "-1", "/data/data/android_world/snapshots"],
            capture_output=True, timeout=60,
        )
        raw = p.stdout.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return set()
    return {ln.strip() for ln in raw.splitlines() if ln.strip() and "/" not in ln}


def check() -> int:
    """只做连通性检查：能不能拿到 a11y 树。**不建前置状态、不改设备。**"""
    print(f"\n{'='*74}")
    print("  检查官方环境连通性")
    print(f"{'='*74}\n")

    t0 = time.time()
    try:
        env = connect(emulator_setup=False)
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 连不上：{type(e).__name__}: {e}\n")
        print("  常见原因：")
        print("    1. 模拟器没带 -grpc 8554 启动")
        print("    2. 模拟器还没开机完（再多等一会儿）")
        print("    3. a11y forwarder 没装上")
        return 1

    print(f"  ✅ 连上了（{time.time()-t0:.1f}s）")

    try:
        state = env.get_state(wait_to_stabilize=False)
        els = state.ui_elements or []
        print(f"  ✅ a11y 树：{len(els)} 个元素")
        named = [e for e in els if (e.text or e.content_description)]
        print(f"     其中有文字/描述的：{len(named)}")
        for e in named[:6]:
            label = (e.text or e.content_description or "")[:44]
            print(f"       · {label}")
        if state.pixels is not None:
            print(f"  ✅ 截图：{state.pixels.shape}")
        else:
            print("  ⚠️ 截图拿不到（多数判分不看像素，不影响）")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ 取状态失败：{type(e).__name__}: {e}")
        return 1

    try:
        print(f"  · 屏幕尺寸 {env.controller.device_screen_size}")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ 屏幕尺寸取不到：{type(e).__name__}: {e}")

    print(f"\n  结论：**官方环境可用**，可以跑 `--setup` 建前置状态了。\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只查连通性")
    ap.add_argument("--setup", action="store_true",
                    help="建 app 前置状态（跑一次就够，要很久）")
    ap.add_argument("--skip", default="",
                    help="跳过这些 app 类名，逗号分隔，如 JoplinApp,OsmAndApp")
    ap.add_argument("--only-missing", action="store_true",
                    help="只补设备上还没有快照的 app（推荐，可反复跑到齐）")
    ap.add_argument("--apps", default="",
                    help="只跑这些 app 类名，逗号分隔（配合外部超时逐个跑）")
    ap.add_argument("--aw-path", default=str(AW_PATH))
    args = ap.parse_args()

    if args.apps:
        setup_selected([s.strip() for s in args.apps.split(",") if s.strip()],
                       Path(args.aw_path))
        return 0

    if args.only_missing:
        print(f"\n{'='*74}")
        print("  只补缺的 app")
        print(f"{'='*74}")
        t0 = time.time()
        setup_only_missing(Path(args.aw_path))
        print(f"\n  ✅ 完成，用时 {(time.time()-t0)/60:.1f} 分钟\n")
        return 0

    if args.skip:
        skip = {s.strip() for s in args.skip.split(",") if s.strip()}
        print(f"\n{'='*74}")
        print(f"  建前置状态（跳过 {len(skip)} 个）")
        print(f"{'='*74}")
        t0 = time.time()
        setup_apps_skipping(skip, Path(args.aw_path))
        print(f"\n  ✅ 完成，用时 {(time.time()-t0)/60:.1f} 分钟\n")
        return 0

    if args.check or not args.setup:
        return check()

    print(f"\n{'='*74}")
    print("  建 app 前置状态 —— 这会清空并重装每个 app，耗时较长")
    print(f"{'='*74}\n")
    t0 = time.time()
    env = connect(Path(args.aw_path), emulator_setup=True)
    print(f"\n  ✅ 前置状态建好，用时 {(time.time()-t0)/60:.1f} 分钟")
    print("  之后启动模拟器**不要**再加 `--setup`。\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
