"""把两类任务统一成一个接口，好让前端一视同仁地列出来、跑起来。

    ours:<name>   我们自己编的 6 个任务（判分写在 android_tasks.py）
    aw:<Name>     AndroidWorld 的任务（判分是它原版代码，经 aw_bridge 跑）

## 为什么要统一

前端只该看见"一个任务列表"，不该关心它背后是哪套判分。否则每加一个
benchmark 就要改一次界面。

统一后的接口就四个东西：

    name/instruction   给用户看的
    setup(adb)         跑之前建前置状态
    check(adb)         判分，返回 bool
    teardown(adb)      跑完清场

## ⚠️ AndroidWorld 的 Verify 类任务被排除

它有一批 `*Verify` 任务（`SystemWifiTurnOffVerify` 等），设计上假定**前面
有一个 setup 任务已经把状态设好了**，它只负责验证。单独跑的话，只要设备
碰巧处于那个状态，**在起点就判成功**——白送分。

实测确认过：`SystemBluetoothTurnOffVerify` 等在起点直接返回 1.0。
所以这里把它们剔掉，只保留会真正改变状态、需要 agent 去操作的任务。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from android_tasks import build_tasks, cleanup_device  # noqa: E402

__all__ = ["UnifiedTask", "collect_tasks", "AW_APPS_WE_HAVE"]

AW_PATH = Path("D:/学习/code/_aw")
"""AndroidWorld 源码位置。换机器改这里，或设环境变量 ANDROID_WORLD_PATH。"""

AW_APPS_WE_HAVE = {
    # 系统自带
    "settings", "clock", "chrome", "camera", "contacts", "files", "dialer",
    # AndroidWorld 自带的那批（用 scripts/setup_androidworld_env.py 装的）
    "markor", "broccoli app", "pro expense", "simple calendar pro",
    "simple sms messenger", "simple gallery pro", "retro music", "vlc",
    "clipper", "tasks", "joplin", "audio recorder", "opentracks",
    "simple draw pro", "android world",
}
"""设备上装了哪些 app——**这决定了哪些 AndroidWorld 任务可跑**。

装它那批 app 之前，这个集合只有系统自带的 7 个，于是 116 个任务里只有
20 个能进来，而且其中 6 个还因为缺预置数据跑不了。

现在用 `scripts/setup_androidworld_env.py` 把 markor（17 个任务）、
broccoli（13 个）、pro expense（9 个）这些装上之后，可跑的任务面**大了一个量级**。"""

_NEEDS_AW_ENV = {
    # 这些任务的前置数据（文件、网页）**不在任务代码里，而在快照里**。
    # AndroidWorld 的 `initialize_task` 会去还原预置的 app 快照：
    #     /data/data/android_world/snapshots/<app>/
    # 那个目录来自它**定制的模拟器镜像**，我们这台设备上根本不存在
    # （实测：`/data/data/android_world/` 整个目录都没有）。
    #
    # 后果是 `initialize_task` **静默失败**——只打一行 warning，
    # 任务照跑，但 Agent 要找的东西压根不在设备上，必然失败。
    # 这种失败会被记成"模型不行"，所以**必须从列表里剔掉**，不能留着充数。
    #
    # 换用 AndroidWorld 官方 AVD（带全部 app + 快照）之后可以把它们加回来。
    "BrowserDraw",            # 需要 /sdcard/Download/task.html
    "BrowserMaze",            # 同上
    "BrowserMultiply",        # 同上
    "FilesDeleteFile",        # 需要预置文件 jolly_tree_final.pdf 等
    "FilesMoveFile",          # 需要预置文件
    "ContactsNewContactDraft",  # 读 UI 树，我们的 forest 结构和它家对不上
}

_EXCLUDE_SUFFIX = "Verify"
"""见模块开头对 Verify 白送分的说明。

⚠️ 结尾是 `Verify` **没有下划线**（`SystemBluetoothTurnOffVerify`）。
第一版写成 `_Verify`，一个都没滤掉——**过滤条件写错的后果是静默的**：
列表照常出来，只是里面混着几个起点就判满分的任务。"""


@dataclass
class UnifiedTask:
    key: str                # 给前端的唯一 id，形如 "aw:SystemWifiTurnOn"
    name: str               # 显示名
    source: str             # "ours" | "androidworld"
    instruction: str
    package: str = ""
    steps_hint: tuple[int, int] = (3, 8)
    setup: Callable[[], None] | None = None
    check: Callable[[], bool] | None = None
    teardown: Callable[[], None] | None = None


def _ours(adb: str, serial: str) -> list[UnifiedTask]:
    out = []
    for name, t in build_tasks(adb, serial).items():
        out.append(UnifiedTask(
            key=f"ours:{name}", name=name, source="ours",
            instruction=t.instruction, package=t.package, steps_hint=t.steps_hint,
            check=t.check,
            # 我们的任务没有独立的前置状态，清场用统一的那套
            teardown=lambda a=adb, s=serial: cleanup_device(a, s, verbose=False),
        ))
    return out


def _androidworld(adb: str, serial: str, aw_path: Path) -> list[UnifiedTask]:
    """把 AndroidWorld 的任务包成同一个形状。

    每个任务的三个动作分别接到它自己的方法上：

        setup     -> task.initialize_task(env)   建前置状态（它自带）
        check     -> task.is_successful(env)     读设备真实状态（它原版判分）
        teardown  -> task.tear_down(env)         清场（它自带）
    """
    if str(aw_path) not in sys.path:
        sys.path.insert(0, str(aw_path))
    from android_world import registry  # noqa: PLC0415
    from aw_bridge import AndroidWorldBridge  # noqa: PLC0415

    reg = registry.TaskRegistry().get_registry(registry.TaskRegistry.ANDROID_FAMILY)
    bridge = AndroidWorldBridge(adb=adb, serial=serial, aw_path=aw_path)

    out = []
    for name in sorted(reg):
        cls = reg[name]
        try:
            apps = set(cls.app_names)
        except Exception:  # noqa: BLE001
            continue
        if not apps or not apps <= AW_APPS_WE_HAVE:
            continue          # 依赖我们没装的 app
        if name.endswith(_EXCLUDE_SUFFIX):
            continue          # 见模块开头：起点就判成功，白送分
        if name in _NEEDS_AW_ENV:
            continue          # 前置数据在快照里，我们设备上没有

        # 参数固定下来：随机参数会让两次跑的不是同一个任务，
        # 没法比较，也没法复现。**评测要的是可比性，不是多样性。**
        try:
            params = cls.generate_random_params()
        except Exception:  # noqa: BLE001
            continue

        # 实例化一次，三个回调闭包共用同一个 task 对象——
        # 它内部有 `initialized` 状态，setup/check 必须是同一个实例，
        # 各建一个的话 `is_successful` 会报"还没初始化"。
        task = cls(params)
        goal = task.goal

        def setup(t=task, b=bridge):
            t.initialize_task(b)

        def check(t=task, b=bridge):
            return bool(round(t.is_successful(b)))

        def teardown(t=task, b=bridge):
            try:
                t.tear_down(b)
            except Exception:  # noqa: BLE001 - 清场失败不该影响结果
                pass

        out.append(UnifiedTask(
            key=f"aw:{name}", name=name, source="androidworld",
            instruction=goal, steps_hint=(5, 15),
            setup=setup, check=check, teardown=teardown,
        ))
    return out


def collect_tasks(adb: str, serial: str, *, aw_path: Path | None = None,
                  include_aw: bool = True) -> dict[str, UnifiedTask]:
    """返回全部可用任务，键是 `key`。AndroidWorld 拿不到就只返回我们自己的。"""
    tasks = {t.key: t for t in _ours(adb, serial)}
    if include_aw:
        p = aw_path or Path(__import__("os").environ.get("ANDROID_WORLD_PATH", AW_PATH))
        try:
            for t in _androidworld(adb, serial, p):
                tasks[t.key] = t
        except Exception as e:  # noqa: BLE001
            # **拿不到就降级，不要崩。** 前端还能用我们自己的 6 个任务，
            # 页面顶上会标出 AndroidWorld 不可用。
            print(f"[task_sets] AndroidWorld 不可用，跳过：{type(e).__name__}: {e}")
            tasks["__aw_error__"] = UnifiedTask(
                key="__aw_error__", name="AndroidWorld 不可用", source="error",
                instruction=f"{type(e).__name__}: {e}",
            )
    return tasks
