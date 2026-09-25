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

# 这个模块是**唯一会把 AndroidWorld 拉进来**的地方，所以把 FTS 补丁放这儿兜底：
# 不管谁 import 了 task_sets，都先经过预加载。serve_web.py 里也调了一次是故意的
# ——那边调得更早，这里只是防止别的脚本绕过。两次调用是幂等的。
from _sqlite_fts import ensure_fts  # noqa: E402

ensure_fts()

from android_tasks import build_tasks, cleanup_device  # noqa: E402

__all__ = ["UnifiedTask", "collect_tasks", "AW_APPS_WE_HAVE"]

AW_PATH = Path("D:/学习/code/_aw")
"""AndroidWorld 源码位置。换机器改这里，或设环境变量 ANDROID_WORLD_PATH。"""

AW_APPS_WE_HAVE = "（已废弃，见下）"
"""**这张表已经不用了**，留着只为说明它为什么消失。

它原本写死"设备上装了哪些 app"，用来决定哪些任务能跑。问题是它有两重脆弱：

  1. 装了什么 app，是**运行时的事实**，不是源码里的常量
  2. 更关键的是——**装了 app ≠ 任务能跑**。任务还要 app 里有预置数据
     （联系人、下载好的文件、造好的笔记），那些在**快照**里

所以现在改成查盘上真实的快照目录，见 `snapshot_packages()`。
装了什么、数据齐没齐，都以设备当前状态为准。
"""

SNAPSHOT_DIR = "/data/data/android_world/snapshots"
"""AndroidWorld 官方快照目录。**盘上有没有这个包，决定任务收不收。**

之前这里是一张写死的黑名单（`_NEEDS_AW_ENV`），因为当时没跑官方 setup、
设备上一份快照都没有，只能手工把依赖数据的任务挑出来剔掉。

那张表的毛病和所有写死的表一样：**它会过期，而且过期时不报错**。
官方 setup 跑完之后快照齐了，它还在默默剔任务，最后你会以为
"这些任务就是跑不了"，其实是自己把自己过滤掉了。

改成查盘上真实存在的快照——数据在，任务就进；数据不在，任务就不进。
这张表永远和现实一致，不需要人维护。
"""


def snapshot_packages(adb: str, serial: str) -> set[str]:
    """设备上已经有官方快照的包名集合（形如 `net.gsantner.markor`）。

    查不到就返回空集——**返回空集会让所有依赖数据的任务被剔掉**，
    这是保守方向：宁少跑几个，也不让任务去找不存在的数据然后判负。
    """
    import subprocess  # noqa: PLC0415

    try:
        p = subprocess.run(
            [adb, "-s", serial, "shell", "ls", "-1", SNAPSHOT_DIR],
            capture_output=True, timeout=60,
        )
        raw = p.stdout.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return set()
    return {ln.strip() for ln in raw.splitlines() if ln.strip() and "/" not in ln}


def _app_to_package(aw_path: Path, app_names) -> dict[str, str]:
    """AndroidWorld 的 app 短名 → 包名。

    它自己那套 `get_adb_activity` + `extract_package_name` 是纯查表，
    不需要 env，直接借来用——**不要自己再写一份映射**，
    那种表迟早和它的版本对不上。
    """
    if str(aw_path) not in sys.path:
        sys.path.insert(0, str(aw_path))
    from android_world.env import adb_utils  # noqa: PLC0415

    out = {}
    for a in app_names:
        try:
            act = adb_utils.get_adb_activity(a)
            if act:
                out[a] = adb_utils.extract_package_name(act)
        except Exception:  # noqa: BLE001
            pass
    return out


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


def _official_budget(cls, fallback: int = 20) -> int:
    """官方给这个任务的步数预算 = `int(10 * complexity)`。

    直接读它的 `complexity`，**不要自己另定一套**——那样跑出来的成功率
    和官方榜单上的数字就没法比了，而这个项目里"能和官方对齐"比"跑得好看"重要。

    复杂度拿不到时退回 fallback。**要有 fallback**：任务类没有 complexity
    属性的话，上面直接 `int(10 * None)` 会抛异常，一个任务把整张任务表带崩。
    """
    try:
        comp = getattr(cls, "complexity", None)
        if comp is None:
            return fallback
        return max(1, int(10 * float(comp)))
    except Exception:  # noqa: BLE001
        return fallback


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

    # 盘上真实存在的快照 —— 这是"这个任务的前置数据到底有没有"的唯一依据
    have_snap = snapshot_packages(adb, serial)
    all_apps: set[str] = set()
    for c in reg.values():
        all_apps |= set(getattr(c, "app_names", []) or [])
    pkg_of = _app_to_package(aw_path, sorted(all_apps))

    out = []
    missing: dict[str, str] = {}     # 任务名 -> 缺哪个包，末尾打出来
    for name in sorted(reg):
        cls = reg[name]
        try:
            apps = set(cls.app_names)
        except Exception:  # noqa: BLE001
            continue
        if not apps:
            continue
        if name.endswith(_EXCLUDE_SUFFIX):
            continue          # 见模块开头：起点就判成功，白送分

        # 这个任务的每个 app 都得有快照，否则 initialize_task 会**静默失败**——
        # 只打一行 warning 就继续跑，Agent 去找不存在的东西，必然失败，
        # 然后这笔账记到模型头上。所以宁可不放进来。
        gone = [a for a in apps if pkg_of.get(a) and pkg_of[a] not in have_snap]
        if gone:
            missing[name] = ",".join(gone)
            continue

        # 参数固定下来：随机参数会让两次跑的不是同一个任务，
        # 没法比较，也没法复现。**评测要的是可比性，不是多样性。**
        try:
            params = cls.generate_random_params()
        except Exception:  # noqa: BLE001
            continue

        goal = cls(params).goal

        # ⚠️ **每次跑都要新建实例，不能复用。**
        #
        # 这里的矛盾很微妙：
        #
        #   同一个 run 内：setup 和 check **必须是同一个实例**
        #                  （`is_successful` 要读 `initialize_task` 存下的
        #                    `before_photos` 之类的前置快照）
        #   跨 run：       又**必须换新实例**——`initialize_task` 里有
        #                  "已经调用过就抛异常"的守卫，复用的话第二次跑
        #                  在起点就报 `initialize_task() is already called`
        #
        # 早先为了满足前半条，把实例建在循环外面三个闭包共用，于是**同一个
        # 任务跑第二遍必挂**。三形态要跑同一个任务三遍，正好踩满。
        # 之所以一直没暴露，是因为任务表有 5 分钟缓存：缓存一过期，
        # `collect_tasks` 重跑，实例也就换了——**失败与否取决于两次跑
        # 隔了多久**，这种 bug 最难查。
        #
        # 现在用 `box` 把"这一轮的实例"串起来：setup 建、check/teardown 取。
        box: dict = {}

        def setup(b=bridge, c=cls, p=params, bx=box):
            t = c(p)
            t.initialize_task(b)
            bx["t"] = t

        def check(b=bridge, bx=box):
            t = bx.get("t")
            return bool(round(t.is_successful(b))) if t is not None else False

        def teardown(b=bridge, bx=box):
            t = bx.get("t")
            if t is None:
                return
            try:
                t.tear_down(b)
            except Exception:  # noqa: BLE001 - 清场失败不该影响结果
                pass

        # 步数上限用**官方的算法**：`suite_utils._allocate_step_budget()` 是
        # `int(10 * task.complexity)`。
        #
        # ⚠️ 之前这里写死 12，是这次全量测试最大的坑：官方给这 59 个任务的预算是
        # **14~78 步**，12 比最少的那个还低，于是 **57/59 个任务撞上限被截断**。
        # 结果是"0/59 成功"，但那个 0 主要说明的是**没让模型跑完**，不是模型不行。
        #
        # 补一句反面提醒：**给足步数也救不了所有任务**。实测轨迹里模型是在打转
        # （点快门→进相册→返回→再点快门），给它 78 步它还是打转。
        # 所以截断和"模型弱"是两个并存的原因，报告里要分开说。
        budget = _official_budget(cls)
        out.append(UnifiedTask(
            key=f"aw:{name}", name=name, source="androidworld",
            instruction=goal, steps_hint=(budget, budget),
            setup=setup, check=check, teardown=teardown,
        ))

    # **过滤必须出声。** 静默剔任务的后果和静默失败一样：你看到的任务列表
    # 变短了，但没有任何东西告诉你为什么——最后会得出"这些任务跑不了"的
    # 错误结论，而实际上是数据没准备好。
    print(f"[task_sets] 快照 {len(have_snap)} 个；"
          f"收录 {len(out)} 个 AndroidWorld 任务，"
          f"因缺数据剔除 {len(missing)} 个")
    for nm, pkgs in sorted(missing.items())[:8]:
        print(f"             · {nm} 缺 {pkgs}")
    if len(missing) > 8:
        print(f"             …还有 {len(missing)-8} 个")
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
