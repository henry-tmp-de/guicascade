"""用**官方 android_env** 驱动模拟器，替代自造的 `aw_bridge`。

## 为什么换

之前的架构是：

    Agent -> TaskEval(官方) -> 【我们的 aw_bridge】 -> adb -> 模拟器

中间那层是我们自己写的"把官方请求翻译成 adb"的翻译层。**2026-09-25 一天之内
从它身上查出四个 bug**，而且每一个的症状都长成"模型不行"：

    start_activity 读错 proto 字段名  -> 所有 app 启动失败
    apps.yaml 漏 17 个 app            -> open_app 名字查不到
    BoundingBox 的 x_max 填成了宽度    -> 依赖 bbox 的判分全错
    调试浮层没人关                     -> 屏幕被污染

官方本来就有这一层（`android_env` + `-grpc` 的 accessibility forwarder），
而且它是和 TaskEval 一起测过的。所以现在直接用它，本文件只做**形状转换**：

    我们的 Action  -> 官方 JSONAction
    官方 State     -> 我们的 Observation

**只做转换，不实现任何设备逻辑。** 这条边界要守住：一旦这里开始"自己拼
adb 命令"，就又回到老路上去了。

## 一个不能弄错的契约：编号

官方 `execute_action()` 里，`click(index=N)` 索引进的是
**`state.ui_elements` 全量列表**（`actuation.execute_adb_action(action,
state.ui_elements, ...)`），而且它会**重新取一次 state** 再执行。

所以本文件渲染给模型的编号**必须**是 `ui_elements` 里的原始下标。
我们只过滤"显示哪些行"，**绝不重新编号**（`_should_show` + 原下标），
否则模型说的 [5] 和官方点的 [5] 就不是一个东西 —— 这个坑在自造环境里
已经踩过一次（见 `envs/android.py` 里 `actionable_elements` 的说明）。

## 用法

    from guicascade.envs.android_official import OfficialAndroidEnv
    env = OfficialAndroidEnv()
    obs = env.reset(task)
    res = env.step(Action("click", {"index": 3}))
    env.close()

前提：模拟器带 `-grpc 8554` 启动（当前这台就是）。
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Any

from ..types import Action, Observation, StepResult

__all__ = ["OfficialAndroidEnv", "to_json_action", "render_ui_elements"]

AW_PATH = Path("D:/学习/code/_aw")
"""AndroidWorld 源码位置。换机器改这里，或设环境变量 ANDROID_WORLD_PATH。"""

DEFAULT_ADB = "D:/tools/android-sdk/platform-tools/adb.exe"

MAX_LINES = 80
"""最多渲染多少行。**截断是安全的**：只影响"看不看得见"，
不影响编号 —— 显示出来的行用的仍是它在 `ui_elements` 里的原始下标。"""


def _ensure_aw() -> None:
    import os  # noqa: PLC0415

    p = Path(os.environ.get("ANDROID_WORLD_PATH") or AW_PATH)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# --------------------------------------------------------------------------
# 形状转换：我们的 Action -> 官方 JSONAction
# --------------------------------------------------------------------------

_OURS_TO_OFFICIAL = {
    "click": "click",
    "double_tap": "double_tap",
    "long_press": "long_press",
    "input_text": "input_text",
    "keyboard_enter": "keyboard_enter",
    "navigate_home": "navigate_home",
    "navigate_back": "navigate_back",
    "open_app": "open_app",
    "scroll": "scroll",
    "swipe": "swipe",
    "wait": "wait",
    "answer": "answer",
}
"""我们的动作名 -> 官方动作名。两边基本一一对应（我们的动作空间本来就是
照官方 `json_action.py` 做的），这张表是**显式**的，为的是将来哪边加了动作
能一眼看出少没少。"""


def to_json_action(action: Action):
    """我们的 `Action` -> 官方 `JSONAction`。

    参数名两边几乎一致，只有一处要注意：**官方的 `input_text` 带
    `clear_text`**，不清空的话新文本会接在旧的后面。
    """
    _ensure_aw()
    from android_world.env import json_action  # noqa: PLC0415

    name = _OURS_TO_OFFICIAL.get(action.name)
    if name is None:
        raise ValueError(f"官方动作空间里没有 {action.name!r}（见 _OURS_TO_OFFICIAL）")

    a = dict(action.args or {})
    kw: dict[str, Any] = {"action_type": name}

    if name in ("click", "double_tap", "long_press"):
        if a.get("index") is not None:
            kw["index"] = int(a["index"])
        else:
            kw["x"], kw["y"] = int(a.get("x", 0)), int(a.get("y", 0))

    elif name == "input_text":
        kw["text"] = str(a.get("text", ""))
        # 默认清空：不清的话是在原有内容后面追加，模型多半不是这个意思
        kw["clear_text"] = bool(a.get("clear_text", True))

    elif name == "open_app":
        kw["app_name"] = str(a.get("app_name") or a.get("package") or "")

    elif name in ("scroll", "swipe"):
        kw["direction"] = str(a.get("direction", "down"))

    elif name == "answer":
        kw["text"] = str(a.get("text", ""))

    return json_action.JSONAction(**kw)


# --------------------------------------------------------------------------
# 形状转换：官方 State -> 我们的 Observation
# --------------------------------------------------------------------------


def _should_show(e) -> bool:
    """这个元素值不值得占一行。

    **只决定"显不显示"，绝不决定"编号是多少"** —— 编号永远是它在
    `ui_elements` 里的原始下标，这一点是硬契约（见模块 docstring）。
    """
    if (e.text or "").strip() or (e.content_description or "").strip():
        return True
    return bool(e.is_clickable or e.is_editable or e.is_scrollable)


def render_ui_elements(elements: list, *, max_lines: int = MAX_LINES) -> str:
    """官方 `UIElement` 列表 -> 给模型看的纯文本。

    格式**沿用自造环境那一套**（`[下标] <类名> '文字' 标记 id=xxx bounds=...`），
    理由是模型和提示词都已经习惯了它，换成新格式等于白白让模型重新学一遍。
    """
    lines: list[str] = []
    for i, e in enumerate(elements):
        if not _should_show(e):
            continue
        if len(lines) >= max_lines:
            lines.append(f"…（还有更多元素未显示，共 {len(elements)} 个）")
            break

        label = (e.text or "").strip() or (e.content_description or "").strip()
        flags = []
        if e.is_clickable:
            flags.append("可点击")
        if e.is_editable:
            flags.append("可编辑")
        if e.is_scrollable:
            flags.append("可滚动")
        if e.is_enabled is False:
            flags.append("已禁用")

        ident = ""
        if e.resource_name:
            ident = f" id={e.resource_name.split('/')[-1]}"
        # ⚠️ 官方 `BoundingBox` 的四个值都是**绝对坐标**（x_min,x_max,y_min,y_max），
        # 不是"起点+宽高"。渲染成 (左,上,右,下) 给模型看。
        #
        # ⚠️ `bbox` **可能是 None**（有些节点官方没能算出坐标）。实测踩过：
        # 直接取 `b.x_min` 会抛 AttributeError，整轮任务在这一行就死。
        # 坐标缺失的元素照样要显示出来（它可能有文字信息），只是不写 bounds。
        b = getattr(e, "bbox", None)
        box = (f" bounds=({b.x_min:.0f},{b.y_min:.0f},{b.x_max:.0f},{b.y_max:.0f})"
               if b is not None else "")
        cls = (e.class_name or "?").split(".")[-1]
        lines.append(
            f"[{i}] <{cls}> {label!r}{'' if not flags else ' ' + ' '.join(flags)}"
            f"{ident}{box}"
        )
    return "\n".join(lines) if lines else "(屏幕上没有可交互元素)"


def _encode_png(pixels) -> bytes | None:
    """官方给的是 RGB ndarray。转成 PNG 字节，给前端和演示用。"""
    try:
        from PIL import Image  # noqa: PLC0415

        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="PNG")
        return buf.getvalue()
    except Exception:  # noqa: BLE001 - 截图拿不到不该让整轮跑不起来
        return None


# --------------------------------------------------------------------------
# 环境本体
# --------------------------------------------------------------------------


class OfficialAndroidEnv:
    """把我们框架要的 `reset/step/close` 架在官方 `AsyncEnv` 上。"""

    name = "android-official"

    def __init__(
        self,
        *,
        console_port: int = 5554,
        grpc_port: int = 8554,
        adb_path: str = DEFAULT_ADB,
        capture_image: bool = True,
        emulator_setup: bool = False,
        freeze_datetime: bool = True,
        task_package: str = "",
    ) -> None:
        _ensure_aw()
        from android_world.env import env_launcher  # noqa: PLC0415

        self.capture_image = capture_image
        self.task_package = task_package
        self._env = env_launcher.load_and_setup_env(
            console_port=console_port,
            emulator_setup=emulator_setup,
            # 官方默认把设备时间冻结在 2023-10，为的是让 benchmark 可复现。
            # 跟随官方，别自作主张 —— 时钟类任务对时间敏感。
            freeze_datetime=freeze_datetime,
            adb_path=adb_path,
            grpc_port=grpc_port,
        )
        # 官方自己也把"屏幕上的坐标浮层"当污染，会主动关掉它。
        # 我们之前是手工发现的（截图里那条栏），这里直接走官方的口子。
        try:
            self._env.hide_automation_ui()
        except Exception:  # noqa: BLE001
            pass

    # -- 观察 ---------------------------------------------------------------

    def _observe(self, task: str, state=None) -> Observation:
        st = state if state is not None else self._env.get_state(wait_to_stabilize=False)
        els = list(st.ui_elements)
        return Observation(
            text=render_ui_elements(els),
            image=_encode_png(st.pixels) if self.capture_image else None,
            meta={
                "task": task,
                "n_elements": len(els),
                "n_shown": sum(1 for e in els if _should_show(e)),
                "activity": self._foreground(),
            },
        )

    def _foreground(self) -> str:
        try:
            return self._env.foreground_activity_name
        except Exception:  # noqa: BLE001
            return ""

    # -- 生命周期 -----------------------------------------------------------

    def reset(self, task: str) -> Observation:
        """回到任务的起点。

        `go_home=True` = 先按 HOME。官方 `reset()` 本身不清 app 数据 ——
        任务的前置状态由 `initialize_task()` 负责（那是任务的事，不是环境的事），
        这个分工和官方一致。
        """
        state = self._env.reset(go_home=True)
        time.sleep(0.6)
        return self._observe(task, state)

    def step(self, action: Action) -> StepResult:
        """执行一个动作。

        **动作失败不抛异常**，返回 `ok=False` —— 这是 `Environment` 协议的约定
        （点空、参数错都是 GUI 任务的常态，不该中断整条轨迹）。
        只有环境本身坏掉才让异常冒出去。
        """
        try:
            self._env.execute_action(to_json_action(action))
            ok, error = True, ""
        except Exception as e:  # noqa: BLE001 - 动作失败是常态
            ok, error = False, f"{type(e).__name__}: {e}"

        # `wait` 是唯一需要额外等一拍的：官方把它当空操作，不等的话
        # 下一帧读到的还是旧屏幕。
        if action.name == "wait":
            time.sleep(float(action.args.get("seconds", 1.0) or 1.0))
        elif action.name in ("open_app", "click", "double_tap", "long_press"):
            time.sleep(0.6)

        try:
            state = self._env.get_state(wait_to_stabilize=False)
        except Exception as e:  # noqa: BLE001
            # 拿不到新观察 = 环境真的坏了，这个要让上层知道
            raise RuntimeError(f"取不到设备状态：{type(e).__name__}: {e}") from e

        return StepResult(
            observation=self._observe("", state),
            done=action.name in ("finish", "answer"),
            ok=ok,
            error=error,
        )

    def success(self) -> bool | None:
        """本环境不内建判分——判分是**任务**的事（官方的 `is_successful`）。"""
        return None

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:  # noqa: BLE001 - 关不掉不该影响结果
            pass
