"""安卓环境：通过 adb 控制模拟器或真机。

## 这一层做什么

把「一个 `Action`」翻译成 adb 命令，再把「当前屏幕」翻译回一个 `Observation`。
框架的其余部分完全不知道底下是安卓——脏活全关在这个文件里。

## 屏幕怎么变成文本

用的是 `uiautomator dump` 拿到的无障碍树。渲染成这种形态：

    [0] <android.widget.Button> "发送" 可点击 id=send_btn bounds=(940,2100,1180,2260)
    [1] <android.widget.EditText> "" 可编辑 id=msg_input bounds=(60,1990,900,2100)

**这是纯文本，没有图像。** 这一点很关键：级联的常驻监控器只吃文本
（原方法刻意的设计取舍，见 `monitors/base.py` 的说明），而这份无障碍树
正好是一份结构化的文字屏幕描述，不需要任何视觉信息。

同时也保留了截图（`Observation.image`）——那是给**策略模型**用的，
不是给监控器用的。两者用途不同，别混。

## 为什么用 uiautomator 而不是 accessibility forwarder

AndroidWorld 官方在模拟器上用 `-grpc 8554` 的 accessibility forwarder；
但 `uiautomator dump` **在真机和模拟器上都可用**，不依赖启动参数。
代价是慢一些（一次几百毫秒），换来的是"这份代码在真机上也能跑"。
延迟数据本来就要求把环境侧单列（见 types.py），所以慢多少是量得出来的。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..actions import ActionSpace, ActionSpec
from ..types import Action, Observation, StepResult

__all__ = ["AndroidEnv", "android_action_space", "find_adb"]

_DEVICE_XML = "/sdcard/guicascade_ui.xml"


def find_adb() -> str:
    """找一个能用的 adb。

    优先 PATH，其次 ANDROID_HOME/ANDROID_SDK_ROOT，最后几个常见的安装位置。
    """
    env = shutil.which("adb")
    if env:
        return env

    import os

    for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        root = os.environ.get(var)
        if root:
            p = Path(root) / "platform-tools" / ("adb.exe" if os.name == "nt" else "adb")
            if p.exists():
                return str(p)

    for p in (
        Path("D:/tools/android-sdk/platform-tools/adb.exe"),
        Path.home() / "Android/Sdk/platform-tools/adb",
    ):
        if p.exists():
            return str(p)

    raise FileNotFoundError("找不到 adb，请设置 ANDROID_HOME 或把它加进 PATH")


@dataclass
class AndroidEnv:
    """通过 adb 驱动一个安卓设备。"""

    serial: str = "emulator-5554"
    adb: str = field(default_factory=find_adb)
    task_package: str = ""
    """任务的 app 包名。reset 时会先关掉它再重启，保证起点干净。"""

    step_wait: float = 0.6
    """每个动作之后的固定等待。界面动画没结束就截图，拿到的是中间态。"""

    max_elements: int = 60
    """无障碍树里最多渲染多少个元素。全量渲染会长到几千字符，把提示词撑爆，
    而真正可操作的通常就是前面这些。"""

    capture_image: bool = True
    """每一步要不要截图。

    ⚠️ **这是本环境里最贵的一个开关**。实测（Pixel 6 / API 33 / x86_64）：

        adb input keyevent（执行动作）      ~300 ms
        adb exec-out screencap（截图）     ~2200 ms   ← 打开了就是这个数
        adb uiautomator dump（无障碍树）    ~2600 ms

    也就是说，**截图占了单步环境开销的近一半**。

    什么时候可以关掉：**策略模型和监控器都不看图像的时候**。
    级联的常驻监控器本来就只吃文本（原方法刻意的设计），所以只有
    "用视觉模型当策略"才真的需要截图。纯文本 agent 直接关掉，
    单步从 ~5.1s 降到 ~2.9s。
    """

    _space: ActionSpace | None = field(default=None, init=False, repr=False)
    _screen: tuple[int, int] | None = field(default=None, init=False, repr=False)

    name: str = field(default="android", init=False)

    # ------------------------------------------------------------------
    # adb 原语
    # ------------------------------------------------------------------

    def _run(self, *args: str, timeout: float = 60.0) -> bytes:
        proc = subprocess.run(
            [self.adb, "-s", self.serial, *args],
            capture_output=True,
            timeout=timeout,
        )
        if proc.returncode != 0 and proc.stderr:
            raise RuntimeError(f"adb {' '.join(args)} 失败: {proc.stderr.decode('utf-8', 'replace')[:200]}")
        return proc.stdout

    def screenshot(self) -> bytes:
        """当前屏幕的 PNG。

        `exec-out` 而不是 `shell`：后者会把二进制数据按文本处理，PNG 会被
        Windows 换行转换弄坏——这是个很容易踩、又很难查的坑。
        """
        return self._run("exec-out", "screencap", "-p", timeout=30)

    def ui_elements(self) -> list[dict[str, Any]]:
        """无障碍树里的元素列表。"""
        self._run("shell", "uiautomator", "dump", _DEVICE_XML, timeout=30)
        raw = self._run("shell", "cat", _DEVICE_XML, timeout=30).decode("utf-8", "replace")
        return _parse_ui_xml(raw)

    def screen_size(self) -> tuple[int, int]:
        """屏幕分辨率。**只查一次并缓存**——屏幕尺寸在一次运行里不会变，
        而每次 adb 往返要 260ms，每一步都查就是纯浪费。"""
        if self._screen is None:
            out = self._run("shell", "wm", "size").decode("utf-8", "replace")
            m = re.search(r"(\d+)x(\d+)", out)
            self._screen = (int(m.group(1)), int(m.group(2))) if m else (1080, 2400)
        return self._screen

    # ------------------------------------------------------------------
    # Environment 协议
    # ------------------------------------------------------------------

    def reset(self, task: str) -> Observation:
        if self.task_package:
            self._run("shell", "am", "force-stop", self.task_package)
            self._run("shell", "monkey", "-p", self.task_package,
                      "-c", "android.intent.category.LAUNCHER", "1", timeout=30)
            time.sleep(self.step_wait * 2)
        return self._observe(task)

    def step(self, action: Action) -> StepResult:
        try:
            self._dispatch(action)
            time.sleep(self.step_wait)
            ok, error = True, ""
        except Exception as e:  # noqa: BLE001 - 动作失败是常态，不该中断 episode
            ok, error = False, f"{type(e).__name__}: {e}"

        observation = self._observe("")
        return StepResult(
            observation=observation,
            done=action.name in ("finish", "answer"),
            ok=ok,
            error=error,
        )

    def success(self) -> bool | None:
        """本环境不内建判分。

        真实的判分要对设备状态做程序化检查（联系人表、闹钟数据库……），
        那属于**任务**的范畴，不属于**环境**。放在这里会让 env 变成
        "某个 benchmark 的 env"而不是"一个安卓 env"。判分交给外面的
        任务定义去做——见 `tasks/` 的说明。
        """
        return None

    def close(self) -> None:
        self._space = None  # 无长连接需要释放；留着这个钩子是为了接口一致

    # ------------------------------------------------------------------

    def _observe(self, task: str) -> Observation:
        elements = self.ui_elements()
        w, h = self.screen_size()
        return Observation(
            text=render_elements(elements, limit=self.max_elements),
            image=self.screenshot() if self.capture_image else None,
            meta={"task": task, "n_elements": len(elements), "w": w, "h": h},
        )

    def _dispatch(self, action: Action) -> None:
        a, args = action.name, action.args
        if a == "click":
            x, y = self._resolve_point(args)
            self._run("shell", "input", "tap", str(x), str(y))
        elif a == "double_tap":
            x, y = self._resolve_point(args)
            self._run("shell", "input", "tap", str(x), str(y))
            self._run("shell", "input", "tap", str(x), str(y))
        elif a == "long_press":
            x, y = self._resolve_point(args)
            self._run("shell", "input", "swipe", str(x), str(y), str(x), str(y), "1000")
        elif a == "input_text":
            # adb 的 input text 不认空格和中文，空格要转义
            text = str(args.get("text", "")).replace(" ", "%s")
            self._run("shell", "input", "text", text)
        elif a == "swipe":
            self._swipe(args)
        elif a == "scroll":
            self._swipe({"direction": args.get("direction", "down")})
        elif a == "navigate_home":
            self._run("shell", "input", "keyevent", "KEYCODE_HOME")
        elif a == "navigate_back":
            self._run("shell", "input", "keyevent", "KEYCODE_BACK")
        elif a == "keyboard_enter":
            self._run("shell", "input", "keyevent", "KEYCODE_ENTER")
        elif a == "open_app":
            pkg = args.get("app_name") or args.get("package", "")
            self._run("shell", "monkey", "-p", str(pkg), "-c",
                      "android.intent.category.LAUNCHER", "1", timeout=30)
        elif a == "wait":
            time.sleep(float(args.get("seconds", 1.0)))
        elif a in ("finish", "answer"):
            pass  # 终止类动作不碰设备，由上层收尾
        else:
            raise ValueError(f"安卓环境不认识动作 {a!r}")

    def _resolve_point(self, args: dict[str, Any]) -> tuple[int, int]:
        """把 index 或 x/y 解析成屏幕坐标。

        两种写法都支持是刻意的：**无障碍树给的 index 更稳**（不怕分辨率变化、
        不怕界面微调），但模型有时候就是会直接给坐标。两种都接住，比只认一种
        然后频繁报错要好。
        """
        if "x" in args and "y" in args:
            return int(args["x"]), int(args["y"])
        if "index" in args:
            idx = int(args["index"])
            elements = self.ui_elements()
            if not (0 <= idx < len(elements)):
                raise IndexError(f"元素序号 {idx} 越界（屏幕上有 {len(elements)} 个元素）")
            b = elements[idx]["bounds"]
            return (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
        raise ValueError("click 需要 index 或者 x/y")

    def _swipe(self, args: dict[str, Any]) -> None:
        w, h = self.screen_size()
        cx, cy = w // 2, h // 2
        d = args.get("direction", "down")
        span = h // 4
        moves = {
            "down": (cx, cy - span, cx, cy + span),
            "up": (cx, cy + span, cx, cy - span),
            "right": (cx - w // 4, cy, cx + w // 4, cy),
            "left": (cx + w // 4, cy, cx - w // 4, cy),
        }
        if d not in moves:
            raise ValueError(f"不认识的滑动方向 {d!r}")
        x1, y1, x2, y2 = moves[d]
        self._run("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), "300")


# --------------------------------------------------------------------------
# 无障碍树 -> 纯文本
# --------------------------------------------------------------------------


def _parse_ui_xml(raw: str) -> list[dict[str, Any]]:
    """把 `uiautomator dump` 的 XML 解析成元素列表。

    XML 前面可能混着 "UI hierchary dumped to: ..." 这类噪声（官方连
    hierarchy 都拼错了），所以先定位到第一个 `<` 再解析。
    """
    start = raw.find("<?xml")
    if start < 0:
        start = raw.find("<hierarchy")
    if start < 0:
        return []

    try:
        root = ET.fromstring(raw[start:])
    except ET.ParseError:
        return []

    out: list[dict[str, Any]] = []
    for node in root.iter("node"):
        bounds = _parse_bounds(node.get("bounds", ""))
        if bounds is None:
            continue
        out.append({
            "text": node.get("text", "") or "",
            "desc": node.get("content-desc", "") or "",
            "class": (node.get("class", "") or "").split(".")[-1],
            "resource_id": node.get("resource-id", "") or "",
            "clickable": node.get("clickable") == "true",
            "editable": node.get("class", "").endswith("EditText"),
            "scrollable": node.get("scrollable") == "true",
            "enabled": node.get("enabled") == "true",
            "bounds": bounds,
        })
    return out


def _parse_bounds(s: str) -> tuple[int, int, int, int] | None:
    m = re.match(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]", s)
    if not m:
        return None
    return tuple(int(g) for g in m.groups())  # type: ignore[return-value]


def render_elements(elements: list[dict[str, Any]], *, limit: int = 60) -> str:
    """把元素列表渲染成给模型看的纯文本。

    只保留**能交互或带文字**的元素——纯布局容器（FrameLayout/LinearLayout）
    对模型没有信息量，渲染出来只是噪声，还会挤掉真正有用的那几十个。

    ⚠️ 这段文本会原样进提示词，所以它必须包在 `<screen source="device">` 里
    （由 `prompts.py` 负责），而且**不要在这里拼任何指令性文字**——
    屏幕内容是不可信输入，见 prompts.py 关于提示注入的说明。
    """
    lines = []
    shown = 0
    for i, e in enumerate(elements):
        interesting = e["clickable"] or e["editable"] or e["scrollable"] or e["text"] or e["desc"]
        if not interesting:
            continue
        if shown >= limit:
            lines.append(f"...（还有 {len(elements) - i} 个元素未显示）")
            break

        label = e["text"] or e["desc"]
        flags = []
        if e["clickable"]:
            flags.append("可点击")
        if e["editable"]:
            flags.append("可编辑")
        if e["scrollable"]:
            flags.append("可滚动")
        if not e["enabled"]:
            flags.append("已禁用")

        ident = f" id={e['resource_id'].split('/')[-1]}" if e["resource_id"] else ""
        x1, y1, x2, y2 = e["bounds"]
        lines.append(
            f"[{shown}] <{e['class']}> {label!r}{'' if not flags else ' ' + ' '.join(flags)}"
            f"{ident} bounds=({x1},{y1},{x2},{y2})"
        )
        shown += 1
    return "\n".join(lines) if lines else "(屏幕上没有可交互元素)"


# --------------------------------------------------------------------------
# 动作空间
# --------------------------------------------------------------------------


def android_action_space() -> ActionSpace:
    """安卓的 13 个动作。

    描述文字按提示工程的消融结论写：**说清"什么时候用"和"什么时候别用"**。
    删掉描述、只留参数定义会让调用错误率涨四成以上。

    `positional` 是为了救 `click(3)` 这种省略参数名的写法——模型经常这么写。
    """
    space = ActionSpace(name="android")
    _add = space.add

    _add(ActionSpec(
        name="click",
        description="点击一个元素。优先用 index（来自屏幕列表的序号），它比坐标稳。"
                    "只有在元素列表里找不到目标时才用 x/y 坐标。",
        parameters={
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "屏幕列表中元素的序号，如 3"},
                "x": {"type": "integer", "description": "屏幕 x 坐标（仅在没有 index 时使用）"},
                "y": {"type": "integer", "description": "屏幕 y 坐标（仅在没有 index 时使用）"},
            },
            "required": [],
        },
        positional=("index",),
    ))
    _add(ActionSpec(
        name="input_text",
        description="往当前聚焦的输入框里输入文字。用之前一般要先 click 那个输入框。"
                    "会覆盖原有内容，所以输入前通常需要先全选清空。",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要输入的文字"}},
            "required": ["text"],
        },
        positional=("text",),
    ))
    _add(ActionSpec(
        name="scroll",
        description="在当前页面上滑动。目标内容不在屏幕内时用这个，不要盲目点击坐标。",
        parameters={
            "type": "object",
            "properties": {"direction": {"type": "string", "enum": ["up", "down", "left", "right"]}},
            "required": ["direction"],
        },
        positional=("direction",),
    ))
    _add(ActionSpec(
        name="swipe",
        description="从一点拖到另一点，用于调节滑块、拖拽条目这类连续操作。"
                    "一般的翻页请用 scroll。",
        parameters={
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
            },
            "required": ["direction"],
        },
    ))
    _add(ActionSpec(name="navigate_home", description="回到桌面。用于从当前 app 退出、重新开始。"))
    _add(ActionSpec(name="navigate_back", description="返回上一级。弹窗挡住时也用它关掉。"))
    _add(ActionSpec(name="keyboard_enter", description="按回车键，常用于提交搜索或确认输入。"))
    _add(ActionSpec(
        name="long_press",
        description="长按一个元素，用于触发上下文菜单或进入编辑模式。",
        parameters={
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "元素序号"},
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": [],
        },
        positional=("index",),
    ))
    _add(ActionSpec(
        name="double_tap",
        description="双击一个元素，用于打开文件、放大内容。",
        parameters={
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "元素序号"},
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": [],
        },
        positional=("index",),
    ))
    _add(ActionSpec(
        name="open_app",
        description="通过包名启动一个 app。只在任务需要切换 app 时使用。",
        parameters={
            "type": "object",
            "properties": {"app_name": {"type": "string", "description": "app 的包名，如 com.android.settings"}},
            "required": ["app_name"],
        },
        positional=("app_name",),
    ))
    _add(ActionSpec(
        name="wait",
        description="等待若干秒。只在页面明显还在加载时使用，不要用它代替思考。",
        parameters={
            "type": "object",
            "properties": {"seconds": {"type": "number", "description": "等待秒数"}},
            "required": [],
        },
        positional=("seconds",),
    ))
    _add(ActionSpec(
        name="answer",
        description="任务要求回答问题（而不是操作界面）时，用它提交答案。",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        positional=("text",),
    ))
    return space
