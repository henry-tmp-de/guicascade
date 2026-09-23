"""把 AndroidWorld 的任务接到我们自己的安卓执行层上。

## 这是什么

AndroidWorld（google-research/android_world）有 116 个任务，判分逻辑是作者
在真机上反复调过的。我们自己编的任务在这一环上翻了四次车（电话号码被系统
格式化、界面文案假设错误、`and`/`or` 优先级写错……），所以改用它。

问题是：**AndroidWorld 的任务代码假定 `env` 是它家的 `AsyncEnv`**，而
我们有自己的 `AndroidEnv`（adbd 那一套）。直接拿过去会崩。

这个文件就是中间那层翻译。从任务代码的角度看，它是个正常的 AndroidWorld
env；实际上它内部把每个调用转发给我们自己的 adb。

## 为什么这个"假 env"是正当的，不是作弊

**它没有改动任何判分逻辑。** 判分代码还是 AndroidWorld 原版，一行没动。
我们只是给它换了条"查设备状态"的通道：从它家的 gRPC accessibility forwarder，
换成我们的 `adb shell`。

类比：同一个 SQL 查询，换一个数据库驱动去执行。查询本身没变。

## 为什么这层能写得这么薄

`adb_utils` 里**对传入对象只调用一个方法**——`execute_adb_call`，
11 处调用点全是它（`grep -oE "\\benv\\.[a-z_]+\\(" env/adb_utils.py` 的结果）。
所以只要把这一个方法实现对，绝大多数判分就通了。

## 依赖

    pip install android_env==1.2.3     # 只为拿 adb_pb2 类型定义

**不需要起它那套 gRPC 服务，也不需要它定制的模拟器镜像。**

以及一份 android_world 源码（只要 Python 源码，不用装）：

    git clone --depth 1 https://github.com/google-research/android_world
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["AndroidWorldBridge", "load_android_world"]

_AW_HINT = (
    "找不到 android_world 源码。先克隆一份：\n"
    "    git clone --depth 1 https://github.com/google-research/android_world\n"
    "然后用 --aw-path 指向它，或设环境变量 ANDROID_WORLD_PATH。"
)


def load_android_world(aw_path: str | Path) -> dict[str, Any]:
    """把 android_world 的源码目录挂上 sys.path，返回需要的那些模块。

    **不 install、不 pip**：只用源码里的 Python 文件。理由是这个仓库依赖
    很重（dm_env、absl、protobuf…），而我们其实只用到其中的 task 定义和
    `adb_utils`，装一遍成本高、收益低。
    """
    p = Path(aw_path).resolve()
    if not (p / "android_world").is_dir():
        raise FileNotFoundError(f"{p} 下面没有 android_world/ 目录。\n{_AW_HINT}")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

    try:
        from android_env.proto import adb_pb2  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - 环境问题，不是逻辑问题
        raise ImportError(
            "需要 adb_pb2 类型定义：pip install android_env==1.2.3"
        ) from e

    from android_world.env import representation_utils  # noqa: PLC0415

    return {"adb_pb2": adb_pb2, "representation_utils": representation_utils}


class AndroidWorldBridge:
    """同时扮演 AndroidWorld 的 `env` 和 `controller`。

    任务代码两种写法都有：

        adb_utils.issue_generic_request([...], env.controller)   # 传 controller
        ui_elements = env.get_state().ui_elements                # 用 env 本身

    而 `adb_utils` 最终要的只是 `execute_adb_call`。所以**一个类同时当两者**
    最省事：`controller` 属性返回自己，两边都接得住。
    """

    def __init__(self, adb: str, serial: str = "emulator-5554", *, aw_path: str | Path):
        mods = load_android_world(aw_path)
        self._adb_pb2 = mods["adb_pb2"]
        self._ru = mods["representation_utils"]

        self.adb = adb
        self.serial = serial

        # 任务代码会写这个属性（`env.interaction_cache = ""`）。
        # 它用来在"Agent 回答用户提问"这类场景里暂存交互，
        # 对判分没有影响，给个可写的普通属性就够了。
        self.interaction_cache: str = ""

        self._screen: tuple[int, int] | None = None
        self._elements: list[dict[str, Any]] | None = None

    # ==================================================================
    # controller 面 —— 唯一真正重要的方法
    # ==================================================================

    def execute_adb_call(self, request) -> Any:
        """AndroidWorld 所有的设备读写都从这里走。**这一层翻译的核心。**

        按 `request.command`（proto 的 oneof 字段）分派。实测全仓用到的变体
        只有下面这几种，按运行时频率排：generic 最多（`issue_generic_request`
        底下就是它），其余是辅助。
        """
        adb_pb2 = self._adb_pb2
        kind = request.WhichOneof("command")
        timeout = getattr(request, "timeout_sec", 0) or 30
        if timeout <= 0:
            timeout = 30

        def ok(**kw):
            return adb_pb2.AdbResponse(status=adb_pb2.AdbResponse.OK, **kw)

        def fail(msg: str, status=None):
            return adb_pb2.AdbResponse(
                status=status or adb_pb2.AdbResponse.ADB_ERROR,
                error_message=str(msg)[:500],
            )

        try:
            # ---- 主力：任意 shell 命令 ----
            # 判分里读设备状态（settings get / content query / dumpsys）
            # 走的全是这一条。
            if kind == "generic":
                args = list(request.generic.args)
                out = self._shell(*args, timeout=timeout)
                return ok(generic=adb_pb2.AdbResponse.GenericResponse(output=out))

            if kind == "settings":
                return ok(settings=self._settings(request.settings, timeout))

            if kind == "press_button":
                key = request.press_button.button
                name = adb_pb2.PressButtonRequest.KeyEvent.Name(key)
                self._shell("shell", "input", "keyevent", name.split(".")[-1], timeout=timeout)
                return ok(press_button=adb_pb2.AdbResponse.PressButtonResponse())

            if kind == "input_text":
                # ⚠️ adb 的 `input text` 不认非 ASCII 也不认空格。
                # 空格要转义成 %s；中文从 Android 8 起会直接抛异常（见 README）。
                # AndroidWorld 的任务基本用 ASCII，这里够用；
                # 真要做中文得换 ADBKeyboard，那是另一件事。
                text = str(request.input_text.text).replace(" ", "%s")
                self._shell("shell", "input", "text", text, timeout=timeout)
                return ok(input_text=adb_pb2.AdbResponse.InputTextResponse())

            if kind == "tap":
                self._shell("shell", "input", "tap", str(request.tap.x), str(request.tap.y),
                            timeout=timeout)
                return ok()

            if kind == "start_activity":
                a = request.start_activity
                self._shell("shell", "am", "start", "-n", f"{a.activity}/{a.activity}"
                            if "." in a.activity else a.activity, timeout=timeout)
                return ok(start_activity=adb_pb2.AdbResponse.StartActivityResponse())

            if kind == "get_current_activity":
                out = self._shell("shell", "dumpsys", "window", "displays", timeout=timeout)
                m = re.search(rb"mCurrentFocus=Window\{[^}]*?\s([\w\.]+/[\w\.]+)", out)
                full = m.group(1).decode() if m else ""
                return ok(get_current_activity=adb_pb2.AdbResponse
                          .GetCurrentActivityResponse(full_activity=full))

            if kind == "get_orientation":
                out = self._shell("shell", "dumpsys", "window", "displays", timeout=timeout)
                m = re.search(rb"mCurrentRotation=ROTATION_(\d+)", out)
                rot = int(m.group(1)) if m else 0
                return ok(get_orientation=adb_pb2.AdbResponse.GetOrientationResponse(
                    orientation=rot))

            if kind == "package_manager":
                return ok(package_manager=self._package_manager(request.package_manager,
                                                                timeout))

            if kind == "dumpsys":
                out = self._shell("shell", "dumpsys", timeout=timeout)
                return ok(dumpsys=adb_pb2.AdbResponse.DumpsysResponse(output=out))

            if kind in ("push", "pull"):
                # 文件传输：判分里用得少，先返回"未实现"而不是假装成功——
                # **假装成功比报错更糟**：判分会读到一个空文件然后判负，
                # 看起来像模型的问题。
                return fail(f"桥接层暂未实现 {kind}", adb_pb2.AdbResponse.UNKNOWN_COMMAND)

            if kind in ("install_apk", "uninstall_package", "force_stop"):
                return fail(f"桥接层暂未实现 {kind}", adb_pb2.AdbResponse.UNKNOWN_COMMAND)

            return fail(f"不认识的 adb 请求：{kind!r}",
                        adb_pb2.AdbResponse.UNKNOWN_COMMAND)

        except subprocess.TimeoutExpired:
            # 超时要**明确报 TIMEOUT**，不要伪装成成功或普通错误——
            # 上层能据此重试，也能在统计里把"环境抖动"和"判负"分开
            return fail(f"adb 超时：{kind}", adb_pb2.AdbResponse.TIMEOUT)
        except Exception as e:  # noqa: BLE001
            return fail(f"{type(e).__name__}: {e}")

    @property
    def controller(self) -> "AndroidWorldBridge":
        """任务代码会 `adb_utils.xxx(env.controller)`。

        `adb_utils` 最终只调 `execute_adb_call`，所以返回自己就行——
        不需要真的有一个独立的 controller 对象。
        """
        return self

    # ==================================================================
    # env 面 —— 判分里只有少数任务用到
    # ==================================================================

    def get_state(self, wait_to_stabilize: bool = False) -> Any:
        """当前屏幕：截图 + 无障碍树 + 解析出的 UIElement 列表。

        ⚠️ **116 个任务里只有 5 个文件用 `ui_elements`**（其余全走 adb 判分）。
        所以这一支先做到"字段齐全、够用"，不追求和它家完全等价。
        """
        state_cls = self._state_class()
        forest = self._dump_forest()
        elements = self._to_ui_elements(forest)
        return state_cls(pixels=self._screenshot(), forest=forest,
                         ui_elements=elements)

    def reset(self, go_home: bool = False) -> Any:
        if go_home:
            self._shell("shell", "input", "keyevent", "KEYCODE_HOME")
        self.interaction_cache = ""
        self._elements = None
        time.sleep(0.6)
        return self.get_state()

    def execute_action(self, action) -> None:
        """执行一个 AndroidWorld 的 JSONAction。

        **这条路径我们用不到**——我们的 Agent 自己产生动作、自己执行，
        不用它家的 agent。留着是为了接口完整，将来想跑它家的 baseline
        （M3A / SeeAct）时能直接接上。
        """
        raise NotImplementedError(
            "桥接层只用于复用 AndroidWorld 的**任务与判分**；"
            "执行动作走我们自己的 Agent。要跑它家 agent 的话在这里接 actuation。"
        )

    def display_message(self, message: str, header: str = "") -> None:
        """它家会在屏幕上叠一层状态提示，我们没有，忽略即可。"""

    def ask_question(self, question: str, timeout_seconds: float = -1.0):
        raise NotImplementedError

    def hide_automation_ui(self) -> None:
        self._shell("shell", "settings", "put", "system", "pointer_location", "0")

    def close(self) -> None:
        self._elements = None

    @property
    def foreground_activity_name(self) -> str:
        out = self._shell("shell", "dumpsys", "window", "displays")
        m = re.search(rb"mCurrentFocus=Window\{[^}]*?\s([\w\.]+/[\w\.]+)", out)
        return m.group(1).decode() if m else ""

    @property
    def device_screen_size(self) -> tuple[int, int]:
        return self._screen_size()

    @property
    def logical_screen_size(self) -> tuple[int, int]:
        # 竖屏时两者相同。我们不做横屏适配，所以直接返回物理尺寸——
        # **不要在这里假装能处理旋转**，免得判分用的坐标和实际不一致。
        return self._screen_size()

    @property
    def orientation(self) -> int:
        out = self._shell("shell", "dumpsys", "window", "displays")
        m = re.search(rb"mCurrentRotation=ROTATION_(\d+)", out)
        return int(m.group(1)) if m else 0

    @property
    def physical_frame_boundary(self) -> tuple[int, int, int, int]:
        w, h = self._screen_size()
        return (0, 0, w, h)

    # ==================================================================
    # 内部
    # ==================================================================

    def _shell(self, *args: str, timeout: float = 30) -> bytes:
        """跑一条 adb 命令（`adb -s <serial> <args...>`），返回**原始字节**。

        ⚠️ **本函数不自动加 `shell`。** 因为 AndroidWorld 的
        `issue_generic_request` 传进来的 args **本身就带 `shell`**：

            issue_generic_request(['shell', 'settings', 'get', ...], env)

        早先这里又加了一次，实际执行成 `adb shell shell settings get ...`，
        设备上根本没有 `shell` 这个命令，输出是空串。
        症状很隐蔽：**判分拿到空字符串，`int('')` 抛异常**，
        看起来像"判分代码有 bug"，其实是我们的翻译层多插了一层。

        返回 bytes 而不是 str 是刻意的：`issue_generic_request` 的结果会被
        `.decode()`，我们提前解码一次反而制造了两次转码的机会。
        二进制安全，交给调用方决定怎么解。
        """
        proc = subprocess.run(
            [self.adb, "-s", self.serial, *[str(a) for a in args]],
            capture_output=True, timeout=timeout,
        )
        return proc.stdout

    def _settings(self, req, timeout: float):
        adb_pb2 = self._adb_pb2
        R = adb_pb2.AdbResponse.SettingsResponse
        ns = req.name_space
        if req.get:
            out = self._shell("shell", "settings", "get", ns, req.get, timeout=timeout)
            return R(get=out.decode("utf-8", "replace").strip())
        if req.put:
            self._shell("shell", "settings", "put", ns, req.put, req.value, timeout=timeout)
            return R()
        if req.delete_key:
            self._shell("shell", "settings", "delete", ns, req.delete_key, timeout=timeout)
            return R()
        return R()

    def _package_manager(self, req, timeout: float):
        adb_pb2 = self._adb_pb2
        R = adb_pb2.AdbResponse.PackageManagerResponse
        if req.list:
            out = self._shell("shell", "pm", "list", "packages", timeout=timeout)
            return R(list=out.decode("utf-8", "replace"))
        if req.clear:
            self._shell("shell", "pm", "clear", req.clear, timeout=timeout)
            return R()
        return R()

    def _screen_size(self) -> tuple[int, int]:
        if self._screen is None:
            out = self._shell("shell", "wm", "size").decode("utf-8", "replace")
            m = re.search(r"(\d+)x(\d+)", out)
            self._screen = (int(m.group(1)), int(m.group(2))) if m else (1080, 2400)
        return self._screen

    def _dump_forest(self) -> list:
        """dump 无障碍树并解析成节点列表。"""
        self._shell("shell", "uiautomator", "dump", "/sdcard/_aw_bridge.xml")
        raw = self._shell("shell", "cat", "/sdcard/_aw_bridge.xml").decode("utf-8", "replace")
        start = raw.find("<?xml")
        if start < 0:
            start = raw.find("<hierarchy")
        if start < 0:
            return []
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(raw[start:])
        except ET.ParseError:
            return []
        return list(root.iter("node"))

    def _to_ui_elements(self, forest: list) -> list:
        """把无障碍节点转成 AndroidWorld 的 UIElement。

        字段名对齐它们的 dataclass（`content_description` 而不是我们的 `desc`，
        `bbox` 是 BoundingBox 对象不是四元组）。**只填它们真的会读的那几个**，
        其余留 None——填错比留空更危险。
        """
        ru = self._ru
        out = []
        for n in forest:
            b = n.get("bounds", "")
            m = re.match(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]", b)
            if not m:
                continue
            x1, y1, x2, y2 = (int(g) for g in m.groups())
            box = ru.BoundingBox(x_min=x1, y_min=y1, x_max=x2 - x1, y_max=y2 - y1)
            out.append(ru.UIElement(
                text=n.get("text") or None,
                content_description=n.get("content-desc") or None,
                class_name=n.get("class") or None,
                bbox=box,
                bbox_pixels=box,
                is_clickable=n.get("clickable") == "true",
                is_editable="EditText" in (n.get("class") or ""),
                is_enabled=n.get("enabled", "true") == "true",
                is_scrollable=n.get("scrollable") == "true",
                is_focused=n.get("focused") == "true",
                is_checkable=n.get("checkable") == "true",
                is_checked=n.get("checked") == "true",
                is_selected=n.get("selected") == "true",
                package_name=n.get("package") or None,
                resource_name=(n.get("resource-id") or None),
            ))
        return out

    def _screenshot(self):
        """当前屏幕的像素。取不到就返回 None。

        ⚠️ 返回 None 而不是抛异常：**绝大多数判分不看像素**，
        为了一个用不到的字段把整条判分链搞崩，不划算。
        """
        try:
            import io

            import numpy as np
            from PIL import Image

            png = subprocess.run(
                [self.adb, "-s", self.serial, "exec-out", "screencap", "-p"],
                capture_output=True, timeout=30,
            ).stdout
            return np.array(Image.open(io.BytesIO(png)).convert("RGB"))
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _state_class():
        """它家的 `State` dataclass。延迟 import——只有 get_state 用得到。"""
        from android_world.env.interface import State  # noqa: PLC0415

        return State
