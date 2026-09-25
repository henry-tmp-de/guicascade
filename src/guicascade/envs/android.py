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
from typing import Any, Mapping

from ..actions import ActionSpace, ActionSpec
from ..types import Action, Observation, StepResult

__all__ = ["AndroidEnv", "android_action_space", "find_adb", "load_apps"]

_DEVICE_XML = "/sdcard/guicascade_ui.xml"


def load_apps(path: str | Path = "") -> dict[str, str]:
    """读「应用显示名 -> 包名」表。默认读两份并**合并**，先读到的优先：

        configs/apps.yaml      手工 + 扫描，权威，有中文别名
        configs/apps_aw.yaml   由 `gen_apps.py` 从 AndroidWorld 的表生成

    分成两份是刻意的：生成的那份会随设备反复重写，而手工那份里有解释性的
    注释和中文别名，不能被生成器冲掉。

    **读不到就返回空表，而不是抛异常。** 理由是退化的后果很轻：`open_app`
    依然接受包名，只是模型不能再用名字指代应用。为了缺一个配置文件就让
    整个环境起不来，不划算。

    传了 `path` 就只读那一份（调试用），不合并。

    ⚠️ 这张表**漏了应用不是"少几个名字"这么轻**：查不到会抛异常，异常被
    `env.step` 吞进 `StepResult.error`，而那个 error 不进提示词 —— 模型于是
    重复同一个动作直到步数用完。实测漏了 17 个官方应用（录音机、Broccoli、
    VLC…），这是"模型在打转"的主要来源。补表用 `python scripts/gen_apps.py`。

    表是**设备数据，不是源代码**，换设备必须重新生成。见 `AndroidEnv.apps`。
    """
    if path:
        return _read_apps_file(Path(path))

    cfg = Path(__file__).resolve().parents[3] / "configs"
    out: dict[str, str] = {}
    for name in ("apps.yaml", "apps_aw.yaml"):
        for k, v in _read_apps_file(cfg / name).items():
            out.setdefault(k, v)      # 先读到的赢：手工表压过生成表
    return out


def _read_apps_file(p: Path) -> dict[str, str]:
    """读一个 YAML 的 apps 段。坏了就当空表——缺个配置不该让环境起不来。"""
    if not p.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}
    apps = data.get("apps")
    return {str(k): str(v) for k, v in apps.items()} if isinstance(apps, dict) else {}


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
    """任务的 app 包名。`reset` 时会把它强制停止，保证起点干净。"""

    apps: Mapping[str, str] = field(default_factory=dict)
    """应用显示名 -> 包名。**这是环境的属性，不是模型的知识。**

    ⚠️ 这张表**绝不能写进提示词**。踩过的坑：早先把这张表塞在 `open_app`
    的参数说明里，模型于是完全不看屏幕——直接拿任务里的"时钟"两个字去表里
    查包名，一步到位。整个 benchmark 退化成查表题，而我们还以为在测 GUI 能力。

    表本身是必要的（"设置"对应哪个包，这是设备的事实，不是推理能得出的），
    但正确的做法是**模型说名字、环境去查**，模型不该看见这张表。

    表从哪来：`configs/apps.yaml`，由 `scripts/scan_apps.py` 扫描设备自动生成。
    换设备必须重扫——这正是它属于环境配置而不是源码的原因。
    """

    _app_index: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    launch_on_reset: bool = False
    """`reset` 时要不要顺带把 `task_package` 启动起来。

    ⚠️ 默认 **False**，这是个踩过坑的默认值。

    对"打开某个 app"这类任务，reset 时把它打开等于**把答案直接送给模型**：
    模型第一步看到目标界面，直接输出 `finish`，程序化判分在第一秒就通过——
    但 agent 什么都没做。这个假成功极难发现，因为**每个环节看起来都正常**。

    默认停在桌面，让 agent 真的去走这一步。只有任务确实是"在这个 app 内部
    做点什么"时，才把它设成 True。
    """

    go_home_on_reset: bool = True
    """`reset` 时是否先回桌面。起点不确定会让不同 run 之间没法比较。"""

    step_wait: float = 0.6
    """普通动作（点击、滑动）之后的等待。"""

    launch_wait: float = 3.0
    """启动 app 之后的等待。**比普通动作长得多，这个差别很关键。**

    启动一个 app 要 1~3 秒才把界面画出来，而点击只要几百毫秒。
    如果统一用 0.6 秒，`open_app` 之后拿到的还是**上一个界面**——
    模型会以为自己的动作没生效，于是原地重试同一动作，
    看起来像"模型卡住了"，实际是**环境返回得太早**。

    实测踩过：模型第一步就用 open_app 成功打开了设置，但因为看到的还是
    桌面，它把同一个动作重复了 8 次。那条轨迹被标成"卡住"，
    但根因在环境不在模型——**这种假样本会污染监控器的评测集。**
    """

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
    _elements: list[dict[str, Any]] | None = field(default=None, init=False, repr=False)
    _actionable: list[dict[str, Any]] | None = field(default=None, init=False, repr=False)
    """**模型看到的那份列表，也是 `click(index=N)` 里 N 的所指。**

    和 `_elements`（全量无障碍树）一起缓存、一起失效。分开缓存是必要的：
    索引契约要求"模型看到的"和"执行时取的"是同一个列表，随手重算一遍
    就会错位——这个 bug 代价极大且极难发现。
    """
    _elements_at: float = field(default=0.0, init=False, repr=False)

    elements_ttl: float = 3.0
    """无障碍树的缓存有效期（秒）。

    ⚠️ **这个缓存值 2.5 秒/步**，因为它避免了一次完全重复的 `uiautomator dump`：
    `click(index=N)` 需要知道第 N 个元素的坐标，而观察时刚刚 dump 过整棵树。
    不缓存的话每个点击步骤都要白等 2.5 秒。

    用 TTL 而不是"永久缓存"是因为**动作会改变屏幕**：点完之后整棵树就过期了。
    3 秒足够覆盖"观察 -> 决策 -> 执行"这一个来回，又短到不会让两次动作之间
    误用旧树。
    """

    name: str = field(default="android", init=False)

    def __post_init__(self) -> None:
        # 名字大小写不敏感：模型有时写 "Clock"、有时写 "clock"
        self._app_index = {
            str(k).strip().lower(): v for k, v in (self.apps or {}).items()
        }

    def resolve_app(self, name: str) -> str:
        """把模型给的应用名解析成包名。**只看名字，不看提示词。**

        接受两种写法：

        - 已经是包名（含 `.`）-> 原样使用。这样即使这张表空了，
          `open_app` 依然可用，环境也不会因为缺配置而瘫掉。
        - 显示名 -> 查 `apps` 表。中英文都行（表里两种键都登记）。

        查不到就抛一个**说清楚有哪些可选**的错。这句话会回灌给模型，
        让它能自己改——报个"KeyError"它只能瞎猜。
        """
        q = str(name or "").strip()
        if not q:
            raise ValueError("open_app 需要 app_name，你什么也没给")

        if "." in q and " " not in q:
            return q

        hit = self._app_index.get(q.lower())
        if hit:
            return hit

        # 报错里列**原始写法**（Clock / 时钟），不是内部小写索引。
        # 这句话是给模型看的，给它一份全小写的清单会诱导它继续写小写。
        #
        # ⚠️ 清单要**截断**。表补全之后有 100 多条，全列出来是上千字符，
        # 而这句话会进提示词的 `Result:` 行 —— 会把屏幕内容挤掉，得不偿失。
        # 表里没有的名字本来就不该指望靠报错去撞。
        names = sorted(self.apps)
        known = ", ".join(names[:24]) if names else "（本设备未登记任何应用名，请直接用包名）"
        if len(names) > 24:
            known += f" …（共 {len(names)} 个）"
        raise ValueError(f"不认识的应用名 {name!r}。可直接用包名，或用以下任一名字：{known}")

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

    dump_retries: int = 2
    """`uiautomator dump` 失败时重试几次。

    ⚠️ **必须有这个重试。** `uiautomator dump` 在模拟器上会偶发超时
    （实测踩过：`TimeoutExpired` 直接冒泡出去，**把整条 episode 打死**，
    前面十几步全白跑）。

    它只是个瞬时抖动——重试一次基本就好。把一个可恢复的环境抖动
    升级成任务失败，是最不划算的一种失败：数据里看不出区别，
    只表现为"模型成功率莫名偏低"。
    """

    def ui_elements(self, *, fresh: bool = False) -> list[dict[str, Any]]:
        """无障碍树里的元素列表。默认走缓存，见 `elements_ttl`。"""
        now = time.monotonic()
        if (not fresh and self._elements is not None
                and now - self._elements_at < self.elements_ttl):
            return self._elements

        # 重试只针对 dump 这一段。**解析失败不重试**——那说明 dump 出来的
        # 内容本身有问题（比如屏幕上有动画、拿到半截 XML），重试也一样。
        last: Exception | None = None
        for attempt in range(self.dump_retries + 1):
            try:
                self._run("shell", "uiautomator", "dump", _DEVICE_XML, timeout=30)
                raw = self._run("shell", "cat", _DEVICE_XML, timeout=30).decode("utf-8", "replace")
                els = _parse_ui_xml(raw)
                if els:
                    self._elements = els
                    self._elements_at = time.monotonic()
                    return self._elements
                last = RuntimeError("dump 解析出来是空的")
            except Exception as e:  # noqa: BLE001 - 超时、adb 断连都算
                last = e
            if attempt < self.dump_retries:
                time.sleep(1.0 + attempt)

        raise RuntimeError(
            f"uiautomator dump 连续 {self.dump_retries + 1} 次失败：{last}。"
            "模拟器可能卡住了，考虑 `adb reboot`。"
        )

    def actionable(self) -> list[dict[str, Any]]:
        """模型能操作的元素列表——**和 `_observe` 渲染出去的必须是同一份**。

        `click(index=N)` 里的 N 就是这个列表的下标。走到这里说明点击时机
        通常紧跟在观察之后（TTL 内），缓存命中，两份自然一致；即使缓存
        过期重新 dump，也是从同一棵新树重新算出来的，仍然自洽。
        """
        if self._actionable is None:
            self._actionable = actionable_elements(
                self.ui_elements(), limit=self.max_elements
            )
        return self._actionable

    def invalidate(self) -> None:
        """让缓存的屏幕信息立即失效。

        执行完动作后调用——**点了之后屏幕就变了，旧的元素树和坐标都不能再用**。
        """
        self._elements = None
        self._actionable = None
        self._elements_at = 0.0

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
        """把设备恢复到确定的起点。

        顺序是刻意的：先停掉目标任务（清掉残留状态，否则上一次 run 的界面
        会留在屏幕上），再回桌面。两步都做才叫"起点确定"——
        不同 run 从不同起点出发，结果之间就没法比较。
        """
        if self.task_package:
            self._run("shell", "am", "force-stop", self.task_package)
        if self.go_home_on_reset:
            self._run("shell", "input", "keyevent", "KEYCODE_HOME")
        time.sleep(self.step_wait)

        if self.launch_on_reset and self.task_package:
            self._run("shell", "monkey", "-p", self.task_package,
                      "-c", "android.intent.category.LAUNCHER", "1", timeout=30)
            time.sleep(self.step_wait * 2)

        return self._observe(task)

    def step(self, action: Action) -> StepResult:
        try:
            self._dispatch(action)
            # 启动类动作要多等：界面还没画出来就取观察，等于让模型看旧屏幕
            wait = self.launch_wait if action.name == "open_app" else self.step_wait
            time.sleep(wait)
            ok, error = True, ""
        except Exception as e:  # noqa: BLE001 - 动作失败是常态，不该中断 episode
            ok, error = False, f"{type(e).__name__}: {e}"
        finally:
            # 不管成没成，屏幕都可能变了（点了个不存在的元素也可能触发了滚动）
            self.invalidate()

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
        # 走 actionable() 而不是自己过滤一遍：渲染出去的元素和 click 解析用的
        # 元素**必须是同一个列表**，否则序号错位。
        elements = self.actionable()
        w, h = self.screen_size()
        return Observation(
            text=render_elements(elements),
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
            pkg = self.resolve_app(args.get("app_name") or args.get("package", ""))
            self._run("shell", "monkey", "-p", pkg, "-c",
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

        ⚠️ `index` 查的是 `actionable()`——**模型看到的那个列表**，不是全量
        无障碍树。取错列表会让序号整体错位（见 `actionable_elements` 的说明）。
        """
        if "x" in args and "y" in args:
            return int(args["x"]), int(args["y"])
        if "index" in args:
            idx = int(args["index"])
            elements = self.actionable()
            if not (0 <= idx < len(elements)):
                raise IndexError(
                    f"元素序号 {idx} 越界（屏幕上可操作的元素是 0~{len(elements) - 1}）"
                )
            return elements[idx]["_click_xy"]
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


_EDITABLE_HINTS = ("EditText", "AutoCompleteTextView", "SearchView")
"""判断"能不能输入"。

不用 `class.endswith("EditText")`：那只认得住 `android.widget.EditText`，
遇到 `AppCompatEditText`、`TextInputEditText`、`AutoCompleteTextView`
（搜索框基本都是这个）就漏了。**换成子串匹配是刻意放宽**——
把不可输入的认成可输入，模型顶多白试一次；反过来漏掉一个输入框，
它就只能干瞪眼。"""


def _looks_editable(cls: str) -> bool:
    return any(h in cls for h in _EDITABLE_HINTS)


def _parse_ui_xml(raw: str) -> list[dict[str, Any]]:
    """把 `uiautomator dump` 的 XML 解析成元素列表。

    XML 前面可能混着 "UI hierchary dumped to: ..." 这类噪声（官方连
    hierarchy 都拼错了），所以先定位到第一个 `<` 再解析。

    **每个元素记下自己的祖先链**（`_ancestors`，由近及远）。界面上真正挂着
    点击事件的，经常是包住文字的那层容器而不是文字本身，点哪里要靠它算。
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

    def visit(node: ET.Element, ancestors: tuple[int, ...]) -> None:
        bounds = _parse_bounds(node.get("bounds", ""))
        if bounds is None:
            return
        idx = len(out)
        out.append({
            "text": node.get("text", "") or "",
            "desc": node.get("content-desc", "") or "",
            "class": (node.get("class", "") or "").split(".")[-1],
            "resource_id": node.get("resource-id", "") or "",
            "clickable": node.get("clickable") == "true",
            "editable": _looks_editable(node.get("class", "") or ""),
            "scrollable": node.get("scrollable") == "true",
            # ⚠️ 缺省必须是 **true**（fail-open），不能是 false。
            #
            # 有些机型的 dump 不带 `enabled` 属性，写成 `== "true"` 会把它们
            # 全判成"已禁用"——后果不只是少显示一个标记：点击时找可点击祖先
            # 也会因此跳过它们，于是**所有点击都上浮不过去**，模型点哪都没用。
            # 这个失败模式极难查：界面上一切正常，只是点什么都没反应。
            "enabled": node.get("enabled", "true") == "true",
            "bounds": bounds,
            "_ancestors": ancestors,
        })
        for child in node:
            if child.tag == "node":
                visit(child, (idx,) + ancestors)

    for node in root:
        if node.tag == "node":
            visit(node, ())
    return out


def _parse_bounds(s: str) -> tuple[int, int, int, int] | None:
    m = re.match(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]", s)
    if not m:
        return None
    return tuple(int(g) for g in m.groups())  # type: ignore[return-value]


def _is_interesting(e: dict[str, Any]) -> bool:
    """值得给模型看的元素：能交互，或者带文字。

    纯布局容器（FrameLayout / LinearLayout 这些）对模型没有信息量，
    渲染出来只是噪声，还会把真正有用的那几十个挤出窗口。
    """
    return bool(e["clickable"] or e["editable"] or e["scrollable"]
                or e["text"] or e["desc"])


def _has_area(e: dict[str, Any]) -> bool:
    """宽高都大于 0。

    无障碍树里存在 `[0,0][0,0]` 这种零尺寸占位节点——它们看不见也点不着，
    留在列表里只会白白占掉一个序号。
    """
    x1, y1, x2, y2 = e["bounds"]
    return x2 > x1 and y2 > y1


def actionable_elements(
    elements: list[dict[str, Any]], *, limit: int = 60
) -> list[dict[str, Any]]:
    """**模型能看到的元素列表，同时也就是 `click(index=N)` 里 N 的所指。**

    ⚠️ 这个函数是整个索引契约的唯一来源，**渲染和点击都必须用它**。

    踩过的坑：早先渲染时过滤掉了不可交互节点、序号重新编，而点击时却按
    未过滤的原始列表取坐标。只要前面被过滤掉一个节点，后面**所有序号全部
    错位**——模型说点第 7 个，实际点到了别的地方。表现是模型"点了没反应"，
    看起来像它卡住，其实是框架在骗它。
    """
    out = []
    for e in elements:
        if not (_is_interesting(e) and _has_area(e)):
            continue
        shown = dict(e)
        shown["_click_xy"] = _resolve_click(elements, e)
        out.append(shown)
        if len(out) >= limit:
            break
    return out


def _resolve_click(raw: list[dict[str, Any]], e: dict[str, Any]) -> tuple[int, int]:
    """算出"点这个元素"时该往哪点。

    规则：**元素自己可点击就点自己；否则往上找最近的可点击祖先。**

    为什么要往上找：安卓界面里很常见 `<可点击的列表项><不可点击的文字>` 这种
    结构，真正挂 `OnClickListener` 的是外层容器，文字只是容器里的一个标签。

    只找**最近**的那一个，不做"面积最小""不超过屏幕几成"之类的启发式——
    最近即最具体，而任何基于面积/比例的规则都会随界面布局变化，
    那就不叫泛用了。

    `raw` 是未过滤的全量列表，祖先存的是它的下标。
    """
    if e["clickable"] or not e.get("_ancestors"):
        return _center(e["bounds"])
    for a in e["_ancestors"]:                      # 由近及远
        anc = raw[a]
        if anc["clickable"] and anc["enabled"]:
            return _center(anc["bounds"])
    return _center(e["bounds"])


def _center(bounds: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bounds
    return (x1 + x2) // 2, (y1 + y2) // 2


def render_elements(elements: list[dict[str, Any]], *, limit: int = 60) -> str:
    """把**已经过滤好的**元素列表渲染成给模型看的纯文本。

    入参必须是 `actionable_elements()` 的返回值——它编的序号就是模型说的
    序号，本函数**不再做任何过滤**（过滤一次就够了，过滤两次就会错位）。

    ⚠️ 这段文本会原样进提示词，所以它必须包在 `<screen source="device">` 里
    （由 `prompts.py` 负责），而且**不要在这里拼任何指令性文字**——
    屏幕内容是不可信输入，见 prompts.py 关于提示注入的说明。
    """
    lines = []
    for i, e in enumerate(elements):
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
            f"[{i}] <{e['class']}> {label!r}{'' if not flags else ' ' + ' '.join(flags)}"
            f"{ident} bounds=({x1},{y1},{x2},{y2})"
        )
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
        description=(
            "按名字启动一个 app。**这是打开 app 最快的方式**，"
            "优先用它，不要去桌面上找图标——很多 app 的图标并不在桌面第一页，"
            "在桌面上滚动或搜索往往要试很多步。"
            "app_name 填任务里那个应用的名字就行，环境会自己查它对应哪个包。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    # ⚠️ 这里**故意不写任何包名对照表**。
                    #
                    # 早先的版本写了（"com.google.android.deskclock=时钟，…"），
                    # 后果是模型完全不看屏幕：拿任务里的"时钟"两个字去表里一查，
                    # 一步到位，整个 benchmark 退化成查表题——而我们还以为
                    # 自己在测 GUI 能力。
                    #
                    # 对照表本身没错（"设置"是哪个包，那是设备的事实，推不出来），
                    # 错在**放进了模型能看见的地方**。现在它属于环境配置，
                    # 由 scripts/scan_apps.py 扫设备生成。
                    "description": "应用的显示名，如「时钟」「Clock」「设置」。也可以直接给包名。",
                },
            },
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
