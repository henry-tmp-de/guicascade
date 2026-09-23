"""翻译层：把「模型说的话」变成「可以执行的动作」。

这是框架里唯一一处**一定会有脏活**的地方，值得说清楚为什么。

模型只会吐字，它不会真的去点屏幕。中间必须有东西把那段文字翻译成
`Action(name, args)`。用原生 tool calling 时，翻译由 API 保证格式合法，
几乎零风险；但走文本时，模型写出来的东西五花八门：

    Action: click(index=3)          <- 正常，能解析
    Action: click(3)                <- 省了参数名，得靠位置参数映射救回来
    click(index=3)                  <- 没写 "Action:" 前缀
    ```json {"name": "click", ...}  <- 换成了 JSON
    我点击了 index=3 的按钮          <- 压根没给结构化动作，救不了

**每一条救不回来，整个任务就废在这里。** 所以这个文件不追求优雅，
追求"能救的都救回来，救不回来的给出人话解释"。

`DecodeError` 的消息是**写给模型看的**——上层会把它回灌进对话让模型
自我纠正（这个做法来自 mini-swe-agent 的 FormatError）。所以错误信息
要说清"哪里不对、应该写成什么样"，而不是只报个错因。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .tools import Toolkit
from .types import GUI, TOOL, Action, ModelResponse

__all__ = ["ActionSpec", "ActionSpace", "DecodeError", "decode"]


class DecodeError(ValueError):
    """模型输出无法翻译成动作。消息文本面向模型，可读、可执行。"""


# --------------------------------------------------------------------------
# 动作的定义
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionSpec:
    """一个动作的定义。

    描述文字（`description`）不是可选的装饰。消融实验显示，**把工具的描述
    全删掉、只留参数定义，调用错误率会涨四成以上**。写描述时要说清楚
    "什么时候该用"和"什么时候别用"，后者尤其容易被漏掉。
    """

    name: str
    description: str
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    positional: tuple[str, ...] = ()
    """位置参数的映射顺序。

    用来救 `click(3)` 这种省略参数名的写法——模型经常这么写，直接判失败
    太浪费。比如 click 的 positional=("index",)，那么 `click(3)` 会被
    修复成 `click(index=3)`。
    """

    @property
    def properties(self) -> Mapping[str, Any]:
        return self.parameters.get("properties", {})

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(self.parameters.get("required", ()))

    def to_tool_schema(self) -> dict[str, Any]:
        """产出 OpenAI 风格的 function schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }

    def describe(self) -> str:
        """产出人话版描述，给走文本模式的模型看。"""
        params = []
        for pname, pspec in self.properties.items():
            flag = "必填" if pname in self.required else "可选"
            desc = pspec.get("description", "")
            params.append(f"      {pname}（{flag}）：{desc}")
        body = "\n".join(params) if params else "      （无参数）"
        return f"  {self.name}：{self.description}\n{body}"


@dataclass
class ActionSpace:
    """一个环境的动作空间。

    刻意做成**具体类**而不是 Protocol：各环境（安卓/浏览器/桌面）的动作
    只是"名字 + 参数"的差异，用同一套结构填不同内容即可。真需要特殊校验
    时子类覆写 `validate` 就行，不必为每种环境写一个类。
    """

    name: str
    specs: dict[str, ActionSpec] = field(default_factory=dict)
    """动作名 -> 定义。"""

    def add(self, spec: ActionSpec) -> "ActionSpace":
        self.specs[spec.name] = spec
        return self

    def get(self, name: str) -> ActionSpec | None:
        return self.specs.get(name)

    def describe(self) -> str:
        return "\n".join(s.describe() for s in self.specs.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [s.to_tool_schema() for s in self.specs.values()]

    def validate(self, action: Action) -> Action | None:
        """校验并尽量修复，修不了返回 None。

        修复优先于拒绝：模型漏个参数名、多写个空字符串，都不该让整个任务
        重来。真正救不了的（动作名不存在、必填项缺失）才返回 None。
        """
        spec = self.specs.get(action.name)
        if spec is None:
            return None

        args = {k: v for k, v in action.args.items() if v is not None}
        # 丢掉 schema 里没声明的额外字段：模型偶尔会自作主张加字段
        if spec.properties:
            args = {k: v for k, v in args.items() if k in spec.properties}

        missing = [r for r in spec.required if r not in args]
        if missing:
            return None

        return Action(action.name, args, action.kind)


# --------------------------------------------------------------------------
# 从模型输出里抠动作 —— 多策略兜底
# --------------------------------------------------------------------------

_ACTION_LINE = re.compile(r"^\s*Action\s*[:：]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_CALL_SYNTAX = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)
_KWARG = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)(?=,\s*[A-Za-z_][A-Za-z0-9_]*\s*=|$)", re.DOTALL)

# 键值对分行写法。视觉模型（Qwen-VL / UI-TARS 这些）很爱这么输出：
#
#     action: click
#     x: 76
#     y: 1873
#
# 它和 `Action: click(index=3)` 表达的是同一件事，只是一个横着写、一个竖着写。
# 光靠正则救不回来，得单独认。
_KV_ACTION = re.compile(r"^\s*action\s*[:：]\s*([A-Za-z_][A-Za-z0-9_]*)\s*$",
                        re.IGNORECASE | re.MULTILINE)
_KV_PAIR = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:：]\s*(.+?)\s*$")


def decode(
    response: ModelResponse,
    space: ActionSpace,
    toolkit: Toolkit | None = None,
) -> tuple[str, Action]:
    """把模型回应翻译成 `(理由, 动作)`。两条通道在这里汇合。

    优先用 tool_calls（格式由 API 保证），没有才回去抠文本。

    Returns:
        (reason, action)。reason 是模型的自由文本——**级联的监控器吃的
        就是它**，所以即使走了 tool calling，这段文字也不能丢。

    Raises:
        DecodeError: 消息面向模型，上层可直接回灌让它自我纠正。
    """
    reason = (response.text or "").strip()

    if response.tool_calls:
        return reason, _from_tool_call(response.tool_calls[0], space, toolkit)

    if not reason:
        raise DecodeError(
            "你的回复里既没有文字也没有动作，我无法判断你想做什么。"
            "请重新输出：先用一句话说明理由，再给出一个动作。"
        )

    return reason, _from_text(reason, space, toolkit)


def _from_tool_call(call, space: ActionSpace, toolkit: Toolkit | None) -> Action:
    kind = _kind_of(call.name, space, toolkit)

    if kind == GUI:
        if space.get(call.name) is None:
            raise DecodeError(
                f"没有名为 {call.name!r} 的界面动作。可用的界面动作："
                f"{', '.join(sorted(space.specs)) or '（无）'}"
            )
        action = space.validate(Action(call.name, dict(call.arguments), GUI))
        if action is None:
            spec = space.get(call.name)
            raise DecodeError(
                f"动作 {call.name!r} 的参数不合法。它需要："
                f"{', '.join(spec.required) or '（无必填项）'}；"
                f"你给的是：{dict(call.arguments)}"
            )
        return action

    return Action(call.name, dict(call.arguments), TOOL)


def _from_text(text: str, space: ActionSpace, toolkit: Toolkit | None) -> Action:
    # 策略零：键值对分行写法。必须**先于** _extract_payload 试——
    # 因为 `action: click` 这一行本身会被 _ACTION_LINE 命中、截成 "click"，
    # 后面几行的 x/y 就丢了。
    kv = _try_kv(text, space, toolkit)
    if kv is not None:
        return kv

    payload = _extract_payload(text)

    # 策略一：整个 payload 就是个 JSON 对象
    data = _try_json(payload)
    if isinstance(data, dict):
        name = data.get("name") or data.get("action") or data.get("action_type")
        args = data.get("args") or data.get("arguments") or data.get("parameters")
        if name and args is None:
            # 平铺写法：{"action_type": "click", "index": 3}
            args = {k: v for k, v in data.items() if k not in ("name", "action", "action_type")}
        if name:
            return _finish(str(name), dict(args or {}), space, toolkit)

    # 策略二：call 语法 name(k=v, ...) 或 name(v, ...)
    m = _CALL_SYNTAX.match(payload)
    if m:
        return _finish(m.group(1), _parse_args(m.group(2), m.group(1), space), space, toolkit)

    raise DecodeError(
        "没能从你的回复里看出要执行什么动作。请严格按下面的格式重新输出：\n"
        "  Action: 动作名(参数名=值, 参数名=值)\n"
        "例如：Action: click(index=3)\n"
        f"可用的界面动作：{', '.join(sorted(space.specs)) or '（无）'}"
    )


def _try_kv(text: str, space: ActionSpace, toolkit: Toolkit | None) -> Action | None:
    """认 `action: click` 换行再写 `x: 76` 的写法。

    返回 None 表示"这段文本不是这个格式"，交给后面的策略。
    注意正则要求 action 名**单独占一行**（没有括号），所以
    `Action: click(index=3)` 不会被误判。
    """
    m = _KV_ACTION.search(text)
    if not m:
        return None

    name = m.group(1)
    args: dict[str, Any] = {}
    for line in text[m.end():].splitlines():
        if not line.strip():
            continue
        pair = _KV_PAIR.match(line)
        if pair:
            args[pair.group(1)] = _coerce(pair.group(2))
            if len(args) >= 8:
                break
        elif args:
            # 已经收到参数了，再遇到非键值行就认为这一段结束了
            break

    # 名字既不认识也不是工具 -> 这段文本不是在说动作，别硬解
    known = space.get(name) is not None or (toolkit is not None and name in toolkit.tools)
    if not known:
        return None

    return _finish(name, args, space, toolkit)


def _extract_payload(text: str) -> str:
    """从一大段模型输出里，定位到"动作那一段"。

    按可靠性从高到低试，第一个命中就用：
      1. `Action: xxx` 行   —— 我们要求的格式
      2. ```json 围栏       —— 模型爱用
      3. 整段里最后一个 {...} —— 兜底，通常是对的那个
      4. 整段文本           —— 可能是裸的 call 语法
    """
    m = _ACTION_LINE.search(text)
    if m:
        return m.group(1).strip()

    m = _JSON_FENCE.search(text)
    if m:
        return m.group(1).strip()

    start, end = text.rfind("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]

    # 裸 call 语法：取最后一行非空内容
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _parse_args(raw: str, name: str, space: ActionSpace) -> dict[str, Any]:
    """解析括号里的参数，兼容 k=v 和位置参数两种写法。"""
    raw = raw.strip()
    if not raw:
        return {}

    # 先试 JSON：click({"index": 3})
    data = _try_json(raw)
    if isinstance(data, dict):
        return data

    # 再试 k=v 写法
    args: dict[str, Any] = {}
    for m in _KWARG.finditer(raw):
        args[m.group(1)] = _coerce(m.group(2).strip())

    if args:
        return args

    # 最后试位置参数，靠 ActionSpec.positional 映射回参数名
    parts = [p.strip() for p in _split_top_level(raw) if p.strip()]
    spec = space.get(name)
    if spec and spec.positional and len(parts) <= len(spec.positional):
        return {spec.positional[i]: _coerce(p) for i, p in enumerate(parts)}

    return {}


def _split_top_level(raw: str) -> list[str]:
    """按逗号切分，但忽略引号和括号内部的逗号。"""
    parts, depth, quote, buf = [], 0, "", []
    for ch in raw:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def _coerce(value: str) -> Any:
    """把字面量字符串转回 Python 类型。"""
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("none", "null"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    data = _try_json(v)
    return data if data is not None else v


def _try_json(raw: str) -> Any:
    """尽量宽容地解析 JSON，失败返回 None。"""
    raw = raw.strip()
    if not raw:
        return None
    for candidate in (raw, raw.replace("'", '"')):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _finish(name: str, args: dict[str, Any], space: ActionSpace, toolkit: Toolkit | None) -> Action:
    """收尾：定 kind、校验、修不了就报错。"""
    kind = _kind_of(name, space, toolkit)

    if kind == TOOL:
        return Action(name, args, TOOL)

    if space.get(name) is None:
        raise DecodeError(
            f"没有名为 {name!r} 的动作。可用的界面动作："
            f"{', '.join(sorted(space.specs)) or '（无）'}"
        )
    action = space.validate(Action(name, args, GUI))
    if action is None:
        spec = space.get(name)
        raise DecodeError(
            f"动作 {name!r} 缺少必填参数。它需要："
            f"{', '.join(spec.required) or '（无必填项）'}；你给的是：{args}"
        )
    return action


def _kind_of(name: str, space: ActionSpace, toolkit: Toolkit | None) -> str:
    """判断这个动作该由谁执行。

    判定顺序很重要：**先看是不是界面动作**。因为环境动作优先——万一某个
    工具和环境动作重名，界面操作更可能是模型的本意，而且它是无害的那一侧。
    """
    if space.get(name) is not None:
        return GUI
    if toolkit is not None and name in toolkit.tools:
        return TOOL
    return GUI  # 都不认识：当成界面动作，让上层报出"没有这个动作"的错
