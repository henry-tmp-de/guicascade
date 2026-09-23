"""解析层测试。

这是全项目最容易坏的地方——模型换个说法，整个任务就废在这里。
所以这里测的不是"正常输入能跑"，而是**每一种畸形输入能不能被救回来**。
"""

from __future__ import annotations

import pytest

from guicascade.actions import ActionSpace, ActionSpec, DecodeError, decode
from guicascade.tools import FinishTool, NoteTool, Toolkit
from guicascade.types import GUI, TOOL, Action, ModelResponse, ToolCall


@pytest.fixture
def space() -> ActionSpace:
    return ActionSpace(name="test").add(
        ActionSpec(
            name="click",
            description="点击",
            parameters={
                "type": "object",
                "properties": {"index": {"type": "integer"}},
                "required": ["index"],
            },
            positional=("index",),
        )
    ).add(ActionSpec(name="finish", description="结束"))


@pytest.fixture
def toolkit() -> Toolkit:
    return Toolkit().add(FinishTool()).add(NoteTool())


# --------------------------------------------------------------------------
# 文本通道的兜底策略 —— 每一种模型写歪的方式都要能救回来
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Action: click(index=3)",                      # 标准写法
        "先看一眼屏幕。\nAction: click(index=3)",       # 前面有解释
        "Action: click(index=3)\n还有别的吗",           # 后面有废话
        "```json\n{\"name\": \"click\", \"args\": {\"index\": 3}}\n```",  # JSON 围栏
        "{\"action_type\": \"click\", \"index\": 3}",   # 平铺 JSON
        "click(index=3)",                             # 没有 Action: 前缀
        "Action: click(3)",                           # 省略参数名 -> 靠 positional 修
        "Action: click(index='3')",                   # 数字被加了引号
    ],
)
def test_text_fallbacks_all_parse(text: str, space: ActionSpace) -> None:
    """八种写法都要能解析成同一个动作。"""
    _, action = decode(ModelResponse(text=text), space)
    assert action.name == "click"
    assert action.kind == GUI


def test_positional_argument_is_repaired(space: ActionSpace) -> None:
    """`click(3)` 这种省略参数名的写法，靠 ActionSpec.positional 救回来。"""
    _, action = decode(ModelResponse(text="Action: click(3)"), space)
    assert action.args == {"index": 3}
    assert isinstance(action.args["index"], int)


def test_reason_is_preserved(space: ActionSpace) -> None:
    """理由必须原样带出来——级联的监控器吃的就是它。"""
    reason, _ = decode(ModelResponse(text="我看到按钮在右下角。\nAction: click(index=2)"), space)
    assert "右下角" in reason


# --------------------------------------------------------------------------
# 原生 tool calling 通道
# --------------------------------------------------------------------------


def test_tool_call_channel(space: ActionSpace) -> None:
    resp = ModelResponse(
        text="我需要点这个按钮。",
        tool_calls=(ToolCall(name="click", arguments={"index": 5}),),
    )
    reason, action = decode(resp, space)
    assert reason == "我需要点这个按钮。"   # 文字通道同时也在，喂监控器用
    assert action.name == "click"
    assert action.args["index"] == 5


def test_tool_call_takes_priority_over_text(space: ActionSpace) -> None:
    """两个通道都有时，以结构化的为准——它不会解析错。"""
    resp = ModelResponse(
        text="Action: click(index=1)",
        tool_calls=(ToolCall(name="click", arguments={"index": 9}),),
    )
    _, action = decode(resp, space)
    assert action.args["index"] == 9


# --------------------------------------------------------------------------
# 工具 vs 界面动作：靠 kind 区分
# --------------------------------------------------------------------------


def test_unknown_name_goes_to_tools(space: ActionSpace, toolkit: Toolkit) -> None:
    _, action = decode(ModelResponse(text="Action: note(content='记一笔')"), space, toolkit)
    assert action.kind == TOOL
    assert action.name == "note"


def test_environment_action_wins_on_name_clash(space: ActionSpace, toolkit: Toolkit) -> None:
    """重名时优先当界面动作——它更可能是模型的本意，而且是无害的那一侧。"""
    toolkit.add(FinishTool(name="click"))  # 故意造一个重名
    _, action = decode(ModelResponse(text="Action: click(index=2)"), space, toolkit)
    assert action.kind == GUI


# --------------------------------------------------------------------------
# 救不回来的时候，错误信息要是给人/模型看的
# --------------------------------------------------------------------------


def test_unparseable_raises_with_actionable_message(space: ActionSpace) -> None:
    with pytest.raises(DecodeError) as exc:
        decode(ModelResponse(text="我点击了那个按钮。"), space)
    msg = str(exc.value)
    assert "Action:" in msg          # 告诉它该写成什么样
    assert "click" in msg            # 列出可用动作


def test_unknown_action_lists_available(space: ActionSpace) -> None:
    with pytest.raises(DecodeError) as exc:
        decode(ModelResponse(text="Action: teleport(index=1)"), space)
    assert "click" in str(exc.value)


def test_missing_required_argument_is_reported(space: ActionSpace) -> None:
    with pytest.raises(DecodeError) as exc:
        decode(ModelResponse(text="Action: click()"), space)
    assert "index" in str(exc.value)


def test_empty_response_is_reported(space: ActionSpace) -> None:
    with pytest.raises(DecodeError):
        decode(ModelResponse(text=""), space)
