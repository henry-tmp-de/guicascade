"""假模型：按剧本回复，用于测试。

**这个文件看着不起眼，但它是这个项目能被验证的关键。**

GUI Agent 的测试通常要 GPU、要模拟器、要 API key——三者缺一就没法验证。
有了假模型，整个框架（主循环、解析、路由、级联、落盘）都能在**纯 CPU、
零依赖**的条件下跑通。于是：

    pip install -e . && pytest     # 任何人都能在三十秒内验证这个仓库

对求职项目来说这一点尤其值钱——**招聘方不会为了看你的代码去配环境**。

顺带它还让"级联逻辑"能被独立验证：给定一组固定的模型输出，升级时机对不对、
交接文本有没有注入、统计口径准不准，全都可以断言，不受模型随机性干扰。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..types import ModelResponse, ToolCall
from .base import Message

__all__ = ["ScriptedModel", "CodeBlockModel"]


@dataclass
class ScriptedModel:
    """按剧本依次返回预设回应，用完后重复最后一个。

    Example:
        >>> m = ScriptedModel("small", ["Action: click(index=1)", "Action: finish()"])
        >>> m.generate([]).text
        'Action: click(index=1)'
        >>> m.generate([]).text
        'Action: finish()'
        >>> m.generate([]).text      # 用完了，重复最后一条
        'Action: finish()'
    """

    name: str
    script: list[str | ModelResponse] = field(default_factory=list)
    repeat_last: bool = True
    calls: list[Sequence[Message]] = field(default_factory=list, repr=False)
    """记录每次收到的消息，供测试断言"喂进去的提示词长什么样"。"""

    _cursor: int = field(default=0, repr=False)

    def generate(self, messages: Sequence[Message], **kwargs: Any) -> ModelResponse:
        self.calls.append(list(messages))

        if not self.script:
            return ModelResponse(text="")

        idx = min(self._cursor, len(self.script) - 1)
        item = self.script[idx]
        if self._cursor < len(self.script) - 1:
            self._cursor += 1
        elif not self.repeat_last:
            self._cursor += 1

        if isinstance(item, ModelResponse):
            return item
        return ModelResponse(text=item)

    def reset(self) -> None:
        """把剧本倒回开头，方便一个假模型跑多个 episode。"""
        self._cursor = 0
        self.calls.clear()


@dataclass
class CodeBlockModel:
    """固定返回一个 JSON 对象的假模型，用来测**原生 tool calling 通道**。

    和 `ScriptedModel`（走文本通道）配对使用，两条通道就都能被覆盖。
    """

    name: str = "toolcall"
    reason: str = "I will click the button."
    tool_name: str = "click"
    arguments: dict[str, Any] = field(default_factory=lambda: {"index": 1})
    calls: list[Sequence[Message]] = field(default_factory=list, repr=False)

    def generate(self, messages: Sequence[Message], **kwargs: Any) -> ModelResponse:
        self.calls.append(list(messages))
        return ModelResponse(
            text=self.reason,
            tool_calls=(ToolCall(name=self.tool_name, arguments=dict(self.arguments)),),
        )

    def reset(self) -> None:
        self.calls.clear()
