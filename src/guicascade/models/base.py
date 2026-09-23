"""模型层：把「一段对话」变成「一段文本」。

这一层刻意只做一件事——`generate`。它不知道什么是 GUI 动作、不知道
什么是监控器，也不负责解析输出。

为什么要把「调用模型」和「解析输出」拆开：这两件事的变化频率不同。
换一个模型供应商不该动解析逻辑；改动作空间不该碰模型代码。拆开之后，
两边都可以独立替换、独立测试。
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from ..types import ModelResponse

__all__ = ["Message", "Model"]

Message = dict[str, Any]
"""一条对话消息，形状与 OpenAI Chat Completions 一致。

用这个众所周知的形状而不是自造结构，好处是所有 OpenAI 兼容的服务
（vLLM、SGLang、本地常驻服务、各家云 API）都能直接对接，不用写胶水。
多模态内容直接用 OpenAI 的 `content: [{"type": "image_url", ...}]` 形式。
"""


@runtime_checkable
class Model(Protocol):
    """一个可调用的语言/视觉模型。

    实现只需提供 `name` 和 `generate`。刻意用 Protocol 而不是抽象基类：
    实现方不需要 import 本框架的任何东西，也不需要继承——任何长得像的
    对象都能直接塞进来，包括测试用的假模型和第三方客户端。
    """

    name: str
    """模型标识，会写进轨迹、用于统计「这一步是谁做的」。"""

    def generate(self, messages: Sequence[Message], **kwargs: Any) -> ModelResponse:
        """给定对话，返回模型的回应。

        刻意返回 `ModelResponse` 而不是裸字符串，因为**回应有两个通道**：
        自由文本（喂监控器）和结构化工具调用（直接构造动作）。

        只支持文本的模型把内容放进 `text` 即可，`tool_calls` 留空；
        支持原生 tool calling 的模型两个都填。调用方不需要为两种模型写
        两套逻辑——`ActionSpace.decode()` 统一处理。

        参数保持最窄：只吃 messages。采样参数、停止符之类通过 `**kwargs`
        透传，由各实现自行决定支不支持。
        """
        ...
