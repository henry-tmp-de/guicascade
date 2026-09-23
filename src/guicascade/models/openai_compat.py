"""OpenAI 兼容的模型客户端。

对接任何说 OpenAI Chat Completions 协议的服务：vLLM、SGLang、服务器上的
常驻推理服务、以及各家云 API。**一套代码打通本地和云端**，这是选这个协议
而不是自造协议的全部理由。

## 为什么同时收两个通道

响应里 `content` 和 `tool_calls` 是并存的。级联恰好两边都要：

    content    -> 喂给监控器（它读的是"理由"）——**任何情况下都不能丢**
    tool_calls -> 直接构造 Action，不用正则去猜

所以这里把两个都取回来，原样交给 `ModelResponse`。
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..types import ModelResponse, ToolCall
from .base import Message

__all__ = ["OpenAICompatModel", "text_part", "image_part"]


def text_part(text: str) -> dict[str, Any]:
    """多模态 content 里的一段文本。"""
    return {"type": "text", "text": text}


def image_part(image: bytes, *, detail: str = "auto") -> dict[str, Any]:
    """多模态 content 里的一张图。

    GUI 任务每步都要传一张截图，所以编码开销是实打实的——这里用 base64
    内联而不是先上传再引用，是为了**少一次往返**：截图本来就是本地拿到的，
    多一次上传会引入额外延迟，而延迟正是本项目要量的东西。
    """
    b64 = base64.b64encode(image).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": detail}}


@dataclass
class OpenAICompatModel:
    """一个 OpenAI 兼容的模型端点。

    Example:
        >>> model = OpenAICompatModel("qwen3-vl-2b", base_url="http://host:8000/v1")
        >>> resp = model.generate([{"role": "user", "content": "hi"}])
        >>> resp.text
    """

    name: str
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 300.0
    extra_body: dict[str, Any] = field(default_factory=dict)
    """透传给服务端的额外字段。vLLM 的 `chat_template_kwargs` 之类走这里。"""

    tools: list[dict[str, Any]] | None = None
    """要暴露给模型的工具 schema。None 表示这次不提供工具（纯文本模式）。"""

    def generate(self, messages: Sequence[Message], **kwargs: Any) -> ModelResponse:
        import httpx

        payload: dict[str, Any] = {
            "model": self.name,
            "messages": list(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            **self.extra_body,
        }
        tools = kwargs.get("tools", self.tools)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        url = self.base_url.rstrip("/") + "/chat/completions"
        resp = httpx.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return _parse(resp.json())


def _parse(data: dict[str, Any]) -> ModelResponse:
    """把服务端响应拆成两个通道。

    容错优先：服务端实现五花八门（有的把 tool_calls 放在别处、有的返回
    空 choices），这里**尽量不抛异常**——一次解析失败不该打断整个 episode。
    真拿不到东西时返回空响应，让上层报出可读的错误。
    """
    choices = data.get("choices") or []
    if not choices:
        return ModelResponse(text="", raw=data)

    message = choices[0].get("message") or {}

    text = message.get("content") or ""
    if isinstance(text, list):
        # 有些服务端把 content 也做成多模态数组，只取文本片段
        text = "\n".join(p.get("text", "") for p in text if isinstance(p, dict))

    calls: list[ToolCall] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            # 参数不是合法 JSON：保留原文，让 ActionSpace 去报可读的错
            args = {"_raw": raw_args}
        calls.append(ToolCall(name=fn.get("name", ""), arguments=args, call_id=tc.get("id", "")))

    return ModelResponse(text=text, tool_calls=tuple(calls), raw=data)
