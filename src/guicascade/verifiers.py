"""里程碑验证器：用强模型核对「刚才那一步是不是真的完成了阶段性成果」。

这是级联机制里**唯一一次昂贵调用**，也是它和「卡住检测」的关键区别：

- 卡住监控器发现异常 → **直接升级**（局部循环，没什么好确认的）
- 里程碑监控器发现可能完成 → **先验证再决定**（因为「以为完成了」和
  「真的完成了」差别很大，误判会导致后续步骤全建在错误前提上）

验证刻意引入截图：监控器本身是纯文本的（廉价、每步都能跑），但到了
这一步，问题已经从「轨迹像不像有进展」变成「界面上到底变了没有」，
必须看真实像素。这是本框架里唯一一处把图像接回来的地方。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from .models.base import Message, Model
from .types import Step

__all__ = ["MilestonePacket", "MilestoneVerdict", "MilestoneVerifier", "LlmMilestoneVerifier"]


@dataclass(frozen=True, slots=True)
class MilestonePacket:
    """交给验证器的完整证据包。

    「前后截图 + 期间的轨迹」这个组合是刻意的：单看截图无法判断
    「这一步是在朝目标前进，还是在乱试」；单看文本又看不出
    「嘴上说点发送、手上点到隔壁删除」这类空间错位。两者对照才有意义。
    """

    task: str
    """任务描述——判断「有没有进展」离不开目标。"""

    reasoning: str
    """自上一个已确认里程碑以来的轨迹文本。"""

    before_image: bytes | None = None
    """上一个已确认里程碑时刻的截图。"""

    after_image: bytes | None = None
    """当前时刻的截图。"""

    steps_since: int = 0


@dataclass(frozen=True, slots=True)
class MilestoneVerdict:
    success: bool
    inferred_milestone: str = ""
    reasoning: str = ""
    raw: str = ""
    latency_s: float = 0.0


@runtime_checkable
class MilestoneVerifier(Protocol):
    """判定一个候选里程碑是否真的达成。"""

    def verify(self, packet: MilestonePacket) -> MilestoneVerdict: ...


_PROMPT = """\
You are an expert evaluator of GUI agent progress. Determine whether the attempted \
milestone was actually achieved.

You are given the task description, the actions taken since the previous milestone, \
a BEFORE screenshot from the previous milestone, and an AFTER screenshot from the \
current step.

Instructions:
- Infer what milestone the agent was attempting from the task description and the \
recent actions.
- Compare BEFORE and AFTER to decide whether that milestone was actually reached.
- Use the action history as supporting evidence, but judge primarily on whether the \
AFTER state shows meaningful progress relative to BEFORE.
- Mark "success" true only if the milestone is clearly achieved. Mark it false if the \
screenshots do not support completion, if the state is inconsistent with the intended \
progress, or if the evidence suggests failure.
- Do NOT invent UI details not supported by the screenshots or the action history.
- Give a concise reasoning for your decision.

Return JSON only:
{{"inferred_milestone": "...", "success": true/false, "reasoning": "..."}}
"""


class LlmMilestoneVerifier:
    """用一个大模型做里程碑验证。

    刻意复用 `Model` 接口而不是另起一套：验证器本质就是「带图的模型调用」，
    没有理由为它发明新的抽象。
    """

    def __init__(self, model: Model, max_reasoning_chars: int = 4000) -> None:
        self.model = model
        self.max_reasoning_chars = max_reasoning_chars

    def verify(self, packet: MilestonePacket) -> MilestoneVerdict:
        # 导入放在函数内：这一步避免让「不跑验证器」的用法也背上图像编码的依赖
        import json
        import time

        from .models.openai_compat import image_part, text_part

        content: list[dict] = [text_part(self._build_text(packet))]
        if packet.before_image:
            content.append(text_part("BEFORE (previous milestone):"))
            content.append(image_part(packet.before_image))
        if packet.after_image:
            content.append(text_part("AFTER (current step):"))
            content.append(image_part(packet.after_image))

        messages: Sequence[Message] = [{"role": "user", "content": content}]

        t0 = time.perf_counter()
        raw = self.model.generate(messages)
        latency = time.perf_counter() - t0

        return _parse_verdict(raw, latency)

    def _build_text(self, packet: MilestonePacket) -> str:
        reasoning = packet.reasoning
        if len(reasoning) > self.max_reasoning_chars:
            reasoning = "...(earlier steps omitted)...\n" + reasoning[-self.max_reasoning_chars :]
        return (
            f"{_PROMPT}\n"
            f"## Task Description\n{packet.task}\n\n"
            f"## Actions Since Previous Milestone\n{reasoning}\n"
        )


def _parse_verdict(raw: str, latency: float) -> MilestoneVerdict:
    """从模型输出里抠出 JSON 判决。

    容错优先：验证器解析失败时**默认判失败（升级）**是刻意的选择。
    宁可多花一次大模型调用，也不要因为解析问题把「没完成」当成「完成了」
    ——后者会让整条轨迹建在错误前提上，代价高得多。
    """
    import json
    import re

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return MilestoneVerdict(success=False, reasoning="unparseable verdict", raw=raw, latency_s=latency)

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return MilestoneVerdict(success=False, reasoning="invalid JSON verdict", raw=raw, latency_s=latency)

    return MilestoneVerdict(
        success=bool(data.get("success", False)),
        inferred_milestone=str(data.get("inferred_milestone", "")),
        reasoning=str(data.get("reasoning", "")),
        raw=raw,
        latency_s=latency,
    )
