"""提示词监控器：不训练，直接让一个语言模型判断。

## 为什么需要它

官方那个 149M 的卡住监控器实测不可用（AUC 0.61，见 README）。
这个文件是它的替代方案：**用一份提示词 + 一个通用模型来做同样的判断**。

好处是三个"不用"：

    · 不用训练数据（原方法的训练集从未公开）
    · 不用 GPU 常驻（打一次分才调一次模型，可以复用已有的推理服务）
    · 换环境不用重训（跨环境迁移是原方法没解决的问题，提示词版天然免疫）

代价也很明确，必须说清楚：

    ✗ 比 149M 分类器慢得多，而且是每一步都调
    ✗ 成本随步数线性增长

**所以它不是为了取代分类器，而是为了证明"监督信号"和"机制"哪个才是瓶颈。**
如果把分类器换成提示词版，级联效果没有明显下滑，那说明原方法的增益主要
来自级联机制本身，而不是那个分类器训得多好——这是本项目要回答的问题之一。

## 提示词怎么来的

改写自官方用于**标注训练数据**的提示词（附录 D.2）。那份提示词原本是给
GPT-5.2 标注整条轨迹用的，这里改成"只看最近几步、给一个 0~1 的分"。
这样两边判的是同一个概念，对比才有意义。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..models.base import Message, Model
from ..types import Step
from .base import Monitor

__all__ = ["PromptMonitor", "STUCK_PROMPT", "MILESTONE_PROMPT"]

_HISTORY = 6

STUCK_PROMPT = """\
You are an expert evaluator of computer-use agents. Decide whether the agent is \
currently STUCK.

An agent is stuck if any of these holds:
1. It repeats the same action multiple times without progress.
2. It is in an error loop.
3. It has failed to make meaningful progress for several consecutive steps.

Judge only from the action trace below. Do NOT invent details that are not in it.

Return JSON only:
{"stuck": true/false, "confidence": 0.0-1.0, "reason": "one short sentence"}

`confidence` is how sure you are of your verdict, NOT the probability of being stuck.
If the agent is clearly making progress, answer false with high confidence.
"""

MILESTONE_PROMPT = """\
You are an expert evaluator of computer-use agents. Decide whether the most recent \
step completes a meaningful, verifiable milestone toward the task.

Rules:
- A milestone must be meaningful and verifiable from the step text alone.
- Prefer higher-level progress markers over routine clicks.
- If the trajectory is stuck (repeating itself with no progress), do not call it a milestone.
- Do NOT invent UI details you cannot support from the trace.

Return JSON only:
{"milestone": true/false, "confidence": 0.0-1.0, "reason": "one short sentence"}

`confidence` is how sure you are of your verdict, NOT the probability of a milestone.
"""


@dataclass
class PromptMonitor:
    """用提示词让通用模型打分，替代训练出来的分类器。"""

    model: Model
    name: str = "prompt"
    mode: str = "stuck"
    threshold: float = 0.5
    history: int = _HISTORY
    max_reason_chars: int = 600
    """每步理由截断长度。整条窗口一股脑塞进去会撑爆上下文，而判卡住不需要全文。"""

    _last: dict[str, Any] = field(default_factory=dict, repr=False)
    """上一次的完整判决，便于事后做误报/漏报的定性分析。"""

    def __post_init__(self) -> None:
        if self.mode not in ("stuck", "milestone"):
            raise ValueError(f"mode 必须是 stuck 或 milestone，收到 {self.mode!r}")

    def score(self, task: str | None, steps: Sequence[Step]) -> float:
        if not steps:
            return 0.0

        verdict = self._ask(task, steps)
        self._last = verdict

        # 置信度是"多确定"，不是"多可能是正类"。把它直接当概率用是错的。
        # 这里的做法是：判决为正类 -> 用置信度当分数；判决为负类 -> 1 减置信度
        # 作为"其实没那么负"的余地。这样阈值语义单调，也能表达边界情况。
        conf = float(verdict.get("confidence", 0.5) or 0.5)
        is_positive = bool(verdict.get(self._key, False))
        return conf if is_positive else 1.0 - conf if conf > 0.5 else 0.0

    @property
    def _key(self) -> str:
        return "stuck" if self.mode == "stuck" else "milestone"

    def _ask(self, task: str | None, steps: Sequence[Step]) -> dict[str, Any]:
        prompt = STUCK_PROMPT if self.mode == "stuck" else MILESTONE_PROMPT
        messages: list[Message] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": self._render(task, steps)},
        ]
        try:
            raw = self.model.generate(messages).text
        except Exception as e:  # noqa: BLE001 - 打分失败不该打断 episode
            return {"_error": f"{type(e).__name__}: {e}", "confidence": 0.0}

        data = _parse_json(raw)
        data["_raw"] = raw
        return data

    def _render(self, task: str | None, steps: Sequence[Step]) -> str:
        recent = list(steps)[-self.history :]
        lines = []
        if task:
            lines.append(f"## Task\n{task}\n")
        lines.append("## Recent steps")
        for i, s in enumerate(recent, start=1):
            reason = s.decision.reason
            if len(reason) > self.max_reason_chars:
                reason = reason[: self.max_reason_chars] + "…"
            lines.append(f"Step {i}:")
            lines.append(f"  Action: {s.decision.action}")
            lines.append(f"  Reasoning: {reason}")
        lines.append("\nGive your verdict as JSON.")
        return "\n".join(lines)


def _parse_json(raw: str) -> dict[str, Any]:
    """尽量宽容地抠出 JSON。抠不出就返回空字典，由调用方兜底。"""
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
