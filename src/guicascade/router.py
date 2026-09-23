"""路由：决定「下一步该用哪个模型」。

这一层和「模型怎么生成动作」完全分开——它只做一件事：看已有的轨迹，
判断这一步值不值得上大模型。

这么切的好处很实际：原论文的消融基线是「固定间隔验证」，换成那个只需要
换一个 Router，策略代码一个字不用动。要试别的路由方式（启发式规则、
更贵的监控器）也是这样。

## 升级的两种触发，性质完全不同

- **卡住监控器触发 → 直接升级。** 局部循环没什么好确认的，轨迹明显在原地
  打转，换个大模型是唯一出路。
- **里程碑监控器触发 → 先验证，再决定。** "以为完成了"和"真的完成了"
  差别很大，误判会让后续所有步骤建在错误前提上。所以这里要花一次
  昂贵调用去看前后截图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

from .monitors.base import Monitor
from .types import Step
from .verifiers import MilestonePacket, MilestoneVerifier

__all__ = ["RouterDecision", "Router", "CascadeRouter"]


@dataclass(frozen=True, slots=True)
class RouterDecision:
    """这一步的路由结论。"""

    use_large: bool = False
    trigger: str = ""
    """触发原因：`""` / `"stuck"` / `"milestone_failed"` / `"hold"`。

    留这个字段是为了事后能回答"到底是哪条路把大模型叫起来的"——
    没有它，升级率这个数字就没法解释。
    """

    signals: Mapping[str, float] = field(default_factory=dict)
    """各路监控器在这一步打的分，会原样写进轨迹。"""

    verified_milestone: bool = False
    """里程碑验证是否通过（通过则更新"上一个里程碑"的基准点）。"""


@runtime_checkable
class Router(Protocol):
    """决定每一步用哪个模型。"""

    def reset(self) -> None: ...

    def route(self, task: str, history: Sequence[Step]) -> RouterDecision: ...


class CascadeRouter:
    """默认实现：两个监控器 + 阈值比较 + 可选的里程碑验证。

    ⚠️ 关于 `theta_milestone`：官方发布的里程碑监控器判别力很弱
    （训练日志显示最后一轮 F1 掉到 0，最佳 checkpoint 也才 0.47），
    **直接用 0.5 几乎不会触发**。上线前必须重新校准，见
    `scripts/calibrate_thresholds.py`。
    """

    def __init__(
        self,
        stuck: Monitor | None = None,
        milestone: Monitor | None = None,
        verifier: MilestoneVerifier | None = None,
        *,
        theta_stuck: float = 0.5,
        theta_milestone: float = 0.5,
        min_steps: int = 2,
        hold_steps: int = 0,
    ) -> None:
        """
        Args:
            min_steps: 前几步不判——轨迹太短时监控器没有足够上下文，
                打分基本是噪声。官方实现里也是 2。
            hold_steps: 升级后至少再走几步大模型。

                0 表示完全按论文的逐步判定来（每步重新看信号，掉下去就
                回落小模型）。但实践中这会让大小模型来回横跳，所以留了这个
                旋钮。**做成参数而不是写死，是因为它是个实验变量**——
                想知道"保持几步最划算"，扫这个参数就行。
        """
        self.stuck = stuck
        self.milestone = milestone
        self.verifier = verifier
        self.theta_stuck = theta_stuck
        self.theta_milestone = theta_milestone
        self.min_steps = min_steps
        self.hold_steps = hold_steps
        self.reset()

    def reset(self) -> None:
        """每个任务开始时归零。"""
        self._last_milestone_idx = 0
        self._last_milestone_image: bytes | None = None
        self._hold_left = 0

    def route(self, task: str, history: Sequence[Step]) -> RouterDecision:
        signals = self._score(task, history)

        # 还没走够步数：不判，也别消耗 hold
        if len(history) < self.min_steps:
            return RouterDecision(signals=signals)

        # 上一轮升级的余温：还在保持期内就继续用大模型
        if self._hold_left > 0:
            self._hold_left -= 1
            return RouterDecision(use_large=True, trigger="hold", signals=signals)

        trigger = ""
        verified = False

        if self.stuck is not None and signals.get("stuck", 0.0) >= self.theta_stuck:
            # 卡住 -> 不用确认，直接升级
            trigger = "stuck"

        elif self.milestone is not None and signals.get("milestone", 0.0) >= self.theta_milestone:
            # 疑似完成一个里程碑 -> 花一次昂贵调用去核实
            if self.verifier is not None:
                verdict = self.verifier.verify(self._build_packet(task, history))
                if verdict.success:
                    verified = True
                    self._last_milestone_idx = len(history) - 1
                    if history:
                        self._last_milestone_image = history[-1].observation.image
                else:
                    # 以为完成了、其实没有 -> 这是语义漂移，升级
                    trigger = "milestone_failed"
            else:
                # 没配验证器就把触发本身当成里程碑：至少推进基准点，
                # 免得同一段轨迹反复触发同一次判定
                verified = True
                self._last_milestone_idx = len(history) - 1

        if trigger:
            self._hold_left = self.hold_steps

        return RouterDecision(
            use_large=bool(trigger),
            trigger=trigger,
            signals=signals,
            verified_milestone=verified,
        )

    # ------------------------------------------------------------------

    def _score(self, task: str, history: Sequence[Step]) -> dict[str, float]:
        """跑监控器。任何一路没配就跳过它。"""
        signals: dict[str, float] = {}
        if not history:
            return signals

        if self.stuck is not None:
            # 卡住监控器拿 None 当 task：它只看局部行为的重复性，
            # 不需要知道目标是什么
            signals["stuck"] = self.stuck.score(None, history)

        if self.milestone is not None:
            signals["milestone"] = self.milestone.score(task, history)

        return signals

    def _build_packet(self, task: str, history: Sequence[Step]) -> MilestonePacket:
        """组装交给验证器的证据包：任务 + 期间轨迹 + 前后截图。"""
        since = list(history)[self._last_milestone_idx :]
        lines = []
        for step in since:
            lines.append(f"Step {step.index + 1}:")
            lines.append(f"Response: {step.decision.reason}")
            lines.append(f"Action: {step.decision.action}")
            lines.append("")

        return MilestonePacket(
            task=task,
            reasoning="\n".join(lines).rstrip(),
            before_image=self._last_milestone_image,
            after_image=history[-1].observation.image if history else None,
            steps_since=len(since),
        )
