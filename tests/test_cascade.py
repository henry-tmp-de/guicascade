"""级联测试 —— 这个项目最核心的逻辑。

要验证的设计承诺：

  1. 监控器不报警时，**一步都不用大模型**（这是级联的全部价值）
  2. 监控器报警后，下一步确实换人
  3. 交接说明**只在升级那一刻注入一次**，不是每步都塞
  4. 里程碑验证通过 vs 不通过，走向不同
  5. 三组对照（全大 / 全小 / 级联）**共用同一个主循环**，统计口径一致
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pytest

from guicascade.actions import ActionSpace, ActionSpec
from guicascade.agent import Agent
from guicascade.envs.scripted import ScriptedEnvironment
from guicascade.models.scripted import ScriptedModel
from guicascade.policies import CascadePolicy, SingleModelPolicy
from guicascade.router import CascadeRouter
from guicascade.types import Step
from guicascade.verifiers import MilestonePacket, MilestoneVerdict


class FakeMonitor:
    """按预设序列打分的假监控器。"""

    def __init__(self, name: str, scores: list[float], threshold: float = 0.5) -> None:
        self.name = name
        self._scores = scores
        self.threshold = threshold
        self.calls = 0

    def score(self, task, steps: Sequence[Step]) -> float:
        i = min(self.calls, len(self._scores) - 1) if self._scores else 0
        self.calls += 1
        return self._scores[i] if self._scores else 0.0


@dataclass
class FakeVerifier:
    success: bool = True
    calls: list[MilestonePacket] = field(default_factory=list)

    def verify(self, packet: MilestonePacket) -> MilestoneVerdict:
        self.calls.append(packet)
        return MilestoneVerdict(success=self.success, reasoning="fake")


@pytest.fixture
def space() -> ActionSpace:
    return ActionSpace(name="t").add(
        ActionSpec(name="click", description="点击",
                   parameters={"type": "object",
                               "properties": {"index": {"type": "integer"}},
                               "required": ["index"]},
                   positional=("index",))
    )


def make_cascade(space, stuck_scores, *, milestone=None, verifier=None,
                 hold_steps=0, handoff="【交接】换人"):
    small = SingleModelPolicy(
        ScriptedModel("small", ["Action: click(index=1)"]), space, label="small"
    )
    large = SingleModelPolicy(
        ScriptedModel("large", ["Action: click(index=2)"]), space, label="large"
    )
    router = CascadeRouter(
        stuck=FakeMonitor("stuck", stuck_scores),
        milestone=milestone,
        verifier=verifier,
        theta_stuck=0.5,
        hold_steps=hold_steps,
    )
    return CascadePolicy(small, large, router=router, handoff=handoff)


def run(space, policy, screens=None, max_steps=6):
    env = ScriptedEnvironment(screens=screens or ["s"] * 30)
    return Agent(env, policy, max_steps=max_steps).run("任务")


# --------------------------------------------------------------------------


def test_no_escalation_when_monitor_stays_quiet(space: ActionSpace) -> None:
    """监控器不报警 -> 一步大模型都不用。这是级联存在的全部意义。"""
    policy = make_cascade(space, [0.0])
    traj = run(space, policy, max_steps=5)

    assert traj.n_escalated == 0
    assert traj.escalation_rate == 0.0
    assert all(s.model == "small" for s in traj.steps)


def test_escalates_after_monitor_fires(space: ActionSpace) -> None:
    """第 3 次打分开始报警 -> 之后的步走大模型。

    min_steps=2 意味着前两步不判，所以打分从第 3 步起才有效。
    """
    policy = make_cascade(space, [0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9])
    traj = run(space, policy, max_steps=6)

    assert traj.n_escalated >= 1
    assert any(s.model == "large" for s in traj.steps)
    assert any(s.decision.signals.get("stuck", 0) > 0.5 for s in traj.steps)


def test_hold_keeps_large_model(space: ActionSpace) -> None:
    """hold_steps 让升级后至少再走几步大模型，避免大小模型来回横跳。

    构造上要让"只报一次警"——如果分数一直高，两种配置都会每步升级，
    就测不出 hold 的作用了。
    """
    #           第3步  第4步  第5步  第6步  第7步
    scores = [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0]

    t_hot = run(space, make_cascade(space, scores, hold_steps=0), max_steps=6)
    t_held = run(space, make_cascade(space, scores, hold_steps=3), max_steps=6)

    assert t_held.n_escalated > t_hot.n_escalated


def test_handoff_is_injected_only_on_escalation(space: ActionSpace) -> None:
    """交接说明只在"刚从小切到大"那一刻出现一次。

    每步都注入等于往上下文里灌废话，还会稀释真正重要的信息。
    """
    policy = make_cascade(space, [0.0, 0.0, 0.0, 0.9, 0.9, 0.9], handoff="【交接】注意弹窗")
    run(space, policy, max_steps=5)

    large_prompts = [c[-1]["content"] for c in policy.large.model.calls]
    assert any("【交接】注意弹窗" in p for p in large_prompts)
    assert sum("【交接】注意弹窗" in p for p in large_prompts) == 1


def test_milestone_verification_pass_updates_baseline(space: ActionSpace) -> None:
    """验证通过 -> 不升级，只把基准点往前推。"""
    verifier = FakeVerifier(success=True)
    mile = FakeMonitor("milestone", [0.9])
    policy = make_cascade(space, [0.0], milestone=mile, verifier=verifier)
    traj = run(space, policy, max_steps=4)

    assert verifier.calls, "里程碑触发时必须真的去验证"
    assert traj.n_escalated == 0, "验证通过不该升级"


def test_milestone_verification_fail_escalates(space: ActionSpace) -> None:
    """验证不通过 -> 语义漂移 -> 升级。"""
    verifier = FakeVerifier(success=False)
    mile = FakeMonitor("milestone", [0.9])
    policy = make_cascade(space, [0.0], milestone=mile, verifier=verifier)
    traj = run(space, policy, max_steps=4)

    assert verifier.calls
    assert traj.n_escalated >= 1


def test_verifier_receives_before_and_after(space: ActionSpace) -> None:
    """验证器必须拿到前后截图和任务描述，否则判不了。"""
    verifier = FakeVerifier(success=True)
    policy = make_cascade(space, [0.0], milestone=FakeMonitor("m", [0.9]), verifier=verifier)
    run(space, policy, max_steps=4)

    packet = verifier.calls[0]
    assert packet.task == "任务"
    assert packet.reasoning
    assert packet.steps_since >= 0


# --------------------------------------------------------------------------
# 三组对照共用同一套口径
# --------------------------------------------------------------------------


def test_all_three_arms_share_one_loop(space: ActionSpace) -> None:
    """全大 / 全小 / 级联——结构上走的是同一个主循环，口径必然一致。"""
    small = SingleModelPolicy(
        ScriptedModel("small", ["Action: click(index=1)"]), space, label="small", is_strong=False
    )
    large = SingleModelPolicy(
        ScriptedModel("large", ["Action: click(index=2)"]), space, label="large", is_strong=True
    )
    cascade = make_cascade(space, [0.0, 0.0, 0.9, 0.9, 0.9, 0.9])

    arms = {
        "always_small": run(space, small, max_steps=5),
        "always_large": run(space, large, max_steps=5),
        "cascade": run(space, cascade, max_steps=5),
    }

    # 三条轨迹的形状完全一样：同一步数、同一套 summary 字段
    assert len({len(t) for t in arms.values()}) == 1
    keys = {frozenset(t.summary()) for t in arms.values()}
    assert len(keys) == 1

    # 级联的升级率严格落在两个基线之间
    assert arms["always_small"].escalation_rate == 0.0
    assert arms["always_large"].escalation_rate == 1.0
    assert 0.0 < arms["cascade"].escalation_rate < 1.0


def test_router_reset_between_tasks(space: ActionSpace) -> None:
    """每个任务开始时路由器归零，否则上一个任务的状态会污染下一个。"""
    policy = make_cascade(space, [0.9])
    router = policy.router
    run(space, policy, max_steps=3)
    assert router._hold_left >= 0
    run(space, policy, max_steps=3)
    assert router._last_milestone_idx == 0   # 归零了
