"""Agent —— 框架的核心循环。

整个文件只有一个类，读完不需要翻第二个文件。这是刻意的：主循环应当小到
可以在脑子里完整跑一遍，复杂度放在可替换的部件里（策略、环境、监控器）。

主循环本身**不知道级联的存在**。它只知道「问策略要一个决策，把决策交给
该执行的人」。换一个策略就能从单模型变成级联、变成别的路由方式，
这个文件一个字都不用改。
"""

from __future__ import annotations

import time

from .envs.base import Environment
from .policies import Policy
from .tools import Toolkit
from .trace import Tracer
from .types import TOOL, Decision, Observation, Step, StepResult, Trajectory

__all__ = ["Agent"]


class Agent:
    """把「策略」「环境」「工具」「追踪」接起来的执行器。

    Example:
        >>> agent = Agent(env, CascadePolicy(small, large, router), toolkit=kit)
        >>> traj = agent.run("把闹钟设到明早 7 点")
        >>> traj.success, traj.escalation_rate
    """

    def __init__(
        self,
        env: Environment,
        policy: Policy,
        toolkit: Toolkit | None = None,
        tracer: Tracer | None = None,
        max_steps: int = 50,
    ) -> None:
        """
        Args:
            max_steps: 步数上限。GUI 任务卡住时不设上限会烧光预算，
                所以这是必需的保护而非可选项。
        """
        self.env = env
        self.policy = policy
        self.toolkit = toolkit if toolkit is not None else Toolkit()
        self.tracer = tracer
        self.max_steps = max_steps

        # 策略需要工具清单来拼提示词。在这里统一注入，是为了避免调用方
        # 给 Agent 和 Policy 各传一个实例——NoteTool 这类工具是有状态的，
        # 两个实例会导致「模型记了笔记但下一步看不见」这种极难排查的问题。
        if hasattr(policy, "toolkit"):
            policy.toolkit = self.toolkit

    def run(self, task: str) -> Trajectory:
        """跑完一个任务，返回完整轨迹。

        无论中途发生什么，环境都会被关闭——GUI 环境（模拟器、浏览器、
        真机连接）是稀缺资源，泄漏一个会影响后续所有实验。
        """
        # 有状态的策略（比如级联要记住上次里程碑）需要在任务开始时归零
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

        trajectory = Trajectory(task=task)
        observation: Observation | None = None

        try:
            observation = self.env.reset(task)

            for index in range(self.max_steps):
                # —— 策略侧：这一步用哪个模型、做什么 ——
                decision = self.policy.act(task, observation, trajectory.steps)

                # —— 执行侧：按动作类型分派 ——
                step = self._execute(index, observation, decision)
                trajectory.steps.append(step)

                if self.tracer is not None:
                    self.tracer.log_step(task, step)

                if step.result is not None and step.result.done:
                    trajectory.reward = self._reward(step, trajectory)
                    break

                # GUI 动作改变了屏幕 -> 取新观察；工具没改屏幕 -> 沿用旧的。
                # 这条分支是 kind 字段存在的全部理由：工具调用后重新截图
                # 会白花一次环境开销（真机上是几百毫秒）。
                if decision.action.kind != TOOL and step.result is not None:
                    observation = step.result.observation

        finally:
            self.env.close()
            if self.tracer is not None:
                self.tracer.log_trajectory(trajectory)

        return trajectory

    # ------------------------------------------------------------------
    # 执行：两条路，各几行
    # ------------------------------------------------------------------

    def _execute(self, index: int, observation: Observation, decision: Decision) -> Step:
        """执行一个决策，返回对应的 `Step`。

        两侧耗时分别计量：模型侧在 `Decision.latency_s`（由策略记录），
        环境侧在这里。级联省掉的是前者，后者一分都省不掉。
        """
        t0 = time.perf_counter()

        if decision.action.kind == TOOL:
            tool_result = self.toolkit.run(decision.action.name, decision.action.args)
            return Step(
                index=index,
                observation=observation,
                decision=decision,
                tool_output=tool_result.output,
                latency_env_s=time.perf_counter() - t0,
                # 观察原样传回：工具不改屏幕，`done` 由工具自己声明
                result=StepResult(observation=observation, done=tool_result.done),
            )

        result = self.env.step(decision.action)
        return Step(
            index=index,
            observation=observation,
            decision=decision,
            latency_env_s=time.perf_counter() - t0,
            result=result,
        )

    def _reward(self, step: Step, trajectory: Trajectory) -> float | None:
        """确定本次 episode 的最终判分。

        优先用这一步环境给的判分；没有就问环境要一次（有些环境的判分是
        独立的一次检查，不在每步返回里）；再没有就沿用轨迹里最后一次
        明确的判分。
        """
        if step.result is not None and step.result.success is not None:
            return float(step.result.success)

        try:
            final = self.env.success()
        except Exception:  # noqa: BLE001 - 判分失败不该抹掉整条轨迹
            final = None
        if final is not None:
            return float(final)

        for s in reversed(trajectory.steps):
            if s.result is not None and s.result.success is not None:
                return float(s.result.success)
        return None
