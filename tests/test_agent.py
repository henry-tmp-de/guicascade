"""主循环测试。

重点测三个"设计承诺"是不是真的成立：

  1. 环境一定会被关闭（哪怕中途抛异常）—— GUI 环境是稀缺资源
  2. 工具动作**不重新截图**，界面动作**必须重新截图** —— 这是 kind 字段存在的全部理由
  3. 模型延迟和环境延迟分开记 —— 级联省的是前者
"""

from __future__ import annotations

import pytest

from guicascade.actions import ActionSpace, ActionSpec
from guicascade.agent import Agent
from guicascade.envs.scripted import ScriptedEnvironment
from guicascade.models.scripted import CodeBlockModel, ScriptedModel
from guicascade.policies import SingleModelPolicy
from guicascade.tools import FinishTool, NoteTool, Toolkit
from guicascade.types import TOOL


@pytest.fixture
def space() -> ActionSpace:
    # 刻意**不放** finish —— 环境动作在重名时优先，把 finish 放进动作空间
    # 会把同名工具遮住，测的就不是工具通道了
    return ActionSpace(name="t").add(
        ActionSpec(name="click", description="点击",
                   parameters={"type": "object",
                               "properties": {"index": {"type": "integer"}},
                               "required": ["index"]},
                   positional=("index",))
    )


def build(space, script, screens, **kw):
    env = ScriptedEnvironment(screens=screens, **kw)
    policy = SingleModelPolicy(
        model=ScriptedModel("small", script), action_space=space, label="small"
    )
    return env, Agent(env, policy)


# --------------------------------------------------------------------------


def test_runs_and_produces_trajectory(space: ActionSpace) -> None:
    env, agent = build(space, ["Action: click(index=1)"], ["a", "b", "c"])
    traj = agent.run("测试任务")

    assert len(traj.steps) == 2          # 两步走到 done
    assert traj.steps[0].decision.action.name == "click"
    assert traj.reward == 1.0


def test_env_is_closed_even_on_crash(space: ActionSpace) -> None:
    """中途炸了也必须关环境。模拟器/browser 泄漏一个会影响后面所有实验。"""

    class Boom(ScriptedEnvironment):
        def step(self, action):
            raise RuntimeError("设备掉线了")

    env = Boom(screens=["a", "b"])
    policy = SingleModelPolicy(ScriptedModel("m", ["Action: click(index=1)"]), space)
    agent = Agent(env, policy)

    with pytest.raises(RuntimeError):
        agent.run("t")
    assert env.closed is True


def test_max_steps_is_respected(space: ActionSpace) -> None:
    env, agent = build(space, ["Action: click(index=1)"], ["x"] * 50)
    agent.max_steps = 5
    traj = agent.run("永不结束的任务")
    assert len(traj.steps) == 5


# --------------------------------------------------------------------------
# 工具 vs 界面动作：观察该不该更新
# --------------------------------------------------------------------------


def test_tool_does_not_refresh_observation(space: ActionSpace) -> None:
    """工具不改屏幕，所以不该重新取观察——否则白花一次截图开销。"""
    env = ScriptedEnvironment(screens=["首页", "第二屏", "第三屏"])
    kit = Toolkit().add(NoteTool())
    policy = SingleModelPolicy(
        model=ScriptedModel("m", ["Action: note(content='记一笔')"]),
        action_space=space, toolkit=kit,
    )
    agent = Agent(env, policy, toolkit=kit, max_steps=2)
    traj = agent.run("t")

    assert traj.steps[0].decision.action.kind == TOOL
    # 环境一次都没被 step 过 —— 工具没碰屏幕
    assert env.actions == []
    # 第二步看到的观察和第一步一样
    assert traj.steps[1].observation.text == traj.steps[0].observation.text


def test_gui_action_refreshes_observation(space: ActionSpace) -> None:
    env, agent = build(space, ["Action: click(index=1)"], ["第一屏", "第二屏", "第三屏"])
    traj = agent.run("t")
    assert traj.steps[1].observation.text != traj.steps[0].observation.text


def test_tool_output_is_recorded_on_step(space: ActionSpace) -> None:
    """工具结果要留在轨迹里，下一步的提示词才看得到。"""
    env = ScriptedEnvironment(screens=["a", "b", "c"])
    kit = Toolkit().add(NoteTool())
    policy = SingleModelPolicy(ScriptedModel("m", ["Action: note(content='张三')"]),
                               action_space=space, toolkit=kit)
    traj = Agent(env, policy, toolkit=kit, max_steps=1).run("t")
    assert "张三" in traj.steps[0].tool_output


def test_finish_tool_ends_episode(space: ActionSpace) -> None:
    env = ScriptedEnvironment(screens=["a"] * 10)
    kit = Toolkit().add(FinishTool())
    policy = SingleModelPolicy(ScriptedModel("m", ["Action: finish(summary='搞定了')"]),
                               action_space=space, toolkit=kit)
    traj = Agent(env, policy, toolkit=kit, max_steps=10).run("t")
    assert len(traj.steps) == 1


# --------------------------------------------------------------------------
# 延迟分开记
# --------------------------------------------------------------------------


def test_latencies_are_separate(space: ActionSpace) -> None:
    env, agent = build(space, ["Action: click(index=1)"], ["a", "b", "c"])
    traj = agent.run("t")
    assert traj.env_latency_s >= 0.0
    assert traj.model_latency_s >= 0.0
    summary = traj.summary()
    # 汇总里两个数必须都在，混成一个就失去意义了
    assert "latency_model_s" in summary and "latency_env_s" in summary


# --------------------------------------------------------------------------
# 两条通道都走通
# --------------------------------------------------------------------------


def test_toolcall_channel_end_to_end(space: ActionSpace) -> None:
    env = ScriptedEnvironment(screens=["a", "b", "c"])
    policy = SingleModelPolicy(
        model=CodeBlockModel(tool_name="click", arguments={"index": 7}, reason="点它"),
        action_space=space, label="toolcall",
    )
    traj = Agent(env, policy).run("t")
    assert traj.steps[0].decision.action.args["index"] == 7
    assert traj.steps[0].decision.reason == "点它"


def test_failed_action_still_continues(space: ActionSpace) -> None:
    """动作失败是 GUI 任务的常态，不该中断 episode。"""
    env, agent = build(space, ["Action: click(index=1)"], ["a", "b", "c"],
                       fail_actions={"click"})
    traj = agent.run("t")
    assert len(traj.steps) == 2
    assert traj.steps[0].result.ok is False
    assert traj.steps[0].result.error
