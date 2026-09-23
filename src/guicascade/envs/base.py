"""环境层：把「一个动作」落到真实界面上，并回报新状态。

这一层是框架里唯一知道「界面长什么样」的地方。桌面、安卓、浏览器、
离线回放——它们的差异全部被关在这个接口后面，上层 Agent 一概不知。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Action, Observation, StepResult

__all__ = ["Environment"]


@runtime_checkable
class Environment(Protocol):
    """一个可交互的 GUI 环境。

    生命周期固定为 `reset` → `step`* → `close`，三段式足够表达
    所有 GUI 环境（含真实设备、模拟器、浏览器、离线回放）。

    为什么 `reset` 要接收 task：不同任务的初始状态不一样（该打开哪个 app、
    从哪个页面开始），环境必须知道。返回初始观察而不是让调用方再问一次，
    少一次无意义的往返。
    """

    name: str
    """环境标识，用于日志与结果表。"""

    def reset(self, task: str) -> Observation:
        """把环境恢复到任务的初始状态，返回第一步要看的观察。"""
        ...

    def step(self, action: Action) -> StepResult:
        """执行一个动作，返回新观察和执行结果。

        实现约定：**不要在这里抛异常来表达「动作无效」**。点到了不存在的
        元素是 GUI 任务的常态而非意外，应当返回 `StepResult(ok=False, error=...)`，
        把控制权交回给 Agent 继续决策。只有环境本身坏掉（设备掉线）才抛。
        """
        ...

    def success(self) -> bool | None:
        """环境对当前 episode 的判分。

        单独一个方法而不是只靠 `StepResult.success`，是因为有些环境
        （比如 OSWorld）的判分是独立的一次检查，不在每步返回里。
        episode 结束时调用。返回 None 表示"这个环境不提供判分"。
        """
        ...

    def close(self) -> None:
        """释放资源。要保证可重复调用。"""
        ...
