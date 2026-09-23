"""假环境：按剧本返回观察，用于测试。

和 `models/scripted.py` 是一对。有了这两个，**整个框架能在没有 GPU、
没有模拟器、没有 API key 的情况下跑通**——这是这个仓库能被验证的前提。

它同时让"环境出错时框架怎么办"变得可测：动作失败了会不会继续、判分拿不到
会不会崩、`close()` 有没有被调用，全都可以断言。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Action, Observation, StepResult

__all__ = ["ScriptedEnvironment"]


@dataclass
class ScriptedEnvironment:
    """按剧本依次返回观察。

    Example:
        >>> env = ScriptedEnvironment(["首页", "设置页", "完成"])
        >>> env.reset("打开设置").text
        '首页'
        >>> env.step(Action("click", {"index": 1})).observation.text
        '设置页'
    """

    screens: list[str] = field(default_factory=lambda: ["screen"])
    """每一步看到的屏幕文本。用完后停在最后一个。"""

    fail_actions: set[str] = field(default_factory=set)
    """这些动作名会被判为执行失败（`ok=False`），用来测错误处理路径。"""

    done_at: int | None = None
    """第几步之后结束。None 表示由剧本长度决定。"""

    success_at_end: bool | None = True
    """`success()` 的返回值。None 表示这个环境不提供判分。"""

    name: str = "scripted"
    closed: bool = field(default=False, init=False, repr=False)
    actions: list[Action] = field(default_factory=list, init=False, repr=False)
    """记录收到的所有动作，供测试断言。"""

    _cursor: int = field(default=0, init=False, repr=False)

    def reset(self, task: str) -> Observation:
        self._cursor = 0
        self.actions.clear()
        self.closed = False
        return Observation(text=self.screens[0] if self.screens else "", meta={"task": task})

    def step(self, action: Action) -> StepResult:
        self.actions.append(action)

        idx = min(self._cursor + 1, len(self.screens) - 1)
        self._cursor = idx
        observation = Observation(text=self.screens[idx] if self.screens else "")

        ok = action.name not in self.fail_actions
        limit = self.done_at if self.done_at is not None else len(self.screens) - 1
        done = idx >= limit

        return StepResult(
            observation=observation,
            done=done,
            success=self.success_at_end if done else None,
            ok=ok,
            error="" if ok else f"动作 {action.name!r} 执行失败",
        )

    def success(self) -> bool | None:
        return self.success_at_end

    def close(self) -> None:
        self.closed = True
