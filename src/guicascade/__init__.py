"""guicascade —— 一个极小、可插拔的 GUI Agent 框架，带步级级联路由。

设计目标只有一句话：**把「Agent 每一步该做什么」和「这一步该用多大的模型」
这两件事彻底解耦**。

前者是策略（Policy），后者是路由（Router）。框架不预设任何一种，
两者都是可替换的部件，可以自由组合：

    Agent(env, policy)                      # 单模型，最朴素的基线
    Agent(env, CascadePolicy(small, large)) # 小模型打底 + 按需升级

快速上手见 README；设计说明见 docs/DESIGN.md。
"""

from __future__ import annotations

from .agent import Agent
from .router import CascadeRouter, Router, RouterDecision
from .trace import Tracer
from .types import Action, Decision, Observation, Step, StepResult, Trajectory

__version__ = "0.1.0"

__all__ = [
    "Action",
    "Agent",
    "CascadeRouter",
    "Decision",
    "Observation",
    "Router",
    "RouterDecision",
    "Step",
    "StepResult",
    "Tracer",
    "Trajectory",
    "__version__",
]
