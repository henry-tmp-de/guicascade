"""假监控器：按预设序列打分，用于测试级联逻辑。

和 `models/scripted.py`、`envs/scripted.py` 凑成一套。有了它，
**"监控器报警时级联会不会正确升级"这件事可以在纯 CPU、零依赖下断言**，
不用加载 600MB 的 BERT，也不用等它跑完。

真实监控器的行为有随机性和长尾，不适合用来验证控制流；控制流该由确定的
假件来测——这也是整个框架到处留 `Scripted*` 的原因。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..types import Step
from .base import Monitor

__all__ = ["ScriptedMonitor"]


@dataclass
class ScriptedMonitor:
    """按剧本依次返回分数，用完后重复最后一个。"""

    name: str = "scripted"
    scores: list[float] = field(default_factory=lambda: [0.0])
    threshold: float = 0.5
    calls: int = field(default=0, init=False, repr=False)

    def score(self, task: str | None, steps: Sequence[Step]) -> float:
        if not self.scores:
            return 0.0
        i = min(self.calls, len(self.scores) - 1)
        self.calls += 1
        return float(self.scores[i])
