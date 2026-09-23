"""轨迹落盘：每步一行 JSONL。

为什么是 JSONL 而不是一个大 JSON：**跑到一半崩了，前面几十步不能丢。**
GUI 实验一轮几十分钟，整份文件最后才写的话，一次崩溃就白跑。一行一写、
立刻 flush，代价是文件大一点，换来的是"任何时候中断都拿得到已有数据"。

第二个理由：JSONL 可以直接 `grep`、`jq`、`pandas.read_json(lines=True)`，
不需要先写解析代码才能看一眼数据。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import Step, Trajectory

__all__ = ["Tracer"]


class Tracer:
    """把轨迹写到 JSONL。传 path=None 就是不落盘（测试时用）。"""

    def __init__(self, path: str | Path | None = None, *, include_summary: bool = True) -> None:
        """
        Args:
            include_summary: 每个任务结束时额外写一行 `_type: "trajectory"` 的
                汇总。分析时不用自己扫全量 step 再聚合，直接过滤这一种行即可。
        """
        self.path = Path(path) if path is not None else None
        self.include_summary = include_summary
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_step(self, task: str, step: Step) -> None:
        self._write({"_type": "step", "task": task, **step.to_dict()})

    def log_trajectory(self, trajectory: Trajectory) -> None:
        if not self.include_summary:
            return
        self._write({"_type": "trajectory", **trajectory.summary()})

    def _write(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        # 一行一开一关：崩溃时最多丢当行，不会丢整个文件
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
