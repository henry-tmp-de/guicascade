"""ModernBERT 监控器：官方那套 149M 文本分类器。

## 两种输入格式，必须严格遵守

原方法给两个监控器喂的**不是同一种文本**，这是从官方源码逐行核对出来的：

    卡住   :  "Step N:\\nResponse: {理由}\\nAction: {动作}\\n"，取最近 6 步
    里程碑 :  "Task: {任务}\\n" + 逐行 "Step N:\\n{理由}"，取最近 6 步

差别不是随意的——卡住监控器关注"动作有没有在重复"，所以需要动作单列一行；
里程碑监控器关注"有没有进展"，所以必须带上任务描述。**喂错了分数就是噪声。**

## ⚠️ 实测结论：这两个模型的可用性完全不同

在真实同源轨迹上实测（见 README）：

    里程碑监控器   AUC 0.78   ✅ 可用，阈值校准后可直接进级联
    卡住监控器     AUC 0.61   ❌ 不可用

卡住侧的"不可用"排除过五种解释（实现 / 格式 / 语言 / 样本选择 / 数据文件），
最后定位到权重本身。**所以 `mode="stuck"` 目前只作为对照保留**，
生产路径请用 `monitors/prompt.py` 的提示词版。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..types import Step
from .base import Monitor

__all__ = ["BertMonitor"]

_HISTORY = 6
"""窗口长度。和官方 `format_step_history(max_history_steps=6)` 对齐。"""


@dataclass
class BertMonitor:
    """加载一个微调过的 ModernBERT 做逐步打分。

    实现 `Monitor` 协议。依赖 torch/transformers，**只在真正用到时才 import**——
    这样不装 torch 也能跑框架的其余部分和测试。
    """

    name: str = "bert"
    model_path: str = ""
    threshold: float = 0.5
    mode: str = "stuck"
    """`"stuck"` 或 `"milestone"`。决定输入怎么拼——两者格式不同，见模块说明。"""

    max_length: int = 2048
    device: str | None = None
    history: int = _HISTORY

    _tok: Any = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _torch: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in ("stuck", "milestone"):
            raise ValueError(f"mode 必须是 stuck 或 milestone，收到 {self.mode!r}")

    def _ensure_loaded(self) -> None:
        """**第一次打分时才加载权重。**

        构造时就加载会让"读一下配置"也要装 torch——配置解析、测试、
        以及只想跑单模型基线的人，都不该被迫装 2GB 的依赖。
        """
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self.device).eval()

    # ------------------------------------------------------------------

    def render(self, task: str | None, steps: Sequence[Step]) -> str:
        """按对应格式拼出输入文本。**改这里之前先看模块说明的格式对照。**"""
        recent = list(steps)[-self.history :]
        if self.mode == "milestone":
            parts = [f"Task: {task or ''}\n"]
            for i, s in enumerate(recent, start=1):
                parts.append(f"Step {i}:\n{s.decision.reason}")
            return "\n".join(parts)
        return "\n".join(
            f"Step {i}:\nResponse: {s.decision.reason}\nAction: {s.decision.action}\n"
            for i, s in enumerate(recent, start=1)
        )

    def score(self, task: str | None, steps: Sequence[Step]) -> float:
        text = self.render(task, steps)
        if not text.strip():
            return 0.0

        self._ensure_loaded()
        enc = self._tok(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self._torch.no_grad():
            logits = self._model(**enc).logits
            probs = self._torch.softmax(logits, dim=-1)
        return float(probs[0, 1].item())  # 类别 1 = 正类（卡住 / 里程碑）
