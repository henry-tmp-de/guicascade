"""全框架共用的词汇表。

这个文件定义「一次 GUI 交互」的最小词汇，是模型层、环境层、监控层之间
唯一的公共语言。三层都只说这套词汇，彼此不知道对方的实现——这是整个
框架可插拔性的来源。

刻意不引入任何第三方依赖：只有标准库。任何模块 import 它都不会有负担。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "Action",
    "Observation",
    "Decision",
    "StepResult",
    "Step",
    "Trajectory",
    "ToolCall",
    "ModelResponse",
    "GUI",
    "TOOL",
]

GUI = "gui"
"""动作交给环境执行 —— 会改变屏幕，**必须**重新获取观察。"""

TOOL = "tool"
"""动作交给工具集合执行 —— 不改变屏幕，观察保持不变。

区分这两者不是分类癖：GUI 动作后不重新截图，模型就是拿旧屏幕做决策；
工具调用后重新截图，则白花一次环境开销（真机上截图 + 无障碍树 dump
是几百毫秒级的事）。"""



@dataclass(frozen=True, slots=True)
class Action:
    """一次 GUI 动作。

    刻意做成「动作名 + 参数字典」而不是给每种动作定义一个类：
    不同环境（桌面 / 安卓 / 浏览器）的动作空间差别很大，参数形态也不一样，
    用一个开放的字典比一套僵硬的类层次更好扩展，也更贴近实际落地的写法。

    Example:
        Action("click", {"index": 3})
        Action("type", {"text": "hello"})
        Action("scroll", {"direction": "down"})
    """

    name: str
    args: Mapping[str, Any] = field(default_factory=dict)
    kind: str = GUI
    """执行者是谁，见 `GUI` / `TOOL` 两个常量的说明。"""

    def __str__(self) -> str:
        """渲染成人类/模型可读的一行，也是喂给监控器的文本形态。"""
        if not self.args:
            return self.name
        inner = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.name}({inner})"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": dict(self.args)}


@dataclass(frozen=True, slots=True)
class Observation:
    """环境在某一时刻的状态。

    `text` 和 `image` 是两条独立的通道，因为本框架的两类消费者需要的
    东西不一样：

    - **策略模型**：可以两者都要（视觉模型吃 image，纯文本模型只吃 text）
    - **监控器**：刻意只吃 `text`——原方法明确把常驻监控器限制在
      文本的理由-动作轨迹上，不看截图，以此换取「每步都能跑」的廉价性

    所以 `image` 是可选的：一个纯文本环境（比如离线轨迹回放）可以完全不提供它。
    """

    text: str
    image: bytes | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_image(self) -> bool:
        return self.image is not None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """模型请求调用一个具名函数。

    和 `Action` 的区别值得说清：`ToolCall` 是**模型侧**的原始表达——它可能
    调了个不存在的工具、可能参数缺了必填项。`Action` 是**校验通过、可以直接
    执行**的表达。两者之间隔着 `ActionSpace` 那道翻译 + 校验。
    """

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """模型的一次回应，两个通道并存。

    这是本框架同时支持「原生 tool calling」和「文本解析」的关键：

    - `text`       —— 自由文本。级联的常驻监控器**只吃这个通道**（它读的是
                      "理由"），所以无论用不用 tool calling，这条通道都不能省。
    - `tool_calls` —— 结构化动作，由 API 保证格式合法，不用正则去猜。

    两条通道互不排斥：一次回应完全可以既有思考文字、又有结构化工具调用。
    这正好各取所需——文字喂监控器，结构化动作直接构造 `Action`。
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    raw: Any = None
    """供应商返回的原始对象，排查解析问题时用。"""

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass(frozen=True, slots=True)
class Decision:
    """策略对当前观察给出的回答——「理由 + 动作」以及它是怎么来的。

    策略侧的所有信息都收在这个对象里，和环境侧的 `StepResult` 分开，
    这样归因很干净：出了问题一眼能看出是「模型想错了」还是「环境没执行对」。
    """

    reason: str
    action: Action
    model: str = ""
    """产出这个决策的模型标识（如 "small" / "large"），用于统计升级比例。"""

    raw: str = ""
    """模型未经解析的原始输出。调试解析器时最有用，也方便事后复盘。"""

    latency_s: float = 0.0
    """模型侧耗时（秒）。级联的核心收益就体现在这个数上。"""

    signals: Mapping[str, float] = field(default_factory=dict)
    """这一步各路监控器打的分，如 {"stuck": 0.08, "milestone": 0.71}。"""

    escalated: bool = False
    """这一步是否由**强模型**执行。

    语义统一到"谁执行的"而不是"是不是级联触发的"，是为了让三组对照
    （全大 / 全小 / 级联）能用同一个指标比较——全大基线恒真、全小恒假、
    级联是路由的产物。见 `policies.SingleModelPolicy.is_strong`。
    """


@dataclass(frozen=True, slots=True)
class StepResult:
    """环境执行完一个动作后的回执。"""

    observation: Observation
    done: bool = False
    """episode 是否应当结束（任务成功、失败、或达到环境自身的终止条件）。"""

    success: bool | None = None
    """程序化判分结果。None 表示「还没判」或「这个环境不提供」。"""

    ok: bool = True
    """动作本身是否被环境成功执行（区别于任务是否成功）。
    点击了一个不存在的元素 -> ok=False；点了但没达成目标 -> ok=True, success=False。"""

    error: str = ""
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Step:
    """轨迹里的一步 = 策略侧的一个决策 + 环境侧的一次执行。

    这个对象同时是 JSONL 落盘的单元，所以字段刻意保持扁平好读。
    """

    index: int
    observation: Observation
    decision: Decision
    tool_output: str = ""
    """工具调用的返回文本（只有 `kind=TOOL` 的步才有）。

    工具不改变屏幕，所以这些步骤没有新观察——结果就记在这里，
    由提示词渲染器拼进历史，让模型下一步能看见。"""

    latency_env_s: float = 0.0
    """环境侧耗时（秒）：截图、adb 通信、界面等待等等。

    ⚠️ 必须和 `Decision.latency_s`（模型侧）分开记。
    级联省掉的是「模型推理」，环境开销一分都省不掉——把两者混在一起报，
    数据就没法解释，这是这类工作最常见的坑。
    """

    result: StepResult | None = None

    @property
    def model(self) -> str:
        return self.decision.model

    @property
    def escalated(self) -> bool:
        return self.decision.escalated

    def to_dict(self) -> dict[str, Any]:
        """展开成 JSONL 的一行。"""
        d: dict[str, Any] = {
            "step": self.index,
            "model": self.decision.model,
            "reason": self.decision.reason,
            "action": str(self.decision.action),
            "escalated": self.decision.escalated,
            "latency_model_s": round(self.decision.latency_s, 4),
            "latency_env_s": round(self.latency_env_s, 4),
        }
        d.update({k: round(v, 4) for k, v in self.decision.signals.items()})
        if self.result is not None:
            d["ok"] = self.result.ok
            d["done"] = self.result.done
            if self.result.success is not None:
                d["success"] = self.result.success
            if self.result.error:
                d["error"] = self.result.error
        return d


@dataclass(slots=True)
class Trajectory:
    """一个任务从头到尾的完整记录，也是评测的基本单元。"""

    task: str
    steps: list[Step] = field(default_factory=list)
    reward: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def success(self) -> bool:
        return bool(self.reward) if self.reward is not None else False

    # ---- 级联专属的统计口径 ----
    # 这些指标是判断「级联到底划不划算」的核心，所以直接做成属性而不是
    # 让每个脚本自己数一遍。

    @property
    def n_escalated(self) -> int:
        return sum(1 for s in self.steps if s.escalated)

    @property
    def escalation_rate(self) -> float:
        return self.n_escalated / len(self.steps) if self.steps else 0.0

    @property
    def model_latency_s(self) -> float:
        """模型侧总耗时——级联真正能省的那部分。"""
        return sum(s.decision.latency_s for s in self.steps)

    @property
    def env_latency_s(self) -> float:
        """环境侧总耗时——级联省不掉的那部分。"""
        return sum(s.latency_env_s for s in self.steps)

    def summary(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "steps": len(self.steps),
            "reward": self.reward,
            "escalated": self.n_escalated,
            "escalation_rate": round(self.escalation_rate, 4),
            "latency_model_s": round(self.model_latency_s, 3),
            "latency_env_s": round(self.env_latency_s, 3),
            "latency_total_s": round(self.model_latency_s + self.env_latency_s, 3),
        }
