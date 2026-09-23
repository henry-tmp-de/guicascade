"""策略：决定「这一步做什么」。

框架里只有两种策略，但它们撑起了整个项目的对照实验：

    SingleModelPolicy(large)              <- 全程大模型（上界基线）
    SingleModelPolicy(small)              <- 全程小模型（下界基线）
    CascadePolicy(small, large, router)   <- 级联（本项目）

**三者共用同一个主循环、同一份日志、同一套统计口径。** 这点比看起来重要：
很多复现工作的基线是单独写的脚本，最后对比时说不清"差异是方法带来的还是
实现差异带来的"，结构上就杜绝了这个问题。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Protocol, Sequence, runtime_checkable

from .actions import ActionSpace, DecodeError, decode
from .models.base import Message, Model
from .prompts import STEP_TEMPLATE, SYSTEM_TEMPLATE, render_episode
from .router import CascadeRouter, Router
from .tools import Toolkit
from .types import Decision, Observation, Step

__all__ = ["Policy", "SingleModelPolicy", "CascadePolicy", "DEFAULT_HANDOFF"]


@runtime_checkable
class Policy(Protocol):
    """决定每一步做什么。"""

    name: str

    def reset(self) -> None: ...

    def act(
        self, task: str, observation: Observation, history: Sequence[Step]
    ) -> Decision: ...


# 升级时注入给大模型的一段经验。
#
# 这不只是客套话。原方法特别强调：升级时必须**把之前的经历重新讲一遍**，
# 否则大模型接手时不知道自己是从半路来的，会从第一步重来。
#
# 这段话也是最值得按任务定制的地方——比如发现"卡住多半是因为有弹窗挡着"，
# 就在这里写清楚。**不做技能加载机制，但留这一个可配置的注入点**，
# 是成本收益最好的位置。
DEFAULT_HANDOFF = """\
【交接说明】
你接手的是一个已经执行到一半的任务。前面用的是较小的模型，它现在出了问题
（检测到它在原地打转，或者它以为完成了其实没完成）。

请先看上面的历史，判断当前真实进度，然后**接着往下做**，
不要从头重新开始。特别检查：
- 是不是有个弹窗/对话框挡在中间，导致之前的点击都落空了？
- 之前反复失败的那个操作，是不是本来就点错了元素？
"""


@dataclass
class SingleModelPolicy:
    """一个模型，从头做到尾。

    这是框架最简单也最重要的策略——**两个基线都是它**，只是换一个模型。
    """

    model: Model
    action_space: ActionSpace
    toolkit: Toolkit | None = None
    label: str = "small"
    """写进轨迹的模型标识。级联里靠它统计"这一步是谁做的"。"""

    is_strong: bool = False
    """这个策略用的是不是强模型（大的那个）。

    存在的理由只有一个——**让三组对照能用同一个指标比较**。
    `Decision.escalated` 在本框架里的语义是「这一步由强模型执行」：
    级联里它是路由算出来的，单模型基线里它恒真或恒假。
    不这么统一的话，「大模型调用占比」这个核心指标在三条臂上算法都不一样，
    对比表就没法看。
    """

    max_history: int = 6
    """历史里带多少步。默认 6 和监控器的窗口对齐，两边看到的东西一致。"""

    max_format_retries: int = 2
    """解析失败时，最多让模型重说几次。

    **这是解析层最重要的一道保险。** 解析器可以把 `click(3)`、JSON、键值对
    这些常见写法都兜住，但模型的表达方式是无穷的——靠加正则去打地鼠永远打不完。
    真正的解法是**把错误喂回去让它自己改**：

        模型: "我看到设置图标，应该点它。\\n下一步：点击设置"
        框架: "没能从你的回复里看出动作。请按 Action: 动作名(参数=值) 的格式重写。"
        模型: "Action: click(index=7)"        ← 改对了

    实测这个机制救回了不少步。代价是失败时多花一两次模型调用——比丢掉
    整条轨迹便宜得多。

    重试仍失败则抛错，由上层决定怎么处理（见 `Agent` 的说明）。
    """

    include_image: bool = False
    """要不要把截图一并发给模型。

    默认 **False**，因为有两类模型：

    - **纯文本模型**：只吃无障碍树渲染出的文本，发图它也不看，白花带宽和 prefill
    - **视觉模型**（Qwen3-VL 这类）：截图能提供无障碍树给不了的信息——
      图标长什么样、界面渲染有没有出错、弹窗是不是挡住了

    默认关掉是刻意的：截图在安卓环境里值 ~1.9 秒/步（见 envs/android.py），
    开着它就该是**明确的选择**而不是默认行为。
    """

    _system_cache: str | None = field(default=None, repr=False)

    def reset(self) -> None:
        """无状态，但保持接口一致——级联需要它。"""

    def act(
        self,
        task: str,
        observation: Observation,
        history: Sequence[Step],
        *,
        extra: str = "",
    ) -> Decision:
        """生成一个决策。

        Args:
            extra: 额外注入的文本（级联用它做交接说明），放在任务之后、
                屏幕之前。
        """
        messages: list[Message] = [
            {"role": "system", "content": self._system()},
            {"role": "user", "content": self._content(task, observation, history, extra)},
        ]

        latency = 0.0
        last_error: DecodeError | None = None

        for attempt in range(self.max_format_retries + 1):
            t0 = time.perf_counter()
            response = self.model.generate(messages)
            latency += time.perf_counter() - t0

            try:
                reason, action = decode(response, self.action_space, self.toolkit)
            except DecodeError as e:
                last_error = e
                if attempt == self.max_format_retries:
                    break
                # 把模型说的原话和解析器的抱怨一起喂回去——只说"格式不对"它
                # 不知道自己错在哪，看到自己的输出才有得改
                messages.append({"role": "assistant", "content": response.text})
                messages.append({"role": "user", "content": str(e)})
                continue

            return Decision(
                reason=reason,
                action=action,
                model=self.label,
                raw=response.text,
                latency_s=latency,
                escalated=self.is_strong,
                format_retries=attempt,
            )

        raise last_error if last_error else DecodeError("解析失败")

    # ------------------------------------------------------------------

    def _system(self) -> str:
        """系统提示词。**只渲染一次并缓存。**

        它是整个请求里唯一完全静态的部分，每次都重算的话，一旦将来接上
        按前缀缓存的推理服务，缓存命中率会被无谓地打掉。
        """
        if self._system_cache is None:
            self._system_cache = SYSTEM_TEMPLATE.format(action_space=self._describe_actions())
        return self._system_cache

    def _describe_actions(self) -> str:
        """把动作和工具合成一份清单给模型看。"""
        blocks = [self.action_space.describe()]
        if self.toolkit is not None and self.toolkit.tools:
            blocks.append("（以下是通用工具，不改动屏幕）")
            blocks.append(self.toolkit.describe())
        return "\n".join(b for b in blocks if b)

    def _content(self, task: str, observation: Observation, history: Sequence[Step], extra: str):
        """拼这一步的 user 消息内容。

        纯文本模型返回字符串，视觉模型返回 OpenAI 的多模态数组。

        历史是**渲染成文本**塞进来的（而不是保留多轮消息结构），两个理由：
        一是提示词长度有界，不会随步数无限增长；二是渲染格式和喂给监控器的
        完全一致——**模型看到的和监控器看到的必须是同一份东西**，
        否则监控器的判断就没有依据。
        """
        parts = [f"# 任务\n\n{task}"]
        if extra:
            parts.append(extra)
        parts.append(_screen_block(observation.text))
        parts.append(f"# 到目前为止\n\n{render_episode(history, max_steps=self.max_history)}")
        parts.append("请按工作流程思考，然后输出下一步动作。")
        text = "\n\n".join(parts)

        if not (self.include_image and observation.image):
            return text

        # 图放在文字之后：先给结构化的无障碍树，再给像素。
        # 反过来会让模型先看图、再看文字，容易忽略后者里更精确的元素序号。
        from .models.openai_compat import image_part, text_part

        return [text_part(text), image_part(observation.image)]


@dataclass
class CascadePolicy:
    """小模型打底，监控器报警时才上大模型。

    这个类本身很薄——**真正的判断逻辑全在 `Router` 里**。这样切是为了让
    "换一种路由方式"变成换一个 Router，而不是改这个类。
    """

    small: SingleModelPolicy
    large: SingleModelPolicy
    router: Router = field(default_factory=CascadeRouter)
    handoff: str = DEFAULT_HANDOFF
    """升级时注入的交接说明。想定制就在这里换一段文本。"""

    name: str = "cascade"

    _prev_used_large: bool = field(default=False, repr=False)
    _toolkit: Toolkit | None = field(default=None, repr=False)

    @property
    def toolkit(self) -> Toolkit | None:
        """转发给两个子策略，保持「只有一个 toolkit 实例」的约束。"""
        return self._toolkit

    @toolkit.setter
    def toolkit(self, value: Toolkit | None) -> None:
        self._toolkit = value
        self.small.toolkit = value
        self.large.toolkit = value

    def reset(self) -> None:
        self.router.reset()
        self._prev_used_large = False

    def act(
        self, task: str, observation: Observation, history: Sequence[Step]
    ) -> Decision:
        routed = self.router.route(task, history)

        # 只在"刚从小的切到大的"那一刻注入交接说明——每一步都塞就没意义了，
        # 反而会污染上下文
        just_escalated = routed.use_large and not self._prev_used_large and bool(history)
        self._prev_used_large = routed.use_large

        worker = self.large if routed.use_large else self.small
        decision = worker.act(
            task, observation, history, extra=self.handoff if just_escalated else ""
        )

        # 把监控信号和升级标记打回决策上，落盘后就能解释"这一步为什么换人"
        return replace(
            decision,
            model=worker.label,
            signals=dict(routed.signals),
            escalated=routed.use_large,
        )


def _screen_block(text: str) -> str:
    """屏幕内容带来源标记。见 prompts.py 里关于提示注入的说明。"""
    return f'# 当前屏幕\n\n<screen source="device">\n{text}\n</screen>'
