"""监控层：读轨迹文本，给出「该不该升级」的信号。

这是整个级联机制的判断中枢，也是本项目复现的核心。监控器是**常驻**的
——每一步都要跑一次，所以必须极轻（本框架用的 ModernBERT 只有 149M）。

⚠️ 一条贯穿始终的设计约束：**监控器刻意只看文本，不看截图**。

理由来自原方法：把常驻监控器限制在「理由 + 动作」的文本轨迹上，换来
三个好处——轻量、与具体策略模型无关、每步都能跑。代价是它看不见
「嘴上说点发送、手上点到隔壁删除」这类**空间错位**，这是该机制已知的
弱点，也是本项目要专门做误报/漏报定性分析的原因。

所以本框架的 `Monitor` 接口签名里**根本没有图像参数**——不是忘了，
是把这个约束写进了类型里。
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..types import Step

__all__ = ["Monitor"]


@runtime_checkable
class Monitor(Protocol):
    """一个逐步打分的轻量监控器。"""

    name: str
    """监控器标识，会作为 `Decision.signals` 的键写进轨迹。"""

    threshold: float
    """触发阈值。⚠️ 官方发布的里程碑监控器判别力很弱，直接用 0.5 基本
    不会触发，实际部署前必须重新校准——见 scripts/calibrate_thresholds.py。"""

    def score(self, task: str | None, steps: Sequence[Step]) -> float:
        """给出「当前这一步属于目标类别」的概率，值域 [0, 1]。

        Args:
            task: 任务描述。卡住监控器不使用它（只看局部行为的重复性），
                里程碑监控器必须有它（判断「有没有进展」离不开目标）。
                这个差异来自两者关注的对象不同，不是实现偷懒。
            steps: 到目前为止的全部步骤，监控器自行截取最近窗口。

        Returns:
            正类概率。语义由各监控器定义：卡住监控器返回 P(卡住)，
            里程碑监控器返回 P(这步是里程碑)。
        """
        ...
