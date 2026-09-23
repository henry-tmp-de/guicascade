"""卡住监控器的替代方案：**数重复动作**。纯代码，不加载任何模型。

## 为什么要有这个

原方法给的卡住监控器（149M 的 ModernBERT）在我们的实测里是不成立的：

    桌面轨迹（AgentNet）   AUC 0.608   基本等于瞎猜
    自采安卓轨迹           AUC 0.423   **比瞎猜还差**（正常窗口均分 0.999，
                                      卡住窗口 0.976——反过来了）

五轮排除（实现 / 输入格式 / 语言 / 样本选择 / 数据文件）之后结论是：
**发布的权重本身就没有判别力**，不是我们喂错了东西。

那就没必要非得用一个神经网络去做一件**看一眼动作序列就能判断**的事。
模型在原地打转的定义是现成的：同一件事，它做了好几遍。

## 和神经网络监控器的关系

不是替代品，是**下界基线**。它有几个神经网络给不了的性质：

- 零参数、零加载、零推理延迟（纯 Python 比较字符串）
- 判据完全可解释——升级日志里能直接写"因为它把 click(index=7) 做了 4 遍"
- 换环境不用重新训练

如果 BERT 版本连这个都比不过，那它就没有上线的理由。这是评测该有的顺序。

## ⚠️ 一个反直觉的实测结论（别跳过）

在本项目自采的安卓轨迹上，**重复动作在成功和失败的轨迹里同样常见**：

    成功：open_app(deskclock) 连做 8 次   -> 第 1 步就把 app 打开了，后面全是无效重复
    失败：click(index=7) 连做 8 次        -> 真的是在乱点

两者从动作序列上**完全无法区分**——都说"同一步做了很多遍"，都让屏幕保持不变。
差别只在"目标有没有达成"，而那是里程碑监控器管的事，不是卡住监控器能看到的。

所以这个监控器的定位要摆正：**它不是一个"卡住分类器"，是一个"升级触发器"。**
它的价值不在于预测失败准不准，而在于——

1. 触发得早不早（在轨迹废掉之前叫醒大模型）
2. 代价小不小（每次误触发就是一次多余的大模型调用）

也就是说，**它该用三臂对照（全大 / 全小 / 级联）来评，不该用 AUC 来评**。
拿 AUC 评它，等于用错尺子。详见 `scripts/eval_repeat_detector.py`。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..actions import _split_top_level
from ..types import Step

__all__ = ["RepeatMonitor", "normalize_action", "repeat_stats"]


_CALL = re.compile(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", re.DOTALL)
_STRIP = "\"' \t\n"


def normalize_action(text: str) -> str:
    """把一个动作渲染成可比较的规范形式。

    直接拿原始字符串比是**不够的**，模型的写法会飘：

        open_app(app_name='com.android.settings')
        open_app(app_name="com.android.settings")
        open_app( app_name = com.android.settings )

    这三行是同一个动作。归一化做四件事：去掉空白、去掉引号、参数按名字
    排序、数字统一成 `float` 的规范形式（`2` 和 `2.0` 是同一个等待）。

    归一化不了（不是 call 语法）就退化成"去掉空白的小写串"——**宁可少
    认几个重复，也不要错认**，错认会让一次误升级看起来有理有据。
    """
    s = " ".join(str(text or "").split())
    m = _CALL.match(s)
    if not m:
        return s.lower()

    name, raw = m.group(1).lower(), m.group(2)
    args: dict[str, str] = {}
    for i, part in enumerate(_split_top_level(raw)):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            key = k.strip().lower()
        else:
            # 位置参数：按下标占位，保证 `click(3)` 和 `click(index=3)`
            # 不会被误判成同一个（少了参数名，语义其实是模糊的）
            key = f"#{i}"
            v = part
        val = v.strip().strip(_STRIP)
        try:
            val = repr(float(val))
        except ValueError:
            val = val.lower()
        args[key] = val

    inner = ",".join(f"{k}={v}" for k, v in sorted(args.items()))
    return f"{name}({inner})"


def _keys(steps: Sequence[Step]) -> list[str]:
    return [normalize_action(str(s.decision.action)) for s in steps]


def repeat_stats(
    steps: Sequence[Step], *, window: int = 6
) -> tuple[int, int]:
    """数一数轨迹末尾的重复程度。返回 `(连击, 窗口内最多重复次数)`。

    - **连击（streak）**：末尾连续同一个动作做了几遍。`A A A` -> 3。
    - **窗口内最多（count）**：最近 `window` 步里，同一个动作出现了几次。
      这个能抓住**交替式打转**——实测见过 `click(6) click(7) click(6)
      click(7) ...`，连击数永远是 2，但它显然卡住了。

    两个数都给出来，因为它们在真实数据上抓到的是不同的东西。
    """
    keys = _keys(steps)
    if not keys:
        return 0, 0

    streak = 0
    for k in reversed(keys):
        if k == keys[-1]:
            streak += 1
        else:
            break

    recent = keys[-window:] if window > 0 else keys
    count = max((recent.count(k) for k in set(recent)), default=0)
    return streak, count


@dataclass
class RepeatMonitor:
    """末尾动作重复到一定次数就报警。**不含任何模型，纯字符串比较。**

    打分口径：`score = 命中的重复次数 / min_repeats`，截断到 `[0, 1]`。
    所以 `score == 1.0` 恰好意味着"达到设定的重复次数了"，
    阈值取默认的 `1.0` 就是精确的判定边界（`Router` 用 `>=` 比较）。
    中间值（0.33 / 0.67）是"正在逼近"，扫阈值时有用。
    """

    name: str = "stuck"
    threshold: float = 1.0

    min_repeats: int = 3
    """重复几次算卡住。**这是唯一需要调的超参**，用真实轨迹扫出来。

    直觉上该是 2，实测不是——见 `scripts/eval_repeat_detector.py` 的扫描结果。
    """

    mode: str = "streak"
    """`"streak"` 数末尾连击，`"count"` 数窗口内出现次数。

    `count` 能抓交替式打转，但也更容易被正常的"来回切换"误触发。
    """

    window: int = 6

    ignore: tuple[str, ...] = ()
    """不计入重复的动作名。默认空——**包括 `wait`**。

    一开始想把 `wait` 排除掉（"等待不算卡住"），实测发现反了：
    反复 `wait` 恰恰是最典型的卡住形态之一（模型不知道该干嘛，就一直等），
    而且它在成功轨迹里出现得并不比失败轨迹多。排除掉反而掉召回。
    """

    def score(self, task: str | None, steps: Sequence[Step]) -> float:
        """`task` 参数收下但不用——**卡住是局部性质，不需要知道目标**。

        这个签名和 `Monitor` 协议一致。里程碑监控器必须拿到 task（"有没有
        进展"离开目标就无从谈起），卡住监控器则不该看它——真去看了，
        就变成在偷偷做里程碑的判断了。
        """
        keys = [k for k in _keys(steps) if not any(k.startswith(g) for g in self.ignore)]
        if not keys:
            return 0.0

        if self.mode == "count":
            recent = keys[-self.window :] if self.window > 0 else keys
            n = max((recent.count(k) for k in set(recent)), default=0)
        else:
            n = 0
            for k in reversed(keys):
                if k == keys[-1]:
                    n += 1
                else:
                    break

        return min(1.0, n / self.min_repeats) if self.min_repeats > 0 else 0.0
