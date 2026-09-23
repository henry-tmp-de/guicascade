"""重复检测器的测试。

这个监控器的卖点是「纯代码、零依赖、判据可解释」，所以测试要守住三条线：

  1. **归一化真的管用** —— 写法飘了但语义相同的动作要被认成同一个
  2. **两种模式各抓各的** —— 连击抓原地不动，窗口计数抓交替打转
  3. **不引入任何重依赖** —— 它要能在没有 torch 的机器上跑，
     这是它相对 BERT 版本的核心优势，退化了就没意义了
"""

from __future__ import annotations

import sys

import pytest

from guicascade.monitors.repeat import RepeatMonitor, normalize_action, repeat_stats
from guicascade.types import Action, Decision, Observation, Step


def steps(*actions: str) -> list[Step]:
    """把一串动作渲染文本造成 `Step` 列表，只需要监控器用得到的那部分。"""
    out = []
    for i, a in enumerate(actions):
        name, _, rest = a.partition("(")
        args = {}
        if rest:
            body = rest.rstrip(")")
            for part in body.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    args[k.strip()] = v.strip().strip("'\"")
        out.append(Step(
            index=i,
            observation=Observation(text=""),
            decision=Decision(reason="", action=Action(name, args)),
        ))
    return out


# --------------------------------------------------------------------------
# 一、归一化
# --------------------------------------------------------------------------


@pytest.mark.parametrize("a, b", [
    ("open_app(app_name='com.x')", 'open_app(app_name="com.x")'),
    ("open_app(app_name='com.x')", "open_app( app_name = com.x )"),
    ("wait(seconds=2)", "wait(seconds=2.0)"),
    ("click(index=7, y=3)", "click(y=3, index=7)"),
])
def test_normalization_collapses_cosmetic_differences(a: str, b: str) -> None:
    """引号、空格、参数顺序、数字写法不同，但**是同一个动作**。

    实测里这几种写法都真实出现过。不归一化的话，模型明明在原地打转，
    监控器却因为每次多打了个空格而认为它在做不同的事。
    """
    assert normalize_action(a) == normalize_action(b)


def test_normalization_keeps_semantically_different_actions_apart() -> None:
    """**宁可漏认，不可错认。** 点不同元素必须是不同的动作。"""
    assert normalize_action("click(index=7)") != normalize_action("click(index=8)")
    # 少了参数名的位置参数语义是模糊的，不该和具名写法混为一谈
    assert normalize_action("click(7)") != normalize_action("click(index=7)")


def test_normalization_survives_comma_inside_a_string_value() -> None:
    """`input_text(text='a,b')` 里的逗号不是参数分隔符。"""
    a = normalize_action("input_text(text='hello, world')")
    b = normalize_action('input_text(text="hello, world")')
    assert a == b


# --------------------------------------------------------------------------
# 二、两种模式
# --------------------------------------------------------------------------


def test_streak_counts_trailing_repeats() -> None:
    s = steps("click(index=1)", "click(index=7)", "click(index=7)", "click(index=7)")
    assert repeat_stats(s) == (3, 3)


def test_count_mode_catches_alternating_loop() -> None:
    """`click(6) click(7) click(6) click(7) ...` 也是卡住。

    这个模式在自采数据里真实出现过（`collect_small/.../open_clock`）。
    **连击模式对它完全无感**——末尾连击永远只有 2——所以必须留着 count 模式。
    """
    s = steps(*["click(index=6)", "click(index=7)"] * 4)

    assert repeat_stats(s)[0] == 1, "连击：末尾是 click(7)，前面是 click(6)，只有 1"
    # 窗口是 6：最近 6 步是 6,7,6,7,6,7，各出现 3 次
    assert repeat_stats(s)[1] == 3, "窗口计数：最近 6 步里每个动作各出现 3 次"

    mon_streak = RepeatMonitor(mode="streak", min_repeats=3)
    mon_count = RepeatMonitor(mode="count", min_repeats=3)

    assert mon_streak.score(None, s) < mon_streak.threshold, "连击模式对此完全无感"
    assert mon_count.score(None, s) == 1.0, "窗口计数模式抓得住"


def test_wait_repeats_also_count() -> None:
    """反复 `wait` 也算卡住——**这是刻意不排除的**。

    一开始把 wait 拉进黑名单（"等待不算卡住"），实测发现反了：
    反复等待恰恰是"模型不知道该干嘛"最典型的表现，而且它在成功轨迹里
    并不比失败轨迹更常见。
    """
    s = steps(*["wait(seconds=2)"] * 4)
    assert RepeatMonitor(mode="streak", min_repeats=3).score(None, s) == 1.0


def test_ignore_filter_drops_named_actions() -> None:
    s = steps(*["wait(seconds=2)"] * 4)
    mon = RepeatMonitor(mode="streak", min_repeats=3, ignore=("wait",))
    assert mon.score(None, s) == 0.0


# --------------------------------------------------------------------------
# 三、阈值语义与边界
# --------------------------------------------------------------------------


def test_score_reaches_exactly_one_at_the_configured_count() -> None:
    """阈值默认 1.0，`Router` 用 `>=` 比较，所以边界必须精确。

    差一个浮点误差就会变成"配置 3 次、实际 4 次才触发"，
    而且这种错**在日志里看不出来**——只会表现为召回率莫名偏低。
    """
    mon = RepeatMonitor(mode="streak", min_repeats=3)
    assert mon.score(None, steps(*["click(index=1)"] * 2)) == pytest.approx(2 / 3)
    assert mon.score(None, steps(*["click(index=1)"] * 3)) == 1.0
    assert mon.score(None, steps(*["click(index=1)"] * 3)) >= mon.threshold
    assert mon.score(None, steps(*["click(index=1)"] * 2)) < mon.threshold


def test_empty_history_scores_zero() -> None:
    assert RepeatMonitor().score(None, []) == 0.0


def test_score_is_a_probability_like_value() -> None:
    """值域必须落在 [0, 1]，否则 `Decision.signals` 的口径就乱了。"""
    mon = RepeatMonitor(mode="streak", min_repeats=2)
    for n in range(1, 20):
        v = mon.score(None, steps(*["click(index=1)"] * n))
        assert 0.0 <= v <= 1.0


def test_task_argument_is_ignored_by_design() -> None:
    """卡住是**局部性质**，不看目标也该给出同样的分。

    这不是偷懒：真去看了 task，它就在偷偷做里程碑监控器的判断了，
    两个监控器的分工也就没了意义。
    """
    s = steps(*["click(index=1)"] * 3)
    mon = RepeatMonitor(mode="streak", min_repeats=3)
    assert mon.score(None, s) == mon.score("打开设置", s)


# --------------------------------------------------------------------------
# 四、不依赖重型库 —— 这是它相对 BERT 版本的核心优势
# --------------------------------------------------------------------------


def test_does_not_pull_in_torch() -> None:
    """导入这个模块**不能**顺带把 torch 拖进来。

    它要在没有 GPU、没有 transformers 的机器上（比如只跑 adb 的那台）
    当默认监控器用。一旦有人往依赖链里加了个重型 import，这条测试会拦下来。
    """
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
