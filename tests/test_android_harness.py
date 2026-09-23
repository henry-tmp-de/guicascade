"""安卓 harness 的**索引契约**测试。

这是整个项目里最该被测试钉死的地方，因为它出过一次代价极大的 bug：

    渲染给模型的列表**过滤掉**了不可交互的节点、序号重新编，
    而点击解析却按**未过滤**的原始树取坐标。于是只要前面被过滤掉一个
    节点，后面**所有序号全部错位**——实测在桌面这一屏上，
    前 12 个序号里有 12 个指向完全不同的元素。

表现是模型"点了没反应、在原地打转"，看起来像它笨。**实际是框架在骗它。**
这一类 bug 的共同特征是：代码读起来完全正确，只有跑起来才发现不对。
所以必须用测试把契约固定下来，而不是靠 review。

契约只有一条：**模型看到的 [N]，点下去必须落在第 N 个元素上。**

测试全部走固定 XML，不碰设备——毕竟这个 bug 的性质是纯逻辑的，
能不能连真机跟它没关系。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from guicascade.envs.android import (
    AndroidEnv,
    _parse_ui_xml,
    actionable_elements,
    render_elements,
)

# 一棵刻意构造的树，覆盖三种会引发错位的情况：
#   1. 纯布局容器（无文字、不可点）—— 会被过滤，是错位的来源
#   2. 零尺寸占位节点           —— 会被过滤
#   3. 可点击容器包着不可点击的文字 —— 点击要往上找祖先
_FIXTURE = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<hierarchy rotation="0">
  <node class="android.widget.FrameLayout" bounds="[0,0][1080,2400]" clickable="false">
    <node class="android.widget.LinearLayout" bounds="[0,0][1080,2400]" clickable="false">
      <node class="android.widget.TextView" text="标题" bounds="[0,100][1080,200]" clickable="false"/>
      <node class="android.view.View" text="" bounds="[0,0][0,0]" clickable="false"/>
      <node class="android.widget.Button" text="确定" bounds="[800,2000][1000,2100]" clickable="true"/>
      <node class="androidx.appcompat.widget.AppCompatEditText" text="" bounds="[50,500][1000,600]" clickable="true"/>
    </node>
    <node class="android.widget.LinearLayout" bounds="[0,300][1080,400]" clickable="true">
      <node class="android.widget.TextView" text="行标题" bounds="[20,310][400,390]" clickable="false"/>
    </node>
  </node>
</hierarchy>
"""


@pytest.fixture
def tree() -> list[dict]:
    return _parse_ui_xml(_FIXTURE)


# --------------------------------------------------------------------------
# 一、索引契约（这个文件存在的理由）
# --------------------------------------------------------------------------


def test_rendered_index_matches_actionable_index(tree: list[dict]) -> None:
    """渲染出来的 [N]，必须就是 actionable[N]——不能各编各的号。"""
    shown = actionable_elements(tree, limit=60)
    lines = render_elements(shown).splitlines()

    assert len(lines) == len(shown)
    for i, line in enumerate(lines):
        assert line.startswith(f"[{i}] "), f"第 {i} 行编号不对：{line!r}"
        # 渲染里出现的标签必须来自同一个元素
        label = shown[i]["text"] or shown[i]["desc"]
        if label:
            assert repr(label) in line


def test_actionable_indices_differ_from_raw_indices(tree: list[dict]) -> None:
    """**证明这棵树确实能触发错位** —— 否则上面的测试是空转的。

    如果过滤前后下标恰好一样，索引 bug 就测不出来。这里刻意让第一层
    容器被过滤掉，好让 raw[0] 和 actionable[0] 是**不同的元素**。
    """
    shown = actionable_elements(tree, limit=60)

    assert tree[0]["class"] == "FrameLayout"        # 原始树第 0 个：被过滤掉的容器
    assert shown[0]["class"] != "FrameLayout"       # 模型看到的第 0 个：不是它
    assert shown[0]["text"] == "标题"

    # 这正是那个 bug 的形状：拿 shown 的下标去索引 raw，会拿到别的东西
    assert tree[0] is not shown[0]


def test_click_point_lands_on_the_element_the_model_saw(tree: list[dict]) -> None:
    """**核心断言**：每个元素的落点，必须落在它自己或它的可点击祖先里。

    「或它的可点击祖先」是刻意的设计而不是放水：安卓界面里
    `<可点击的行><不可点击的文字>` 极为常见，真正响应点击的是外层容器。
    见 `_resolve_click` 的说明。
    """
    shown = actionable_elements(tree, limit=60)

    for i, e in enumerate(shown):
        x, y = e["_click_xy"]
        x1, y1, x2, y2 = e["bounds"]
        in_self = x1 <= x <= x2 and y1 <= y <= y2

        in_anc = False
        for a in e["_ancestors"]:
            anc = tree[a]
            if not anc["clickable"]:
                continue
            ax1, ay1, ax2, ay2 = anc["bounds"]
            if ax1 <= x <= ax2 and ay1 <= y <= ay2:
                in_anc = True
                break

        assert in_self or in_anc, (
            f"第 {i} 个元素 {e['text']!r} bounds={e['bounds']} "
            f"的落点 ({x},{y}) 跑到了它自己外面"
        )


def test_click_falls_back_to_nearest_clickable_ancestor(tree: list[dict]) -> None:
    """不可点击的文字，落点要自动上浮到包着它的可点击容器。

    "行标题"自己不响应点击，但外层那个 `clickable=true` 的容器响应。
    按文字自己的中心点不一定生效，按容器中心点才稳。
    """
    shown = actionable_elements(tree, limit=60)
    row = next(e for e in shown if e["text"] == "行标题")

    assert row["clickable"] is False, "构造前提：这个文字本身不可点击"
    assert row["_click_xy"] == (540, 350), "应当落在可点击容器 [0,300][1080,400] 的中心"


def test_click_uses_own_center_when_already_clickable(tree: list[dict]) -> None:
    """元素自己可点击就用自己，不要多此一举往上找。"""
    shown = actionable_elements(tree, limit=60)
    btn = next(e for e in shown if e["text"] == "确定")
    assert btn["clickable"] is True
    assert btn["_click_xy"] == (900, 2050)


# --------------------------------------------------------------------------
# 二、过滤器本身
# --------------------------------------------------------------------------


def test_layout_containers_and_zero_size_nodes_are_filtered(tree: list[dict]) -> None:
    """纯布局容器和零尺寸节点不该占序号。"""
    shown = actionable_elements(tree, limit=60)
    classes = [e["class"] for e in shown]

    assert "FrameLayout" not in classes, "无文字不可点的容器应被过滤"
    assert "View" not in classes, "零尺寸占位节点应被过滤"

    labels = [e["text"] for e in shown if e["text"]]
    assert labels == ["标题", "确定", "行标题"]


def test_limit_is_enforced_and_truncates_from_the_tail(tree: list[dict]) -> None:
    """`limit` 截断是从尾部截的——保留的一定是**前** N 个。

    这条要紧：如果截断顺序和渲染顺序不一致，序号又会错位。
    """
    full = actionable_elements(tree, limit=60)
    capped = actionable_elements(tree, limit=2)

    assert len(capped) == 2
    assert [e["text"] for e in capped] == [e["text"] for e in full[:2]]
    for i in range(2):
        assert capped[i]["_click_xy"] == full[i]["_click_xy"]


def test_renders_placeholder_when_nothing_is_actionable() -> None:
    assert render_elements([]) == "(屏幕上没有可交互元素)"


# --------------------------------------------------------------------------
# 三、可编辑判定 —— 别只认 android.widget.EditText
# --------------------------------------------------------------------------


def test_editable_detection_covers_common_subclasses(tree: list[dict]) -> None:
    """搜索框基本都是 `AutoCompleteTextView` / `AppCompatEditText`。

    早先用 `class.endswith("EditText")`，遇到 `AppCompatEditText` 还能蒙对，
    遇到 `AutoCompleteTextView`（搜索框的正主）就漏了——模型看不见输入框，
    只能干瞪眼。所以改成子串匹配，**宁可把不可输入的认成可输入**。
    """
    shown = actionable_elements(tree, limit=60)
    box = next(e for e in shown if e["class"] == "AppCompatEditText")
    assert box["editable"] is True


def test_parse_keeps_ancestor_chain(tree: list[dict]) -> None:
    """祖先链要由近及远，`_resolve_click` 靠它找最近的可点击容器。"""
    row = next(e for e in tree if e["text"] == "行标题")
    chain = [tree[a]["class"] for a in row["_ancestors"]]

    assert chain[0] == "LinearLayout", "第一个祖先应当是直接父节点"
    assert "FrameLayout" in chain


def test_parse_survives_garbage_prefix() -> None:
    """`uiautomator dump` 会在 XML 前混一行提示（官方连 hierarchy 都拼错了）。"""
    noisy = "UI hierchary dumped to: /sdcard/x.xml\n" + _FIXTURE
    assert len(_parse_ui_xml(noisy)) == len(_parse_ui_xml(_FIXTURE))


def test_parse_returns_empty_on_broken_xml() -> None:
    assert _parse_ui_xml("not xml at all") == []
    assert _parse_ui_xml("<hierarchy><node bounds='[0,0][1,1]'") == []


# --------------------------------------------------------------------------
# 四、应用名解析 —— 对照表必须在环境里，不在提示词里
# --------------------------------------------------------------------------


def _env(apps: dict[str, str] | None = None) -> AndroidEnv:
    # 不传 adb：这个测试不碰设备，只测纯逻辑
    return AndroidEnv(adb="adb", apps=apps or {})


def test_resolve_app_by_display_name() -> None:
    env = _env({"Clock": "com.google.android.deskclock", "时钟": "com.google.android.deskclock"})
    assert env.resolve_app("Clock") == "com.google.android.deskclock"
    assert env.resolve_app("时钟") == "com.google.android.deskclock"


def test_resolve_app_is_case_insensitive() -> None:
    env = _env({"Clock": "com.google.android.deskclock"})
    assert env.resolve_app("clock") == env.resolve_app("CLOCK") == "com.google.android.deskclock"


def test_resolve_app_passes_through_a_package_name() -> None:
    """表是空的也要能用——直接给包名是永远支持的退路。"""
    env = _env({})
    assert env.resolve_app("com.android.settings") == "com.android.settings"


def test_unknown_app_error_lists_what_is_available() -> None:
    """报错要说清有哪些可选，这句话会回灌给模型让它自己改。

    只报个 `KeyError` 的话模型只能瞎猜。
    """
    env = _env({"Clock": "x", "Settings": "y"})
    with pytest.raises(ValueError) as e:
        env.resolve_app("计算器")
    assert "Clock" in str(e.value) and "Settings" in str(e.value)


def test_empty_app_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        _env({"Clock": "x"}).resolve_app("")


# --------------------------------------------------------------------------
# 五、提示词里不许出现包名
# --------------------------------------------------------------------------


def test_action_space_never_leaks_package_names() -> None:
    """**这条是防回归的。**

    早先 `open_app` 的参数说明里写了一张"中文名=包名"的对照表，模型于是
    完全不看屏幕，拿任务里的词直接查表一步到位——benchmark 退化成查表题，
    整轮三臂对照的数据因此作废。

    以后谁再想把包名写回去方便一下，这条会拦住他。
    """
    from guicascade.envs.android import android_action_space

    spec = android_action_space().get("open_app")
    blob = f"{spec.description} {spec.parameters}"

    assert "com." not in blob, "动作描述里不许出现包名——那等于把答案给模型"
    assert "=" not in str(spec.parameters.get("properties", {}).get("app_name", {})
                          .get("description", "")), "参数说明里也不许有对照表"
