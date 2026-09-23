"""注册表测试 —— "换部件只改 YAML"这个承诺是不是真的。

如果注册表坏了，整个项目的可插拔性就是句空话：配置写得再漂亮，
最后还是要改 Python。所以这里逐条验证。
"""

from __future__ import annotations

import pytest

from guicascade.registry import available, build, get_class, register


def test_builtins_are_registered() -> None:
    reg = available()
    assert "scripted" in reg["model"]
    assert "scripted" in reg["env"]
    assert "cascade" in reg["policy"]
    assert "prompt" in reg["monitor"]


def test_build_from_string() -> None:
    env = build("env", "scripted")
    assert env.name == "scripted"


def test_build_from_dict_passes_kwargs() -> None:
    env = build("env", {"type": "scripted", "screens": ["甲", "乙"], "done_at": 1})
    assert env.reset("t").text == "甲"


def test_missing_type_raises() -> None:
    with pytest.raises(ValueError, match="type"):
        build("env", {"screens": ["x"]})


def test_unknown_component_lists_alternatives() -> None:
    with pytest.raises(KeyError) as exc:
        get_class("env", "nonexistent")
    assert "scripted" in str(exc.value)   # 报错要给出可选项


def test_nested_kind_is_inferred_from_field_name() -> None:
    """嵌套部件的类别由字段名推断（_FIELD_KINDS），YAML 里不用手写。"""
    policy = build("policy", {
        "type": "cascade",
        "small": {
            "type": "single", "label": "small",
            "action_space": {"type": "android"},
            "model": {"type": "scripted", "name": "m1", "script": ["Action: click(index=1)"]},
        },
        "large": {
            "type": "single", "label": "large",
            "action_space": {"type": "android"},
            "model": {"type": "scripted", "name": "m2", "script": ["Action: click(index=2)"]},
        },
        "router": {"type": "cascade_router", "theta_stuck": 0.9},
    })
    assert policy.small.label == "small"
    assert policy.large.label == "large"
    assert policy.router.theta_stuck == 0.9
    # action_space 由字段名推断成 space 类别、再由 type 选出具体实现
    assert policy.small.action_space.get("click") is not None


def test_action_space_is_buildable() -> None:
    space = build("space", "android")
    assert space.get("click") is not None
    assert space.get("input_text") is not None


def test_register_custom_component() -> None:
    """第三方不需要改本框架的任何代码，注册一下就能用。"""

    class MyMonitor:
        name = "mine"
        threshold = 0.5

        def score(self, task, steps):
            return 0.42

    register("monitor", "mine", MyMonitor)
    got = build("monitor", "mine")
    assert got.score(None, []) == 0.42


def test_config_file_loads_end_to_end() -> None:
    """真实 YAML 配置能构造出完整策略——这才是"改 YAML 就跑"的最终验证。"""
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[1] / "configs" / "single_small.yaml"
    if not path.exists():
        pytest.skip("配置文件不存在")

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    policy = build("policy", cfg["policy"])
    assert policy.label == "small"
    assert policy.action_space is not None
