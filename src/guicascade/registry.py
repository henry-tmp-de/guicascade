"""注册表：让「换部件」变成改 YAML，而不是改代码。

前四个模块（models / envs / monitors / policies）用 Protocol 定义了
「一个部件该长什么样」，但没说「它怎么被选中」。没有这一层，想换个监控器
还得去改 Python——那就只是"接口好看"，不叫可插拔。

这一层补上后半句：**名字 → 类** 的映射，加上 **配置字典 → 实例** 的构造。

设计上刻意做成「字符串 → import 路径」而不是「字符串 → 类对象」，
和 mini-swe-agent 的做法一致：这样注册表本身不 import 任何实现，
装不装 torch、装不装 playwright 都不影响框架启动——用到谁才 import 谁。
"""

from __future__ import annotations

import copy
import importlib
from typing import Any, Callable, TypeVar

__all__ = ["register", "build", "get_class", "available"]

T = TypeVar("T")

# kind -> {name -> "module.path:ClassName"}
_REGISTRY: dict[str, dict[str, str]] = {
    "model": {},
    "env": {},
    "monitor": {},
    "policy": {},
    "space": {},
    "tool": {},
}

_BUILTINS_LOADED = False


def register(kind: str, name: str, target: str | type) -> None:
    """登记一个部件。

    Args:
        kind: 部件类别，见 `_REGISTRY` 的键。
        name: 配置里写的名字。
        target: "module.path:ClassName" 字符串，或直接给类对象
            （给类对象时会自动转成 import 路径，保持延迟导入的好处）。

    Example:
        >>> register("monitor", "bert", "guicascade.monitors.bert:BertMonitor")
    """
    if kind not in _REGISTRY:
        raise KeyError(f"未知的部件类别 {kind!r}，可选：{sorted(_REGISTRY)}")

    if isinstance(target, str):
        _REGISTRY[kind][name] = target
        return

    qualname = getattr(target, "__qualname__", None) or target.__name__
    if "<locals>" in qualname:
        # 在函数体里定义的类没法按 import 路径找回——直接存对象。
        # 这条分支主要给测试和临时扩展用；生产代码里的部件都应该是模块级的。
        _REGISTRY[kind][name] = target
        return

    # 其余一律存 import 路径：注册表本身不 import 任何实现，
    # 装不装 torch、装不装 playwright 都不影响框架启动。
    _REGISTRY[kind][name] = f"{target.__module__}:{qualname}"


def get_class(kind: str, name: str) -> type:
    """按名字取回类对象，需要时才 import。"""
    _load_builtins()
    if kind not in _REGISTRY:
        raise KeyError(f"未知的部件类别 {kind!r}，可选：{sorted(_REGISTRY)}")
    if name not in _REGISTRY[kind]:
        raise KeyError(f"{kind} 里没有叫 {name!r} 的部件，可选：{sorted(_REGISTRY[kind])}")

    target = _REGISTRY[kind][name]
    if not isinstance(target, str):      # 局部定义的类，直接就是对象
        return target

    module_path, _, class_name = target.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build(kind: str, spec: Any) -> Any:
    """从配置构造一个部件。

    spec 可以是：
      - 字符串：直接当作部件名，无参数构造
      - 字典：`{"type": "bert", ...其余作为构造参数}`

    Example:
        >>> build("monitor", {"type": "bert", "model_path": "models/stuck-detector"})
    """
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, dict):
        raise TypeError(f"无法从 {type(spec).__name__} 构造部件，需要字符串或字典")

    spec = copy.deepcopy(spec)
    name = spec.pop("type", None)
    if name is None:
        raise ValueError(f"配置里缺少 'type' 字段：{spec!r}")

    # 允许嵌套构造：写 {"type": "cascade", "small": {...}} 时递归展开。
    # 子部件的类别由字段名推断（见 _FIELD_KINDS），所以 YAML 里不用标注。
    kwargs = {k: _maybe_build(v, field=k) for k, v in spec.items()}
    return get_class(kind, name)(**kwargs)


def available(kind: str | None = None) -> dict[str, list[str]]:
    """列出已登记的部件名，用于配置报错时给出提示。"""
    _load_builtins()
    if kind is not None:
        return {kind: sorted(_REGISTRY.get(kind, {}))}
    return {k: sorted(v) for k, v in _REGISTRY.items()}


# 字段名 -> 部件类别。
#
# 有了这张表，YAML 里嵌套部件就不用再标注自己是什么类别了：
#
#     policy:
#       type: cascade
#       small: {type: single, model: {...}}    # 看到 small 就知道是 policy
#       router:
#         stuck: {type: bert, ...}             # 看到 stuck 就知道是 monitor
#
# 字段名本身就携带了语义，再让人手写一遍类别是多余的。
_FIELD_KINDS: dict[str, str] = {
    "model": "model",
    "small": "policy",
    "large": "policy",
    "policy": "policy",
    "router": "policy",
    "env": "env",
    "environment": "env",
    "stuck": "monitor",
    "milestone": "monitor",
    "monitor": "monitor",
    "verifier": "model",
    "action_space": "space",
    "space": "space",
    "toolkit": "tool",
}


def _maybe_build(value: Any, *, field: str = "") -> Any:
    """遇到带 `type` 的字典就递归构造，类别由字段名推断。

    推断不出来时不构造、原样返回——这允许把普通配置字典（比如模型的
    `model_kwargs`）直接透传给构造函数，不会被误当成部件。
    """
    if isinstance(value, dict) and "type" in value:
        kind = _FIELD_KINDS.get(field)
        if kind is not None:
            return build(kind, value)
        return {k: _maybe_build(v, field=k) for k, v in value.items()}

    if isinstance(value, list):
        return [_maybe_build(v, field=field) for v in value]
    if isinstance(value, dict):
        return {k: _maybe_build(v, field=k) for k, v in value.items()}
    return value


def _load_builtins() -> None:
    """把内置部件登记进来。只做一次。

    刻意 import 的是各包的 `_register` 轻量模块，而不是实现模块本身——
    这样不会因为"没装 torch"就打不开框架。
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    for module_name in (
        "guicascade.models._register",
        "guicascade.envs._register",
        "guicascade.monitors._register",
        "guicascade.policies_register",
        "guicascade.actions_register",
    ):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            # 区分两种情况：登记模块本身不存在（正常，这个类别还没实现），
            # 和登记模块存在但它依赖的东西缺失（真错误，必须报）。
            # 一视同仁地吞掉会让"注册表莫名其妙是空的"变成极难查的问题——
            # 这个坑本项目的 policies 就踩过一次。
            if e.name == module_name:
                continue
            raise
