"""登记策略实现。

放在顶层而不是 `policies/` 目录里，是因为 `policies.py` 是个**模块**不是包
——Python 不允许同名模块和包并存，`policies/_register.py` 会被直接遮蔽掉。
（本项目踩过这个坑：注册表里 policy 一直是空的，而错误被 try/except 吞了。）
"""

from .registry import register

register("policy", "single", "guicascade.policies:SingleModelPolicy")
register("policy", "cascade", "guicascade.policies:CascadePolicy")
register("policy", "cascade_router", "guicascade.router:CascadeRouter")
