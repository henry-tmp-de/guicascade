"""登记动作空间。

刻意放在顶层而不是 `actions.py` 里：`actions.py` 是翻译层的核心逻辑，
被到处 import；把登记代码塞进去会让它多一个对 registry 的依赖。
注册表本来就是"外围装配"的活，单独一个文件更干净。
"""

from .envs.android import android_action_space
from .registry import register

# 这里的"类"其实是个工厂函数——registry 只要求"可调用且返回部件"，
# 函数和类在 Python 里没有区别。
register("space", "android", android_action_space)
