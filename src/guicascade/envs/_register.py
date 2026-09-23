"""登记环境实现。"""

from ..registry import register

register("env", "scripted", "guicascade.envs.scripted:ScriptedEnvironment")
register("env", "android", "guicascade.envs.android:AndroidEnv")
