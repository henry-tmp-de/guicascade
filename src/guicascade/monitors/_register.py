"""登记监控器实现。"""

from ..registry import register

register("monitor", "bert", "guicascade.monitors.bert:BertMonitor")
register("monitor", "prompt", "guicascade.monitors.prompt:PromptMonitor")
register("monitor", "scripted", "guicascade.monitors.scripted:ScriptedMonitor")
register("monitor", "repeat", "guicascade.monitors.repeat:RepeatMonitor")
