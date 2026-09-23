"""登记模型实现。延迟导入：只有配置里真的用到才会 import 对应的实现模块。"""

from ..registry import register

register("model", "openai_compat", "guicascade.models.openai_compat:OpenAICompatModel")
register("model", "scripted", "guicascade.models.scripted:ScriptedModel")
register("model", "codeblock", "guicascade.models.scripted:CodeBlockModel")
