"""工具层：界面操作之外的其它能力（搜索、记事、宣告完成……）。

## 为什么工具和 GUI 动作必须分开

工具和 GUI 动作长得一模一样——都是「名字 + 参数」，模型看来没有区别。
但对**主循环**来说它们是两回事：

| | GUI 动作 | 工具 |
|---|---|---|
| 例子 | click / scroll / type | web_search / note |
| 做完之后 | **屏幕变了** | 屏幕没变，只拿到一段文字 |
| 主循环要做的 | 重新截图，拿新观察 | 把结果追加进历史，观察不动 |

这个差别不是美观问题：GUI 动作之后**必须**重新获取观察，否则模型是在
拿旧屏幕做决策；而工具调用之后**不该**重新截图，白花一次环境开销
（在真机上截图 + 无障碍树 dump 是几百毫秒级的事）。

所以 `Action` 上带一个 `kind` 字段把两者分开，Agent 里据此走不同分支。

## 为什么工具集要小

提示词的静态前缀越小，KV Cache 越稳，模型注意力也越集中。工具描述膨胀
会同时吃掉这两样。所以这里只放少数几个真正通用的工具，特定领域的能力
应该做成环境的动作（`ActionSpace`），而不是往全局工具集里堆。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

__all__ = ["Tool", "ToolResult", "Toolkit", "NoteTool", "FinishTool"]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """工具执行的结果。

    工具既能返回裸字符串，也能返回这个对象——后者多一个 `done` 标志，
    让工具可以宣告"任务结束"（`FinishTool` 就是这么用的）。

    `Toolkit` 会把两种返回值统一成这个类型，调用方只面对一种形状。
    """

    output: str
    done: bool = False


@runtime_checkable
class Tool(Protocol):
    """一个界面操作之外的能力。

    实现只需要 `name` / `description` / `parameters` / `run` 四样。
    `parameters` 用 JSON Schema 描述——这是模型侧 tool calling 的通用格式，
    不需要为每种模型再翻译一遍。
    """

    name: str
    """工具名。模型会用它来调用，所以应当是动词开头的短标识符。"""

    description: str
    """给模型看的说明。

    ⚠️ 别省这一段。消融实验显示，**把工具的描述文字删掉、只留函数签名和
    参数定义，工具调用错误率会涨四成以上**——模型会传无效参数或误解参数含义。
    写描述时要说清三件事：干什么、什么时候用、**什么时候不该用**。
    """

    parameters: Mapping[str, Any]
    """参数的 JSON Schema。"""

    def run(self, **kwargs: Any) -> str:
        """执行并返回一段文本结果，这段文本会被追加进对话历史。"""
        ...


@dataclass
class Toolkit:
    """工具集合：负责把工具描述成模型能懂的格式，并按名字派发调用。

    刻意保持成一个薄容器，不做插件发现、不做权限、不做异步——那些等到
    真的需要时再加。现在它只有两个职责：**产出 schema** 和 **派发执行**。
    """

    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> "Toolkit":
        """登记一个工具，返回自身以便链式调用。"""
        self.tools[tool.name] = tool
        return self

    def schemas(self) -> list[dict[str, Any]]:
        """产出 OpenAI 风格的 tools 字段。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": dict(t.parameters),
                },
            }
            for t in self.tools.values()
        ]

    def describe(self) -> str:
        """产出人话版的工具清单，给走文本模式的模型看。"""
        blocks = []
        for t in self.tools.values():
            params = ", ".join(dict(t.parameters).get("properties", {}))
            blocks.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(blocks) if blocks else "(无)"

    def run(self, name: str, args: Mapping[str, Any]) -> ToolResult:
        """按名字执行。未知工具或执行失败都返回可读的错误文本，不抛异常。

        这个选择是刻意的：**模型看到"没有这个工具"会自己改，而抛异常会把
        整个任务打断。** 对 Agent 来说，一个可读的失败远比一次崩溃有价值
        ——GUI 任务动辄几十步，因为一个工具调用写错就丢掉整条轨迹，代价太大。
        """
        if name not in self.tools:
            available = ", ".join(sorted(self.tools)) or "(无)"
            return ToolResult(f"Error: 没有名为 {name!r} 的工具。可用工具：{available}")

        try:
            raw = self.tools[name].run(**args)
        except TypeError as e:
            return ToolResult(f"Error: 工具 {name!r} 的参数不对（{e}）。请检查参数名和必填项。")
        except Exception as e:  # noqa: BLE001 - 工具失败不该炸掉整个 episode
            return ToolResult(f"Error: 工具 {name!r} 执行失败：{type(e).__name__}: {e}")

        # 统一两种返回形状：裸字符串 或 ToolResult
        if isinstance(raw, ToolResult):
            return raw
        return ToolResult(str(raw))


# --------------------------------------------------------------------------
# 两个开箱即用的通用工具
#
# 选这两个不是随手挑的：GUI 任务动辄几十步，模型需要一块"草稿纸"，
# 也需要一个明确的"我干完了"的信号。这两件事每个 GUI 任务都要用。
# --------------------------------------------------------------------------


@dataclass
class NoteTool:
    """工作记忆：把关键信息记下来，后续步骤能看到。

    GUI 任务里模型的上下文会被大量屏幕文本稀释，早先看清的信息（账号、
    金额、临时结论）很容易在十步之后被忘掉。让它主动记一笔，比指望它
    一直记得可靠得多。
    """

    name: str = "note"
    description: str = (
        "把一条关键信息记到工作记忆里，后续步骤可以看到。"
        "适用于从屏幕上读到的、后面还会用到的信息（账号、金额、验证码、"
        "已尝试且失败的路径）。"
        "不要用它记录每一步都看得见的东西——那只是浪费上下文。"
    )
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要记下的内容，一句话说清"},
            },
            "required": ["content"],
        }
    )
    notes: list[str] = field(default_factory=list)

    def run(self, content: str, **_: Any) -> str:
        self.notes.append(content)
        return f"已记录（当前共 {len(self.notes)} 条）：{content}"


@dataclass
class FinishTool:
    """宣告任务结束。

    单独做成工具而不是"模型说完了就算完"：完成与否需要被显式表达，
    才能和"模型不知道下一步该干嘛所以乱说"区分开。主循环据此干净收尾。
    """

    name: str = "finish"
    description: str = (
        "宣告任务已完成，结束本次执行。"
        "只在确实达成目标后调用，并在 summary 里说明凭什么判定成功"
        "（比如「联系人页面上已出现名为张三的条目」）。"
        "NOT 用于放弃任务——做不到时不要调用它。"
    )
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "凭什么判定任务已完成的证据"},
            },
            "required": ["summary"],
        }
    )
    finished: bool = False
    summary: str = ""

    def run(self, summary: str = "", **_: Any) -> "ToolResult":
        self.finished = True
        self.summary = summary
        return ToolResult(f"任务已标记为完成：{summary}", done=True)
