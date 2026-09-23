"""监控器冒烟测试：加载两个 ModernBERT，验证它们到底能不能用。

这个脚本回答四个问题，**每一个都可能让整个级联机制失效**：

1. `num_labels` 是几？官方 config.json 里**没有这个字段**，加载时靠默认值。
2. 输出的 0/1 哪个是正类？（我们假设 1 = 卡住 / 里程碑，但要验证）
3. 分数**随输入变化吗**？如果不管喂什么都输出同一个值，这个模型就是退化的
   ——它的判断力等于零，阈值怎么调都没用。
4. `classifier_pooling=mean` 有没有生效？

第 3 条是重点。训练日志（`trainer_state.json`）里有一处很可疑：里程碑监控器
最后一轮的 `eval_f1 = 0.0`、`eval_recall = 0.0`、`true_positives = 0`
——**全预测成负类**。如果发布的权重就是那一轮，那它永远不会触发。
本脚本用真实输入实测，不靠日志推断。

用法：
    python scripts/smoke_monitors.py --stuck models/stuck-detector \\
                                     --milestone models/milestone-detector
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 探针：人工构造的输入，每条都有一个"应该"的答案
# --------------------------------------------------------------------------

# 卡住监控器看到的格式（官方 stuck_detector.py 的 format_step_history）
STUCK_PROBES: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "明显卡住-原地点同一个按钮",
        "stuck",
        [("需要点击下一步", "click(index=7)")] * 6,
    ),
    (
        "明显卡住-反复失败后重试",
        "stuck",
        [
            ("应该点这个按钮", "click(index=3)"),
            ("没反应，再试一次", "click(index=3)"),
            ("还是没反应，我再点一次", "click(index=3)"),
            ("可能是加载慢，再等等再点", "click(index=3)"),
            ("继续点击", "click(index=3)"),
            ("点击", "click(index=3)"),
        ],
    ),
    (
        "正常推进-每步都不一样",
        "ok",
        [
            ("打开设置应用", "open_app(app_name='Settings')"),
            ("滚动找到显示选项", "scroll(direction='down')"),
            ("点击显示", "click(index=4)"),
            ("点击亮度", "click(index=2)"),
            ("调整滑块", "swipe(direction='right')"),
            ("返回上一级", "navigate_back()"),
        ],
    ),
    (
        "正常推进-输入并提交",
        "ok",
        [
            ("点击搜索框", "click(index=1)"),
            ("输入关键词", "input_text(text='天气预报')"),
            ("按回车搜索", "keyboard_enter()"),
            ("查看第一条结果", "click(index=5)"),
        ],
    ),
]

# 里程碑监控器看到的格式（官方 milestone_detector.py 的 build_text_for_step）
MILESTONE_PROBES: list[tuple[str, str, str, list[str]]] = [
    (
        "明显里程碑-成功创建了联系人",
        "milestone",
        "Create a new contact named Alice with phone 555-1234",
        [
            "I need to open the Contacts app first. Action: open_app(app_name='Contacts')",
            "The contacts list is showing. I'll tap the add button at the bottom right. "
            "Action: click(index=12)",
            "The new contact form is open. I'll type the name into the first field. "
            "Action: input_text(text='Alice')",
            "Now I'll move to the phone field and enter the number. Action: click(index=3)",
            "I'll type the phone number now. Action: input_text(text='555-1234')",
            "The contact has been saved and now appears in the contacts list with name Alice "
            "and phone 555-1234. The task is complete. Action: finish()",
        ],
    ),
    (
        "非里程碑-中间过程",
        "ok",
        "Create a new contact named Alice with phone 555-1234",
        [
            "I need to open the Contacts app first. Action: open_app(app_name='Contacts')",
            "The contacts list is showing. I'll tap the add button at the bottom right. "
            "Action: click(index=12)",
            "The new contact form is open. I'll type the name into the first field. "
            "Action: input_text(text='Alice')",
        ],
    ),
    (
        "明显里程碑-闹钟已设置",
        "milestone",
        "Set an alarm for 7:30 AM",
        [
            "Opening the Clock app now. Action: open_app(app_name='Clock')",
            "I'm in the clock app. I'll switch to the alarm tab. Action: click(index=2)",
            "The alarm list is empty. I'll tap the add alarm button. Action: click(index=5)",
            "The alarm editor is open showing 00:00. I'll set the hour to 7 by scrolling "
            "the hour wheel. Action: scroll(direction='up')",
            "Now setting the minutes to 30. Action: scroll(direction='up')",
            "The alarm now reads 7:30 AM and I can see it listed in the alarm tab. "
            "Action: click(index=9)",
        ],
    ),
    (
        "非里程碑-乱试一通",
        "ok",
        "Set an alarm for 7:30 AM",
        [
            "Opening the Clock app now. Action: open_app(app_name='Clock')",
            "I don't see the alarm option, let me scroll. Action: scroll(direction='down')",
            "Still nothing obvious. Let me try tapping here. Action: click(index=3)",
            "That opened something unexpected. Going back. Action: navigate_back()",
            "Let me try the menu instead. Action: click(index=1)",
        ],
    ),
]


def fmt_stuck(steps: list[tuple[str, str]], max_steps: int = 6) -> str:
    """官方格式：Step N:\\nResponse: ...\\nAction: ...\\n，取最近 max_steps 步。"""
    recent = steps[-max_steps:]
    offset = len(steps) - len(recent)
    blocks = []
    for i, (resp, act) in enumerate(recent):
        blocks.append(f"Step {offset + i + 1}:\nResponse: {resp}\nAction: {act}\n")
    return "\n".join(blocks)


def fmt_milestone(task: str, steps: list[str]) -> str:
    """官方格式：Task: ...\\n\\nStep N:\\n<response>"""
    parts = [f"Task: {task}\n"]
    for i, resp in enumerate(steps):
        parts.append(f"Step {i + 1}:\n{resp}")
    return "\n".join(parts)


def inspect(name: str, path: str | None, probes, fmt) -> dict:
    """加载一个监控器，跑一遍探针，报告结果。"""
    print(f"\n{'=' * 72}")
    print(f"  {name}")
    print(f"{'=' * 72}")

    if path is None or not Path(path).exists():
        print(f"  [跳过] 路径不存在：{path}")
        return {"loaded": False, "path": path}

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()

    n_labels = getattr(model.config, "num_labels", None)
    id2label = getattr(model.config, "id2label", None)
    pooling = getattr(model.config, "classifier_pooling", None)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  num_labels         : {n_labels}")
    print(f"  id2label           : {id2label}")
    print(f"  classifier_pooling : {pooling}")
    print(f"  参数量             : {n_params:,}")
    print(f"  max_position_emb   : {getattr(model.config, 'max_position_embeddings', None)}")

    scores: list[float] = []
    print(f"\n  {'探针':<34} {'期望':<10} {'P(正类)':>10}  判定")
    print(f"  {'-' * 66}")

    for label, expect, *rest in probes:
        text = fmt(*rest) if len(rest) > 1 else fmt(rest[0])
        inputs = tok(
            text,
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        p1 = probs[0, 1].item()
        scores.append(p1)

        hit = (p1 >= 0.5) == (expect in ("stuck", "milestone"))
        mark = "✓" if hit else "✗"
        print(f"  {label:<34} {expect:<10} {p1:>10.4f}  {mark}")

    spread = max(scores) - min(scores)
    degenerate = spread < 1e-4
    print(f"\n  分数极差: {spread:.6f}  ->  " + ("⚠️ 退化！对任何输入都给同一个值，判断力为零"
                                                if degenerate else "✅ 分数随输入变化，模型有判别力"))

    return {
        "loaded": True,
        "path": path,
        "num_labels": n_labels,
        "id2label": str(id2label),
        "classifier_pooling": pooling,
        "params": n_params,
        "scores": scores,
        "spread": spread,
        "degenerate": degenerate,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stuck", default="models/stuck-detector")
    ap.add_argument("--milestone", default="models/milestone-detector")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    results = {
        "stuck": inspect("卡住监控器", args.stuck, STUCK_PROBES, fmt_stuck),
        "milestone": inspect("里程碑监控器", args.milestone, MILESTONE_PROBES, fmt_milestone),
    }

    print(f"\n{'=' * 72}")
    print("  结论")
    print(f"{'=' * 72}")
    for k, r in results.items():
        if not r.get("loaded"):
            print(f"  {k:10s}: 未加载")
        elif r["degenerate"]:
            print(f"  {k:10s}: ⚠️ 退化——不同输入给出相同分数，阈值无法生效")
        else:
            print(f"  {k:10s}: ✅ 可用（极差 {r['spread']:.4f}）")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  结果已写入 {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
