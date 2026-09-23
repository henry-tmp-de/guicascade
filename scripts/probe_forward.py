"""第四层诊断：走真实前向路径，看模型到底在响应什么。

前一版诊断犯了个错：手算 mean pooling 之后直接接 `classifier`，
**跳过了 `ModernBertPredictionHead` 这个投影层**。所以那一列的 logits
不是模型的真实输出，不能用来判断。

这一版只用 `model(**inputs).logits`——和官方 `stuck_detector.py` 完全一样的
调用路径。然后拿一批**跨度很大**的输入喂进去，看分数分布。

要回答的问题：模型是真的只会输出 0，还是它对某类输入有反应、
只是我们没喂到那类输入上。
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_inputs() -> dict[str, str]:
    """覆盖面尽量拉开：格式、长度、语言、内容都不同。"""
    gui_stuck = "".join(
        f"Step {i + 1}:\nResponse: I will click the same button again\nAction: click(index=7)\n\n"
        for i in range(6)
    )
    gui_normal = (
        "Step 1:\nResponse: open the settings app\nAction: open_app(app_name='Settings')\n\n"
        "Step 2:\nResponse: scroll down\nAction: scroll(direction='down')\n\n"
        "Step 3:\nResponse: tap Display\nAction: click(index=4)\n\n"
    )
    return {
        "GUI格式-卡住重复": gui_stuck,
        "GUI格式-正常推进": gui_normal,
        "纯英文散文": "The quick brown fox jumps over the lazy dog. " * 20,
        "纯中文": "这是一段与图形界面无关的中文文本，用于测试模型对语言的敏感度。" * 10,
        "空字符串": "",
        "单个词": "stuck",
        "全是重复字符": "AAAAAAAAAA" * 50,
        "超长GUI轨迹": "".join(
            f"Step {i + 1}:\nResponse: Now I will do action number {i}\n"
            f"Action: act_{i}(arg={i})\n\n"
            for i in range(150)
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/stuck-detector")
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()

    # 打印真实的结构层级，确认池化之后到底经过了什么
    print(f"模型: {args.model}")
    print(f"顶层子模块: {[n for n, _ in model.named_children()]}")
    if hasattr(model, "head"):
        print(f"  head        = {model.head}")
    print(f"  classifier  = {model.classifier}")
    print(f"  pooling     = {getattr(model.config, 'classifier_pooling', None)}")
    print()

    header = f"{'输入':<20}{'长度':>6}{'logit[0]':>11}{'logit[1]':>11}{'P(正类)':>12}"
    print(header)
    print("-" * len(header))

    probs = []
    for label, text in build_inputs().items():
        enc = tokenizer(
            text, truncation=True, max_length=args.max_length,
            padding="longest", return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**enc).logits
        p = torch.softmax(logits, dim=-1)[0, 1].item()
        probs.append(p)
        print(f"{label:<20}{enc['input_ids'].shape[1]:>6}"
              f"{logits[0, 0].item():>11.4f}{logits[0, 1].item():>11.4f}{p:>12.6f}")

    spread = max(probs) - min(probs)
    print(f"\n极差 = {spread:.6f}   最大值 = {max(probs):.6f}   最小值 = {min(probs):.6f}")

    # ⚠️ 关键：全局极差大不代表模型可用。如果分数是靠**域外输入**（中文、乱码、
    # 空串）拉开的，而它真正要服务的域内输入（英文 GUI 轨迹）全是同一个值，
    # 那这个模型对本任务是退化的——只是没退化在"全局"这个尺度上而已。
    in_domain = [p for label, p in zip(build_inputs(), probs) if label.startswith("GUI格式")]
    domain_spread = (max(in_domain) - min(in_domain)) if in_domain else 0.0

    print(f"其中【域内输入（英文 GUI 轨迹）】极差 = {domain_spread:.6f}")
    print()
    if domain_spread < 1e-6:
        print("❌ 域内输出恒定 —— 模型对本任务是退化的。")
        print("   它的分数差异全部来自域外输入，那在实际使用中永远不会出现。")
    elif max(in_domain) < 0.1:
        print("⚠️  域内分数整体压在低位 —— 阈值 0.5 永不触发，必须重新校准；")
        print("   且要先确认分数与目标概念真的相关，而不只是长度/分布的副作用。")
    else:
        print("✅ 域内输出有动态范围 —— 模型可用，按实测分布重新校准阈值即可。")


if __name__ == "__main__":
    main()
