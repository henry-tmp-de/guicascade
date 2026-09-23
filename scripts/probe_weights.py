"""第三层诊断：分类头是不是死的。

padding 已经排除（三种策略结果相同）。权重加载也干净（missing/unexpected
都是空集）。那问题只可能出在两处：

  a) 池化之后拿到的句向量本身没有信息 -> 编码器没学到东西
  b) 句向量有信息，但分类头把它压死了（权重全零 / 偏置饱和）-> 头是死的

区分这两者看一个东西就够了：**分类头权重和偏置的量级**。
如果 weight 的范数接近 0，输出就完全由 bias 决定，与输入无关——
那就不是"模型不会分"，而是"分类头根本没被训练/没被正确加载"。
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/stuck-detector")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()

    print(f"模型: {args.model}\n")

    print("--- 分类头参数 ---")
    w = model.classifier.weight.detach()
    print(f"classifier.weight : shape={tuple(w.shape)}  norm={w.norm().item():.6f}  "
          f"absmax={w.abs().max().item():.6f}  absmean={w.abs().mean().item():.6f}")
    if model.classifier.bias is not None:
        b = model.classifier.bias.detach()
        print(f"classifier.bias   : {b.tolist()}  norm={b.norm().item():.6f}")
    else:
        print("classifier.bias   : None")

    print("\n--- 编码器是否在学习状态（抽查首层与前几层权重范数）---")
    layers = model.model.layers
    for i in [0, 1, len(layers) // 2, len(layers) - 1]:
        blk = layers[i]
        norms = [p.detach().norm().item() for p in blk.parameters()]
        print(f"  layer[{i:2d}] 参数范数总和 = {sum(norms):.3f}")

    print("\n--- 输入变化时，句向量与 logits 到底动没动 ---")

    inputs_text = {
        "极短": "Step 1:\nResponse: click\nAction: click(index=1)\n",
        "卡住重复": "".join(
            f"Step {i+1}:\nResponse: I will click the same button again\nAction: click(index=7)\n\n"
            for i in range(6)
        ),
        "完全无关文本": "The quick brown fox jumps over the lazy dog. " * 40,
        "超长输入(接近上限)": "".join(
            f"Step {i+1}:\nResponse: Now I will perform a completely different action number {i}\n"
            f"Action: do_thing_{i}(param={i})\n\n"
            for i in range(120)
        ),
    }

    print(f"\n{'输入':<18}{'长度':>6}{'logit[0]':>12}{'logit[1]':>12}{'|句向量|':>12}{'P(正类)':>12}")
    print("-" * 72)

    for label, text in inputs_text.items():
        enc = tokenizer(text, truncation=True, max_length=2048, return_tensors="pt")
        with torch.no_grad():
            out = model.model(**enc)                       # 编码器输出
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)   # 手算 mean pooling
            logits = model.classifier(pooled)
            probs = torch.softmax(logits, dim=-1)
        print(f"{label:<18}{enc['input_ids'].shape[1]:>6}"
              f"{logits[0,0].item():>12.4f}{logits[0,1].item():>12.4f}"
              f"{pooled.norm().item():>12.3f}{probs[0,1].item():>12.6f}")

    print("\n--- 判读 ---")
    print("如果 logit[1] 在所有输入下几乎不变 -> 分类头是死的（问题在权重）")
    print("如果 |句向量| 随输入变化很大但 logit 不变 -> 头把信息压掉了")
    print("如果 logit 会变、只是范围窄 -> 模型活着，只是阈值 0.5 定错了，需要重新校准")


if __name__ == "__main__":
    main()
