"""诊断监控器为什么打不出有区分度的分。

背景：冒烟测试发现两个官方监控器在阈值 0.5 下都永远不触发——
卡住监控器对任何输入都输出 0.0000，里程碑监控器全挤在 0.04 以下。

怀疑根因是推理时的 padding 策略：官方代码用 `padding="max_length"`
把每条输入都补到 2048 个 token，而模型的池化方式是 `mean`。
**如果 mean pooling 没有屏蔽掉 pad，那么短输入的均值会被 pad 淹没**，
输出就会趋近于常数、与输入无关。

本脚本对比三种 padding 策略，验证这个假设。若假设成立，
修法就是改用动态 padding——这是复现这类工作时很容易踩空的一环：
**模型权重是对的，错的是喂给它的方式。**
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

STUCK_INPUT = "".join(
    f"Step {i + 1}:\nResponse: I will click the same button again\nAction: click(index=7)\n\n"
    for i in range(6)
)

NORMAL_INPUT = (
    "Step 1:\nResponse: open the settings app\nAction: open_app(app_name='Settings')\n\n"
    "Step 2:\nResponse: scroll down to find display options\nAction: scroll(direction='down')\n\n"
    "Step 3:\nResponse: tap on Display\nAction: click(index=4)\n\n"
    "Step 4:\nResponse: tap on Brightness\nAction: click(index=2)\n\n"
)

STRATEGIES = [
    ("max_length (官方做法)", {"padding": "max_length"}),
    ("longest (动态)", {"padding": "longest"}),
    ("不 padding", {}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/stuck-detector")
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()

    print(f"模型: {args.model}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, load_info = AutoModelForSequenceClassification.from_pretrained(
            args.model, output_loading_info=True
        )
    model.eval()

    print("--- 权重加载情况 ---")
    print(f"missing_keys    : {load_info.get('missing_keys')}")
    print(f"unexpected_keys : {load_info.get('unexpected_keys')}")
    print(f"mismatched_keys : {load_info.get('mismatched_keys')}")
    if caught:
        print(f"警告            : {[str(w.message)[:140] for w in caught][:3]}")

    print("\n--- 模型结构 ---")
    print(f"classifier_pooling : {getattr(model.config, 'classifier_pooling', None)}")
    print(f"pad_token          : {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"classifier head    : {type(model.classifier).__name__}")

    def score(text: str, **tok_kwargs) -> tuple[float, int]:
        inputs = tokenizer(
            text, truncation=True, max_length=args.max_length, return_tensors="pt", **tok_kwargs
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.softmax(logits, dim=-1)[0, 1].item(), inputs["input_ids"].shape[1]

    print("\n--- 三种 padding 策略对比 ---")
    header = f"{'策略':<22}{'序列长度':>9}{'卡住输入':>12}{'正常输入':>12}{'极差':>11}"
    print(header)
    print("-" * len(header))

    best_spread = -1.0
    best_name = ""
    for label, kwargs in STRATEGIES:
        p_stuck, length = score(STUCK_INPUT, **kwargs)
        p_normal, _ = score(NORMAL_INPUT, **kwargs)
        spread = abs(p_stuck - p_normal)
        if spread > best_spread:
            best_spread, best_name = spread, label
        print(f"{label:<22}{length:>9}{p_stuck:>12.6f}{p_normal:>12.6f}{spread:>11.6f}")

    print("\n--- 结论 ---")
    if best_spread > 0.1:
        print(f"✅ padding 策略影响巨大，'{best_name}' 能救回来（极差 {best_spread:.4f}）")
        print("   -> 官方 `padding='max_length'` 是问题所在，应改用动态 padding")
    elif best_spread > 1e-4:
        print(f"⚠️  padding 有影响但不足以解释（最好 {best_spread:.6f}）")
        print("   -> 还需要查别的原因（权重本身、池化层实现）")
    else:
        print("❌ 换 padding 策略也救不回来，问题在权重本身而非推理方式")

    if Path(args.model).exists():
        print(f"\n提示：卡住检测的实际触发阈值应据实测分数域重新校准，不要用 0.5。")


if __name__ == "__main__":
    main()
