"""沿着真实的"卡住片段"逐步看监控器的分数——这次判据放宽到会逮住它们。

## 上一版为什么漏了

上一版要求「窗口末尾连续 >= 3 步完全相同」，而且窗口只有 6 步。
真实数据里的重复是这样的：

    evocua-32b/1-Task :  click(960, 314) 连续 7 次，但被 hotkey 打断后又是 2 次
    evocua-32b/2-Task :  press('enter') 共 20 次，但中间夹着 typewrite
    qwen3-vl/1-Task   :  typewrite('Out of the Silent Planet') 连续 3 次

连续 3 步 + 6 步窗口这个组合，只会在最长的那个 run 的正中间命中一两次。
所以「卡住窗口 n=4」是判据太严，不是数据里没有。

## 这一版怎么做

1. 放宽到「窗口末尾连续 >= 2 步相同动作」，把三类都逮住
2. **不只看均值**，而是沿着每个卡住片段**逐窗口打印分数**——
   看分数在"开始重复"那一刻有没有抬头、在"重复持续"时有没有继续涨。
   这比一个 AUC 数字更能说明它到底抓没抓到。
3. 再算一个 AUC：末尾在重复中的窗口 vs 同一条轨迹里不重复的窗口。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WINDOW = 6


def read_traj(path: Path) -> list[dict]:
    steps = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))
    steps.sort(key=lambda d: int(d.get("step_num", 0)))
    return steps


def build_window(steps: list[dict], end_idx: int, window: int = WINDOW) -> str:
    """照搬官方 _build_step_text_window。"""
    start = max(0, end_idx - (window - 1))
    chunks = [
        f"Step {int(steps[i].get('step_num', i + 1))}:\n"
        f"Response: {steps[i].get('response', '')}\n"
        f"Action: {steps[i].get('action', '')}\n"
        for i in range(start, end_idx + 1)
    ]
    return "\n".join(chunks).strip() + "\n"


def repeat_runs(actions: list[str], min_len: int = 2) -> list[tuple[int, int, str]]:
    """找出所有「连续相同动作 >= min_len 次」的片段。

    Returns: [(起(0-based), 止(0-based, 含), 动作), ...]
    """
    runs = []
    i = 0
    while i < len(actions):
        j = i
        while j + 1 < len(actions) and actions[j + 1] == actions[i]:
            j += 1
        if j - i + 1 >= min_len:
            runs.append((i, j, actions[i]))
        i = j + 1
    return runs


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/evocua_feedback")
    ap.add_argument("--min-run", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("models/stuck-detector")
    mdl = AutoModelForSequenceClassification.from_pretrained("models/stuck-detector")
    mdl = mdl.to(args.device).eval()

    def score(text: str) -> float:
        enc = tok(text, truncation=True, max_length=2048,
                  padding="max_length", return_tensors="pt").to(args.device)
        with torch.no_grad():
            return torch.softmax(mdl(**enc).logits, -1)[0, 1].item()

    all_pos, all_neg = [], []

    for tp in sorted(Path(args.root).rglob("traj.jsonl")):
        steps = read_traj(tp)
        acts = [s.get("action", "") for s in steps]
        runs = repeat_runs(acts, args.min_run)
        if not runs:
            continue
        rel = tp.relative_to(args.root).parent
        result = (tp.parent / "result.txt").read_text().strip()

        print(f"\n{'=' * 88}")
        print(f"▌ {rel}   {len(steps)} 步   result={result}")
        print(f"{'=' * 88}")
        print(f"  {'步':>4}  {'卡住P':>12}  {'里程碑P':>11}  {'状态':<14} 动作")
        print("  " + "-" * 82)

        stuck_idx = set()
        for a, b, _ in runs:
            stuck_idx.update(range(a, b + 1))

        for i in range(len(steps)):
            if i < WINDOW - 1:
                continue
            text = build_window(steps, i)
            p = score(text)
            mile_tok = AutoTokenizer.from_pretrained("models/milestone-detector")
            # 里程碑只算一次太慢，这里只对卡住监控器逐窗口打分
            state = ""
            if i in stuck_idx:
                state = f"重复中({acts[i][:18]})"
                all_pos.append(p)
            else:
                all_neg.append(p)
            print(f"  {i + 1:>4}  {p:>12.6f}  {'':>11}  {state:<14} {acts[i][:44]}")

        for a, b, act in runs:
            print(f"     ↳ 重复片段：第 {a + 1}~{b + 1} 步  ×{b - a + 1}  {act[:50]}")

    print()
    print("=" * 88)
    print("  汇总")
    print("=" * 88)
    if all_pos and all_neg:
        print(f"  重复中的窗口  n={len(all_pos):<4} 均分 = {sum(all_pos)/len(all_pos):.6f}  "
              f"最大 = {max(all_pos):.6f}")
        print(f"  不重复的窗口  n={len(all_neg):<4} 均分 = {sum(all_neg)/len(all_neg):.6f}  "
              f"最大 = {max(all_neg):.6f}")
        a = auc(all_pos, all_neg)
        print(f"  ⭐ AUC = {a:.4f}")
        if a >= 0.8:
            print("  ✅ 卡住监控器在**真实卡住片段**上是能分辨的！之前测不出是判据问题。")
        elif a >= 0.65:
            print("  🟡 有中等判别力。")
        else:
            print("  ❌ 即使在真实重复片段上，仍接近瞎猜。")
    else:
        print("  样本不足，无法计算。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
