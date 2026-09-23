"""回答一个具体问题：换个阈值，这个监控器能不能用？

起因是一个很有道理的质疑：卡住窗口均分 0.057、正常窗口均分 0.00006，
差了将近一千倍——**那是不是说明它其实分辨得很好，只是阈值 0.5 定高了？**

平均数说明不了问题，因为分布可能是斜的。真正要看的是：

  1. 两个分布的**分位数**长什么样（是不是大量卡住窗口其实压在低位）
  2. 遍历所有阈值，**每个阈值下的 precision / recall / F1**
  3. 存不存在一个阈值，能同时把 precision 和 recall 都做到可用

如果扫完发现"要高 recall 就必须把阈值压到极低，而那时 precision 会崩"，
那就说明**不是阈值没调好，是信号本身不够**。

同时做一层**人工可核验的交叉验证**：把数据集自带的 `reason` 字段里
明确描述了"重复/卡住/循环"的轨迹单独挑出来，看这些**铁定卡住**的轨迹
分数是不是也不高。这能排除"是不是我判的卡住窗口不够卡"这个疑问。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

WINDOW = 6

LOOP_WORDS = re.compile(
    r"stuck|loop|repetit|redundan|cycling|same action|no progress|"
    r"unable to (?:complete|proceed|make progress)|failed to",
    re.IGNORECASE,
)


def load(path: str, limit: int | None) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def act(s: dict) -> str:
    return (s.get("value") or {}).get("code", "") or ""


def thought(s: dict) -> str:
    return (s.get("value") or {}).get("thought", "") or ""


def redundant(s: dict) -> bool:
    return bool((s.get("value") or {}).get("last_step_redundant", False))


def top_ratio(win: list[dict]) -> float:
    codes = [act(s) for s in win if act(s)]
    return Counter(codes).most_common(1)[0][1] / len(codes) if codes else 0.0


def render(win: list[dict]) -> str:
    return "\n".join(
        f"Step {i}:\nResponse: {thought(s)}\nAction: {act(s)}\n"
        for i, s in enumerate(win, start=1)
    )


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/agentnet_ubuntu_5k.jsonl")
    ap.add_argument("--model", default="models/stuck-detector")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    records = load(args.data, args.limit)

    # ---------- 构造样本 ----------
    stuck, normal, ironclad = [], [], []
    for rec in records:
        traj = rec.get("traj") or []
        if len(traj) < WINDOW:
            continue
        task = rec.get("instruction") or ""
        reason = rec.get("reason") or ""
        task_stuck_per_reason = bool(LOOP_WORDS.search(reason))

        for end in range(WINDOW, len(traj) + 1):
            win = traj[end - WINDOW : end]
            sample = {"win": win, "task": task, "reason": reason}
            if any(redundant(s) for s in win):
                stuck.append(sample)
                if task_stuck_per_reason:      # 双重证据：标注 + LLM 分析都说卡住了
                    ironclad.append(sample)
            elif top_ratio(win) < 0.5:
                normal.append(sample)

    stuck, normal, ironclad = (
        stuck[: args.max_samples],
        normal[: args.max_samples],
        ironclad[: args.max_samples],
    )

    print("=" * 78)
    print("  样本构造（人工可核验）")
    print("=" * 78)
    print(f"  卡住窗口（有冗余步标注）        : {len(stuck)}")
    print(f"  铁定卡住（标注 + reason 也说卡住）: {len(ironclad)}")
    print(f"  正常窗口（零冗余 + 动作不重复）  : {len(normal)}")
    print()
    print("  ⭐ 铁定卡住的样本，reason 字段原文摘录：")
    for s in ironclad[:3]:
        r = s["reason"]
        print(f"     · 「…{r[:150]}…」")

    # ---------- 打分 ----------
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(args.device).eval()

    def score_many(wins: list[list[dict]]) -> list[float]:
        texts = [render(w) for w in wins]
        out = []
        for i in range(0, len(texts), 32):
            enc = tok(texts[i : i + 32], truncation=True, max_length=2048,
                      padding="max_length", return_tensors="pt").to(args.device)
            with torch.no_grad():
                out.extend(torch.softmax(model(**enc).logits, -1)[:, 1].tolist())
        return out

    s_stuck = score_many([s["win"] for s in stuck])
    s_normal = score_many([s["win"] for s in normal])
    s_iron = score_many([s["win"] for s in ironclad]) if ironclad else []

    print()
    print("=" * 78)
    print("  分布（看分位数，不看均值）")
    print("=" * 78)
    print(f"  {'类别':<14}{'均值':>11}{'中位数':>11}{'p75':>11}{'p90':>11}{'p95':>11}{'最大':>11}")
    print("  " + "-" * 74)
    for name, xs in [("卡住窗口", s_stuck), ("铁定卡住", s_iron), ("正常窗口", s_normal)]:
        if not xs:
            continue
        print(f"  {name:<14}{sum(xs)/len(xs):>11.6f}{percentile(xs,0.5):>11.6f}"
              f"{percentile(xs,0.75):>11.6f}{percentile(xs,0.90):>11.6f}"
              f"{percentile(xs,0.95):>11.6f}{max(xs):>11.6f}")

    # ---------- 阈值扫描 ----------
    print()
    print("=" * 78)
    print("  阈值扫描：官方默认阈值是 0.5，我们从头扫到尾")
    print("=" * 78)
    print(f"  {'阈值':>10}{'召回率':>10}{'精确率':>10}{'F1':>10}   说明")
    print("  " + "-" * 70)

    best_f1, best_theta = -1.0, 0.0
    for theta in [0.5, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]:
        tp = sum(1 for x in s_stuck if x >= theta)
        fn = len(s_stuck) - tp
        fp = sum(1 for x in s_normal if x >= theta)
        tn = len(s_normal) - fp
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1, best_theta = f1, theta
        note = ""
        if theta == 0.5:
            note = "← 官方默认"
        if f1 == best_f1:
            note += "  ★目前最好"
        print(f"  {theta:>10.4f}{recall:>10.3f}{precision:>10.3f}{f1:>10.3f}   {note}")

    # ---------- 结论 ----------
    print()
    print("=" * 78)
    print("  结论")
    print("=" * 78)
    print(f"  最优阈值 {best_theta:.4f} 下 F1 = {best_f1:.3f}")
    if best_f1 < 0.5:
        print("  ❌ 把阈值扫遍全区间，最好的 F1 也不到 0.5。")
        print("     说明不是「阈值定高了」，而是卡住和正常的分数分布**大面积重叠**——")
        print("     均值差一千倍是因为分布极度右偏：少数卡住窗口分数很高，")
        print("     但大多数卡住窗口和正常窗口一样贴在地板上。")
    elif best_f1 < 0.75:
        print("  🟡 有中等水平的分辨力，但离可用还有距离。")
    else:
        print("  ✅ 找得到可用阈值——之前的失败只是阈值没调，模型本身没问题。")
        print(f"     建议部署阈值 {best_theta:.4f}（**不是官方默认的 0.5**）。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
