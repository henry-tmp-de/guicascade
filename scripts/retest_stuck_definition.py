"""修正版测试：先把「卡住」定义对，再问模型行不行。

## 上一版错在哪

上一版拿 `last_step_redundant == True`（单步冗余）当卡住，圈出 200 个"卡住窗口"。
摊开数据一看：**这些窗口绝大多数只是正常轨迹里夹了一步轻微重复**。
严格标准下——连续 3 步冗余且动作相同——**全数据集 400 条里命中 0 个**。

拿"正常探索"当"卡住"去测，模型测不出来是理所当然的。结论作废。

## 这一版的定义

StepWise 原文对卡住给了三条判据：
  1. 同一个动作重复多次且没有进展
  2. 陷入错误循环
  3. **连续几步没有做出有意义的进展**

AgentNet 恰好有能对应第 3 条的字段：`last_step_correct`——这一步是否达成了它的意图。
**连续 N 步"没达成意图"就是"连续几步没进展"**，这是对判据 3 的直接翻译，
不是我自己发明的标准。

本脚本按连续失败步数分档，每档都报：
  · 有多少个窗口
  · 监控器的 AUC 和最优阈值下的 F1

**如果连连续失败 3 步、4 步这种铁定卡死的窗口都测不出来，那才是真的不行。**
"""

from __future__ import annotations

import argparse
import json

WINDOW = 6


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


def v(s: dict, k: str, d=None):
    return (s.get("value") or {}).get(k, d)


def act(s: dict) -> str:
    return v(s, "code", "") or ""


def thought(s: dict) -> str:
    return v(s, "thought", "") or ""


def correct(s: dict) -> bool:
    """这一步是否达成了它的意图。"""
    return bool(v(s, "last_step_correct", True))


def max_consecutive_wrong(win: list[dict]) -> int:
    """窗口内连续「没达成意图」的最大步数。"""
    best = cur = 0
    for s in win:
        cur = 0 if correct(s) else cur + 1
        best = max(best, cur)
    return best


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


def auc(xs: list[float], ys: list[int]) -> float:
    pos = [s for s, y in zip(xs, ys) if y == 1]
    neg = [s for s, y in zip(xs, ys) if y == 0]
    if not pos or not neg:
        return float("nan")
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def best_f1(s_pos: list[float], s_neg: list[float]) -> tuple[float, float, float, float]:
    """返回 (最优F1, 该阈值, 召回, 精确)。"""
    best = (0.0, 0.0, 0.0, 0.0)
    for theta in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]:
        tp = sum(1 for x in s_pos if x >= theta)
        fp = sum(1 for x in s_neg if x >= theta)
        r = tp / len(s_pos) if s_pos else 0.0
        p = tp / (tp + fp) if (tp + fp) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        if f > best[0]:
            best = (f, theta, r, p)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/agentnet_ubuntu_5k.jsonl")
    ap.add_argument("--model", default="models/stuck-detector")
    ap.add_argument("--scan", type=int, default=800)
    ap.add_argument("--max-samples", type=int, default=150)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    records = load(args.data, args.scan)

    # ---- 阶梯：连续失败 N 步 ----
    tiers: dict[int, list[dict]] = {1: [], 2: [], 3: [], 4: []}
    normal: list[dict] = []
    for rec in records:
        traj = rec.get("traj") or []
        if len(traj) < WINDOW:
            continue
        for end in range(WINDOW, len(traj) + 1):
            win = traj[end - WINDOW : end]
            n = max_consecutive_wrong(win)
            sample = {"win": win, "task": rec.get("instruction") or ""}
            if n >= 1:
                for k in tiers:
                    if n >= k:
                        tiers[k].append(sample)
            else:
                normal.append(sample)

    print("=" * 78)
    print("  一、按「连续几步没达成意图」分档（StepWise 判据 3 的直接翻译）")
    print("=" * 78)
    for k in [1, 2, 3, 4]:
        print(f"  连续失败 >= {k} 步 : {len(tiers[k]):>7} 个窗口")
    print(f"  完全正常窗口      : {len(normal):>7} 个窗口")

    if not tiers[3]:
        print("\n  ⚠️ 连续失败 3 步以上的窗口一个都没有。这份数据里确实没有真正的卡死轨迹。")

    # ---- 打分 ----
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

    s_normal = score_many([s["win"] for s in normal[: args.max_samples]])

    print()
    print("=" * 78)
    print("  二、监控器表现")
    print("=" * 78)
    print(f"  正常窗口均分 = {sum(s_normal)/len(s_normal):.6f}   "
          f"中位数 = {percentile(s_normal,0.5):.6f}   "
          f"p95 = {percentile(s_normal,0.95):.6f}   最大 = {max(s_normal):.6f}")
    print()
    print(f"  {'卡住定义':<22}{'样本':>6}{'均分':>12}{'中位数':>10}{'AUC':>9}{'最优F1':>9}{'召回':>8}{'精确':>8}")
    print("  " + "-" * 82)

    summary = []
    for k in [1, 2, 3, 4]:
        subset = tiers[k][: args.max_samples]
        if not subset:
            print(f"  连续失败 >= {k} 步{'':<10}{0:>6}   （无样本）")
            continue
        s_pos = score_many([s["win"] for s in subset])
        a = auc(s_pos + s_normal, [1] * len(s_pos) + [0] * len(s_normal))
        f, theta, r, p = best_f1(s_pos, s_normal)
        summary.append((k, a, f, theta))
        print(f"  连续失败 >= {k} 步{'':<10}{len(s_pos):>6}"
              f"{sum(s_pos)/len(s_pos):>12.6f}{percentile(s_pos,0.5):>10.6f}"
              f"{a:>9.4f}{f:>9.3f}{r:>8.3f}{p:>8.3f}")

    print()
    print("=" * 78)
    print("  三、结论")
    print("=" * 78)
    if summary:
        k, a, f, theta = max(summary, key=lambda x: (x[1] if x[1] == x[1] else -1))
        print(f"  表现最好的档：连续失败 >= {k} 步，AUC = {a:.4f}，最优 F1 = {f:.3f}（阈值 {theta:.4f}）")
        print()
        if a >= 0.8:
            print("  ✅ 在真正的失败轨迹上，监控器有可用判别力。")
            print("     上一轮结论作废——测不出是因为样本选错了，不是模型不行。")
        elif a >= 0.65:
            print("  🟡 有中等判别力，但仍不足以直接当开关用。")
        else:
            print("  ❌ 即使在铁定失败的窗口上，AUC 仍接近 0.5。")
        print()
        print("  ⚠️ 但必须说清楚：本数据集是**真人演示录制**的（OpenCUA），")
        print("     真人极少钻牛角尖，'连续失败 3 步以上'的样本本来就稀缺。")
        print("     所以这里的 AUC 反映的是「在人类失败轨迹上的表现」，")
        print("     不等于它在真实 agent 卡死轨迹上的表现——那需要 agent 自己跑出来的数据。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
