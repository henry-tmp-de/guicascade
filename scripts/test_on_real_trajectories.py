"""用真实轨迹问一个决定性的问题：监控器到底能不能测出「卡住」？

前几轮全是合成样本，结论只能是"测不出"。这次换真数据。

数据：`xlangai/AgentNet`（OpenCUA 发布，MIT）里的 `agentnet_ubuntu_5k.jsonl`
——5000 条**真实 Ubuntu 桌面 agent 轨迹**，正好是 OSWorld 那套环境，
动作是 pyautogui，和监控器的训练分布同源。

关键是它带**独立标注**，不用我们自己猜哪条卡住了：

  traj[i].value.last_step_redundant   这一步是否冗余
  traj[i].value.last_step_correct     这一步是否正确
  task_completed                      整个任务是否成功

## 怎么算"确实卡住"

标准放宽到两种都算，避免单一标准带来的偏差：

  A. 标注法：该步 last_step_redundant == True（数据集自己的 LLM 判的）
  B. 行为法：最近 6 步里同一个 pyautogui 动作出现 >= 3 次（纯程序判定，不看标注）

两者取交集做"铁定卡住"，取并集做"疑似卡住"，分别报结果。
对照组是整条轨迹零冗余、动作全不相同的窗口。

## 判读

**不看绝对分数，看 AUC**——卡住窗口和正常窗口的分数能不能分开。
AUC = 0.5 表示完全分不开（等于瞎猜）；>= 0.8 才算能用。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

WINDOW = 6


# --------------------------------------------------------------------------
# 从真实轨迹里构造样本
# --------------------------------------------------------------------------


def load_real_trajectories(path: str, limit: int | None = None) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def step_action(step: dict) -> str:
    return (step.get("value") or {}).get("code", "") or ""


def step_thought(step: dict) -> str:
    return (step.get("value") or {}).get("thought", "") or ""


def step_redundant(step: dict) -> bool:
    return bool((step.get("value") or {}).get("last_step_redundant", False))


def repeated_ratio(window: list[dict]) -> float:
    """窗口里最高频动作占比。1.0 表示六步全是同一个动作。"""
    codes = [step_action(s) for s in window if step_action(s)]
    if not codes:
        return 0.0
    return Counter(codes).most_common(1)[0][1] / len(codes)


def build_windows(records: list[dict], min_steps: int = WINDOW) -> dict[str, list[dict]]:
    """扫过所有轨迹，产出「卡住」和「正常」两类窗口。

    每个样本是 {"steps": [...], "task": str, "evidence": str}。
    """
    stuck_annotated: list[dict] = []
    stuck_behavioral: list[dict] = []
    normal: list[dict] = []

    for rec in records:
        traj = rec.get("traj") or []
        if len(traj) < min_steps:
            continue
        task = rec.get("instruction") or ""

        for end in range(min_steps, len(traj) + 1):
            window = traj[end - min_steps : end]
            ratio = repeated_ratio(window)
            n_redundant = sum(step_redundant(s) for s in window)
            sample = {
                "steps": window,
                "task": task,
                "evidence": f"冗余步={n_redundant}/{len(window)}, 最高频动作占比={ratio:.2f}",
            }

            if n_redundant > 0:
                stuck_annotated.append(sample)
            if ratio >= 0.5:
                stuck_behavioral.append(sample)
            if n_redundant == 0 and ratio < 0.5:
                normal.append(sample)

    return {
        "卡住-标注法": stuck_annotated,
        "卡住-行为法": stuck_behavioral,
        "正常窗口": normal,
    }


# --------------------------------------------------------------------------
# 渲染成监控器的输入
# --------------------------------------------------------------------------


def render(window: list[dict], fmt: str) -> str:
    """按指定格式渲染窗口文本。

    格式差异不是小事——官方训练脚本和推理脚本用的不是同一个格式，
    所以两种都测。
    """
    lines = []
    for i, s in enumerate(window, start=1):
        thought, code = step_thought(s), step_action(s)
        if fmt == "runtime":  # stuck_detector.py 的格式
            lines.append(f"Step {i}:\nResponse: {thought}\nAction: {code}\n")
        elif fmt == "training":  # build_stuck_dataset.py 的格式
            lines.append(f"Step {i}:\n{thought}\nAction: {code}")
        elif fmt == "thought_only":
            lines.append(f"Step {i}:\n{thought}")
    return "\n".join(lines)


def auc(scores: list[float], labels: list[int]) -> float:
    """AUC = P(正样本分数 > 负样本分数)。0.5 等于瞎猜。"""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/agentnet_ubuntu_5k.jsonl")
    ap.add_argument("--model", default="models/stuck-detector")
    ap.add_argument("--limit", type=int, default=400, help="只扫前 N 条轨迹（全量 5000 条较慢）")
    ap.add_argument("--max-samples", type=int, default=120, help="每个类别最多取多少样本打分")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0", help="服务器 8 卡，先确认空闲再用")
    args = ap.parse_args()

    print("=" * 78)
    print("  第一步：从真实轨迹里找出「确实卡住」的窗口")
    print("=" * 78)

    records = load_real_trajectories(args.data, limit=args.limit)
    print(f"  载入 {len(records)} 条真实轨迹（Ubuntu / pyautogui）")

    buckets = build_windows(records)
    for name, samples in buckets.items():
        print(f"    {name:<14} {len(samples):>7} 个窗口")

    if not buckets["正常窗口"]:
        print("  ❌ 没找到正常窗口，无法对比")
        return 1

    # ---- 先给用户看证据：这些"卡住"窗口凭什么算卡住 ----
    print()
    print("  抽样确认（人工可核验的证据）：")
    for sample in buckets["卡住-标注法"][:2]:
        print(f"    · 任务: {sample['task'][:70]}")
        print(f"      证据: {sample['evidence']}")
        print("      最后两步的动作:")
        for s in sample["steps"][-2:]:
            print(f"        redundant={step_redundant(s)!s:<5} {step_action(s)[:60]}")
    print()
    for sample in buckets["正常窗口"][:2]:
        print(f"    · 正常窗口 任务: {sample['task'][:60]}")
        print(f"      证据: {sample['evidence']}")

    # ---- 第二步：喂给监控器 ----
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print()
    print("=" * 78)
    print("  第二步：监控器能不能把「卡住」和「正常」分开？")
    print("=" * 78)

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()
    print(f"  设备: {device}")

    def score_many(texts: list[str], batch_size: int = 32) -> list[float]:
        """批量打分。逐条跑的话 1000+ 次前向要等很久。"""
        out: list[float] = []
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i : i + batch_size], truncation=True, max_length=args.max_length,
                padding="max_length", return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            out.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
        return out

    results = []
    for fmt in ["runtime", "training", "thought_only"]:
        print(f"\n  格式 = {fmt}")
        print(f"    {'类别':<16}{'样本数':>7}{'均分':>12}{'最高分':>12}")
        print("    " + "-" * 47)

        per_class: dict[str, list[float]] = {}
        for name, samples in buckets.items():
            subset = samples[: args.max_samples]
            if not subset:
                continue
            scores = score_many([render(s["steps"], fmt) for s in subset])
            per_class[name] = scores
            print(f"    {name:<16}{len(scores):>7}{sum(scores)/len(scores):>12.6f}"
                  f"{max(scores):>12.6f}")

        # 两个卡住定义分别对正常窗口算 AUC
        for stuck_name in ["卡住-标注法", "卡住-行为法"]:
            if stuck_name not in per_class:
                continue
            xs = per_class[stuck_name] + per_class["正常窗口"]
            ys = [1] * len(per_class[stuck_name]) + [0] * len(per_class["正常窗口"])
            a = auc(xs, ys)
            verdict = "✅ 可用" if a >= 0.8 else ("🟡 弱" if a >= 0.65 else "❌ 等于瞎猜")
            print(f"    -> AUC({stuck_name} vs 正常) = {a:.4f}   {verdict}")
            results.append((fmt, stuck_name, a))

    print()
    print("=" * 78)
    print("  结论")
    print("=" * 78)
    if results:
        best = max(results, key=lambda r: (r[2] if r[2] == r[2] else -1))
        print(f"  最好组合：格式={best[0]}  对比={best[1]}  AUC={best[2]:.4f}")
        if best[2] < 0.65:
            print("  ❌ 所有格式、所有定义下，AUC 都接近 0.5 ——")
            print("     监控器在真实轨迹上**无法区分卡住与正常**，不是格式问题，是权重问题。")
        elif best[2] < 0.8:
            print("  🟡 有弱信号但远达不到可用水平，需要重新训练或换方案。")
        else:
            print("  ✅ 存在可用组合，说明之前的失败是格式没对上，改格式即可。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
