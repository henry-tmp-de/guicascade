"""在**自己跑出来的安卓轨迹**上测两个监控器。

## 为什么这批数据比之前所有测试集都好

之前用的都是别人的数据：

    AgentNet       真人演示 —— 真人不会卡住，没有真卡死样本
    EvoCUAFeedback 只有 8 条，且场景是桌面

这批是**这个项目自己的分布**：小模型（Qwen3-VL-2B）在安卓上真的会卡，
而且**卡住是自然发生的**（连续重复同一动作），不是人工构造的。
强模型（8B）跑出来的干净轨迹则提供了对照。

一句话：**这是我们真正要用它的那个分布。**

## 真值怎么来

  卡住   末尾连续 >=3 步同一动作     （纯程序判定，不看模型输出）
  里程碑 成功轨迹的最后一步          （任务跑完必有完成动作）

两个都是**独立于监控器**的判据，所以用来算 AUC 是干净的。

## 顺带回答一个问题

监控器在"桌面轨迹"上训的，搬到"安卓轨迹"上还灵吗？
——这正是原方法没有回答、而本项目要回答的那个问题。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

WINDOW = 6


def load_runs(root: str) -> list[dict[str, Any]]:
    """读一个 collect 目录下所有 episode。"""
    runs = []
    for d in sorted(Path(root).rglob("traj.jsonl")):
        rows = []
        for line in d.open(encoding="utf-8"):
            line = line.strip()
            if line:
                import json

                rows.append(json.loads(line))
        if len(rows) < 2:
            continue
        rf = d.parent / "result.txt"
        runs.append({
            "name": f"{d.parent.parent.name}/{d.parent.name}",
            "steps": rows,
            "success": rf.exists() and rf.read_text().strip() == "1",
        })
    return runs


def build_window(steps: list[dict], end: int, window: int = WINDOW) -> str:
    """按监控器训练时的格式构造输入：Step N:\\nResponse: ..\\nAction: .."""
    start = max(0, end - (window - 1))
    return "\n".join(
        f"Step {int(steps[i].get('step_num', i + 1))}:\n"
        f"Response: {steps[i].get('response', '')}\n"
        f"Action: {steps[i].get('action', '')}\n"
        for i in range(start, end + 1)
    ).strip() + "\n"


def repeat_streak(steps: list[dict], end: int, window: int = WINDOW) -> int:
    """窗口末尾连续同一动作的步数。"""
    start = max(0, end - (window - 1))
    acts = [steps[i].get("action", "") for i in range(start, end + 1)]
    n = 0
    for a in reversed(acts):
        if a == acts[-1]:
            n += 1
        else:
            break
    return n


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="一个或多个 collect 输出目录")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    models = {}
    for label, path in [("卡住", "models/stuck-detector"), ("里程碑", "models/milestone-detector")]:
        models[label] = (
            AutoTokenizer.from_pretrained(path),
            AutoModelForSequenceClassification.from_pretrained(path).to(args.device).eval(),
        )

    def score(label: str, texts: list[str]) -> list[float]:
        tok, mdl = models[label]
        out = []
        for i in range(0, len(texts), 32):
            enc = tok(texts[i : i + 32], truncation=True, max_length=2048,
                      padding="max_length", return_tensors="pt").to(args.device)
            with torch.no_grad():
                out.extend(torch.softmax(mdl(**enc).logits, -1)[:, 1].tolist())
        return out

    runs = []
    for r in args.roots:
        got = load_runs(r)
        runs.extend(got)
        print(f"  {r}: {len(got)} 条")

    n_ok = sum(r["success"] for r in runs)
    print(f"\n共 {len(runs)} 条轨迹，成功 {n_ok} 条\n")

    # ---------- 卡住 ----------
    print("=" * 78)
    print("  一、卡住监控器：真值 = 窗口末尾连续 >=3 步同一动作")
    print("=" * 78)

    # ⚠️ 负类取"整条轨迹都没有卡住行为"的窗口，而不是"这一步恰好没重复"。
    #
    # 一条 8 步的轨迹可能前 5 步在乱试、后 3 步卡住；如果按"单窗口是否重复"
    # 划分，它前面那些窗口会被当成正常样本——**但它们来自一条卡住的轨迹**，
    # 混进负类会把 AUC 压得没有任何意义。
    pos_t, neg_t = [], []
    for r in runs:
        streaks = [repeat_streak(r["steps"], i) for i in range(WINDOW - 1, len(r["steps"]))]
        traj_clean = all(s < 3 for s in streaks) and bool(streaks)
        for i in range(WINDOW - 1, len(r["steps"])):
            t = build_window(r["steps"], i)
            if repeat_streak(r["steps"], i) >= 3:
                pos_t.append(t)
            elif traj_clean:
                neg_t.append(t)

    if pos_t:
        sp, sn = score("卡住", pos_t), score("卡住", neg_t)
        a = auc(sp, sn)
        print(f"  卡住窗口 n={len(sp):<4} 均分={mean(sp):.6f}  最高={max(sp):.6f}")
        print(f"  正常窗口 n={len(sn):<4} 均分={mean(sn):.6f}  最高={max(sn):.6f}")
        print(f"  ⭐ AUC = {a:.4f}")
        print("  " + ("✅ 能分辨" if a >= 0.8 else ("🟡 有信号但弱" if a >= 0.65 else "❌ 接近瞎猜")))
    else:
        print("  ⚠️ 没有卡住窗口")

    # ---------- 里程碑 ----------
    print()
    print("=" * 78)
    print("  二、里程碑监控器：真值 = 成功轨迹的最后一步")
    print("=" * 78)

    last_t, mid_t = [], []
    for r in runs:
        n = len(r["steps"])
        if n < 2:
            continue
        if r["success"]:
            last_t.append(build_window(r["steps"], n - 1))
        # 中间步一律当负类。注意这里**混进了失败轨迹**——
        # 它们本来就是失败的，不该被当成"没达到里程碑的正常步"。
        for i in range(WINDOW - 1, n - 1):
            mid_t.append(build_window(r["steps"], i))

    if last_t and mid_t:
        pl, pm = score("里程碑", last_t), score("里程碑", mid_t)
        a = auc(pl, pm)
        print(f"  成功轨迹最后一步 n={len(pl):<4} 均分={mean(pl):.6f}  最高={max(pl):.6f}")
        print(f"  中间步           n={len(pm):<4} 均分={mean(pm):.6f}")
        print(f"  ⭐ AUC = {a:.4f}")
        print("  " + ("✅ 能分辨" if a >= 0.8 else ("🟡 有信号但弱" if a >= 0.65 else "❌ 接近瞎猜")))

    # ---------- 逐条看 ----------
    print()
    print("=" * 78)
    print("  三、逐条轨迹的最后一步得分")
    print("=" * 78)
    texts = [build_window(r["steps"], len(r["steps"]) - 1) for r in runs]
    sc = score("里程碑", texts)
    sk = score("卡住", texts)
    print(f"  {'轨迹':<34}{'成功':>5}{'步数':>5}{'里程碑P':>11}{'卡住P':>11}")
    print("  " + "-" * 68)
    for r, m, k in zip(runs, sc, sk):
        print(f"  {r['name'][:33]:<34}{'✅' if r['success'] else '❌':>5}"
              f"{len(r['steps']):>5}{m:>11.6f}{k:>11.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
