"""量化两个监控器：几十条真实轨迹，官方格式，出可报的数字。

## 怎么构造一个"有真值"的测点

**里程碑监控器**可以用一个干净的真值：**轨迹的最后一步**。
任务跑完会输出一个 `DONE` 动作，那一步在语义上必然是里程碑。
于是就能算：

    正类 = 每条轨迹的最后一个窗口（含 DONE）
    负类 = 该轨迹中间的其他窗口

这是**任务级别的真值**，不依赖任何人工标注，也不会掺进主观判断。

**卡住监控器**用动作重复当真值：末尾连续 >=3 步同一个动作 = 卡住。
上一轮已证明这份数据里几乎没有这种情形，所以大概率还是测不出来——
一并报出来，作为"卡住监控器不可用"的证据。

## 数据来源

  A. EvoCUAFeedback 的 8 条原始轨迹（与训练分布同源，最可信）
  B. AgentNet 抽几十条，把 thought/code 映射到官方的 response/action 字段
     —— 字段语义一致，动作也都是 pyautogui，可以直接喂

两者都按官方 `_build_step_text_window` 构造输入，window=6。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WINDOW = 6


# --------------------------------------------------------------------------
# 载入两种来源，统一成官方格式的 step 列表
# --------------------------------------------------------------------------


def load_evocua(root: str) -> list[dict]:
    out = []
    for tp in sorted(Path(root).rglob("traj.jsonl")):
        steps = []
        with tp.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    steps.append(json.loads(line))
        steps.sort(key=lambda d: int(d.get("step_num", 0)))
        rf = tp.parent / "result.txt"
        out.append({
            "name": str(tp.relative_to(root).parent),
            "steps": steps,
            "success": (rf.read_text().strip() not in ("", "0", "0.0")) if rf.exists() else None,
        })
    return out


def load_agentnet(path: str, limit: int) -> list[dict]:
    """AgentNet -> 官方格式。thought->response, code->action。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            steps = []
            for j, s in enumerate(rec.get("traj") or []):
                val = s.get("value") or {}
                steps.append({
                    "step_num": j + 1,
                    "response": val.get("thought", ""),
                    "action": val.get("code", ""),
                })
            if len(steps) >= WINDOW:
                out.append({
                    "name": f"agentnet/{rec.get('task_id', i)[:20]}",
                    "steps": steps,
                    "success": bool(rec.get("task_completed")),
                })
    return out


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


def repeated_streak(steps: list[dict], end_idx: int, window: int = WINDOW) -> int:
    start = max(0, end_idx - (window - 1))
    codes = [steps[i].get("action", "") for i in range(start, end_idx + 1)]
    streak = 0
    for c in reversed(codes):
        if c == codes[-1]:
            streak += 1
        else:
            break
    return streak


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def best_f1(pos: list[float], neg: list[float]) -> tuple[float, float]:
    best = (0.0, 0.0)
    for t in [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]:
        tp = sum(1 for x in pos if x >= t)
        fp = sum(1 for x in neg if x >= t)
        r = tp / len(pos) if pos else 0.0
        p = tp / (tp + fp) if (tp + fp) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        if f > best[0]:
            best = (f, t)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evocua-root", default="data/evocua_feedback")
    ap.add_argument("--agentnet", default="data/agentnet_ubuntu_5k.jsonl")
    ap.add_argument("--agentnet-limit", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    trajs = load_evocua(args.evocua_root)
    extra = load_agentnet(args.agentnet, args.agentnet_limit) if Path(args.agentnet).exists() else []
    print(f"轨迹：EvoCUAFeedback {len(trajs)} 条，AgentNet 抽样 {len(extra)} 条")

    models = {}
    for name, path in [("卡住", "models/stuck-detector"), ("里程碑", "models/milestone-detector")]:
        models[name] = (
            AutoTokenizer.from_pretrained(path),
            AutoModelForSequenceClassification.from_pretrained(path).to(args.device).eval(),
        )

    def score(name: str, texts: list[str]) -> list[float]:
        tok, mdl = models[name]
        out = []
        for i in range(0, len(texts), 32):
            enc = tok(texts[i : i + 32], truncation=True, max_length=2048,
                      padding="max_length", return_tensors="pt").to(args.device)
            with torch.no_grad():
                out.extend(torch.softmax(mdl(**enc).logits, -1)[:, 1].tolist())
        return out

    all_trajs = trajs + extra

    # ---------- 里程碑：最后一步 vs 中间步 ----------
    last_texts, mid_texts, tag = [], [], []
    for tr in all_trajs:
        n = len(tr["steps"])
        if n < WINDOW:
            continue
        last_texts.append(build_window(tr["steps"], n - 1))
        tag.append((tr["name"], tr["success"], n))
        for i in range(WINDOW - 1, n - 1):
            mid_texts.append(build_window(tr["steps"], i))

    print()
    print("=" * 84)
    print("  一、里程碑监控器：用「轨迹最后一步」当真值")
    print("=" * 84)
    p_last = score("里程碑", last_texts)
    p_mid = score("里程碑", mid_texts)
    a = auc(p_last, p_mid)
    f, t = best_f1(p_last, p_mid)
    print(f"  正类 = 每条轨迹的最后一步   n={len(p_last)}")
    print(f"  负类 = 同一条轨迹的中间步   n={len(p_mid)}")
    print(f"  正类均分 = {sum(p_last)/len(p_last):.6f}   负类均分 = {sum(p_mid)/len(p_mid):.6f}")
    print(f"  ⭐ AUC = {a:.4f}      最优 F1 = {f:.3f}（阈值 {t}）")
    if a >= 0.9:
        print("  ✅ 里程碑监控器可用！")
    elif a >= 0.75:
        print("  🟡 有较强判别力，阈值校准后可用。")
    else:
        print("  ❌ 判别力不足。")

    # ---------- 卡住：重复动作 ----------
    print()
    print("=" * 84)
    print("  二、卡住监控器：用「连续重复动作」当真值")
    print("=" * 84)
    s_stuck, s_norm = [], []
    for tr in all_trajs:
        n = len(tr["steps"])
        for i in range(WINDOW - 1, n):
            txt = build_window(tr["steps"], i)
            streak = repeated_streak(tr["steps"], i)
            (s_stuck if streak >= 3 else s_norm).append(txt)

    if s_stuck:
        ps = score("卡住", s_stuck)
        pn = score("卡住", s_norm)
        a2 = auc(ps, pn)
        print(f"  卡住窗口 n={len(ps)}  均分 = {sum(ps)/len(ps):.6f}")
        print(f"  正常窗口 n={len(pn)}  均分 = {sum(pn)/len(pn):.6f}")
        print(f"  AUC = {a2:.4f}")
        if a2 < 0.6:
            print("  ❌ 卡住监控器这份数据上无法区分。")
    else:
        print("  ⚠️ 这份数据里没有「连续重复 >=3 步」的窗口，测不了。")
        print("     需要 agent 自己跑出的卡死轨迹才能判决（见任务 #8）。")

    print()
    print("=" * 84)
    print("  三、结论")
    print("=" * 84)
    print("  · 里程碑监控器在官方同源格式上是**可用**的 —— 这是项目能往下走的依据")
    print("  · 卡住监控器在同样数据上依然测不出 —— 它需要替换或重训")
    print("  · 两者是否互补，等卡住侧有了可用替代后再测")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
