"""分级测试（真实数据版）：两个监控器一起测，并查清"分数为什么低"。

## 要回答三个问题

1. **为什么合成探针分数只有 1e-4，真实轨迹却有 0.17？** 怀疑是文本长度/丰富度。
   做法：拿同一段内容，只改长度，看分数怎么动。

2. **分数随「连续失败步数」单调上升吗？** 用真实轨迹按等级切，两个模型都测。

3. **里程碑监控器在同样的数据上表现如何？** 它和卡住监控器是不是互补的。

## 为什么用「连续失败步数」当等级

StepWise 原文对卡住的第三条判据是「连续几步没有做出有意义的进展」。
AgentNet 的 `last_step_correct` 正是「这一步有没有达成意图」，
**连续 N 步没达成意图 = 连续 N 步没有进展**，是对判据 3 的直接翻译。
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


def window_text(win: list[dict], *, with_obs: bool = False) -> str:
    """官方推理格式：Step N:\\nResponse: {reason}\\nAction: {action}\\n"""
    out = []
    for i, s in enumerate(win, start=1):
        reason = v(s, "thought", "") or ""
        if with_obs:
            obs = v(s, "observation", "") or ""
            if obs:
                reason = f"{reason}\nScreen: {obs}"
        out.append(f"Step {i}:\nResponse: {reason}\nAction: {v(s, 'code', '')}\n")
    return "\n".join(out)


def max_consec_fail(win: list[dict]) -> int:
    best = cur = 0
    for s in win:
        cur = 0 if bool(v(s, "last_step_correct", True)) else cur + 1
        best = max(best, cur)
    return best


def max_consec_red(win: list[dict]) -> int:
    best = cur = 0
    for s in win:
        cur = cur + 1 if bool(v(s, "last_step_redundant", False)) else 0
        best = max(best, cur)
    return best


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/agentnet_ubuntu_5k.jsonl")
    ap.add_argument("--scan", type=int, default=5000, help="全量扫，尽量多找高等级样本")
    ap.add_argument("--max-per-tier", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    models = {}
    for key, path in [("卡住监控器", "models/stuck-detector"),
                      ("里程碑监控器", "models/milestone-detector")]:
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForSequenceClassification.from_pretrained(path).to(args.device).eval()
        models[key] = (tok, mdl)
    print(f"已加载 {len(models)} 个监控器，设备 {args.device}")

    def score(model_key: str, texts: list[str], batch: int = 32) -> list[float]:
        tok, mdl = models[model_key]
        out = []
        for i in range(0, len(texts), batch):
            enc = tok(texts[i : i + batch], truncation=True, max_length=2048,
                      padding="max_length", return_tensors="pt").to(args.device)
            with torch.no_grad():
                out.extend(torch.softmax(mdl(**enc).logits, -1)[:, 1].tolist())
        return out

    # ================= 第一部分：长度是不是混淆因素 =================
    print()
    print("=" * 80)
    print("  一、文本长度的影响（同一内容，只改长度）")
    print("=" * 80)

    records = load(args.data, args.scan)

    # 找一条有连续失败的轨迹，取它的窗口
    probe_win = None
    for rec in records:
        traj = rec.get("traj") or []
        for end in range(WINDOW, len(traj) + 1):
            w = traj[end - WINDOW : end]
            if max_consec_fail(w) >= 2:
                probe_win = w
                break
        if probe_win:
            break

    if probe_win:
        base = window_text(probe_win)
        print(f"  基准窗口：{len(base)} 字符（真实轨迹，连续失败 >=2 步）")
        print()
        print(f"  {'变体':<34}{'字符数':>9}{'卡住监控器':>14}{'里程碑监控器':>14}")

        variants = [
            ("完整（理由 + 屏幕描述）", window_text(probe_win, with_obs=True)),
            ("完整（只有理由）", base),
            ("只保留前 3 步", "\n".join(base.split("\n")[: 3 * 3])),
            ("理由截断到 60 字", "\n".join(
                (ln[:60] + "…") if ln.startswith("Response:") and len(ln) > 60 else ln
                for ln in base.split("\n"))),
            ("把理由全删掉，只留动作", "\n".join(
                ln for ln in base.split("\n")
                if ln.startswith("Step") or ln.startswith("Action:"))),
        ]
        for name, text in variants:
            s1 = score("卡住监控器", [text])[0]
            s2 = score("里程碑监控器", [text])[0]
            print(f"  {name:<34}{len(text):>9}{s1:>14.6f}{s2:>14.6f}")
    else:
        print("  （没找到连续失败 >=2 步的窗口）")

    # ================= 第二部分：真实数据的分级测试 =================
    print()
    print("=" * 80)
    print("  二、真实轨迹上的分级测试（等级 = 连续失败步数）")
    print("=" * 80)

    tiers: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: [], 4: []}
    red_tiers: dict[int, list[dict]] = {1: [], 2: [], 3: []}

    for rec in records:
        traj = rec.get("traj") or []
        if len(traj) < WINDOW:
            continue
        for end in range(WINDOW, len(traj) + 1):
            w = traj[end - WINDOW : end]
            f = min(max_consec_fail(w), 4)
            if len(tiers[f]) < args.max_per_tier:
                tiers[f].append(w)
            r = min(max_consec_red(w), 3)
            if r >= 1 and len(red_tiers[r]) < args.max_per_tier:
                red_tiers[r].append(w)

    print(f"  扫描 {len(records)} 条轨迹")
    for k in range(5):
        print(f"    连续失败 = {k} 步 : {len(tiers[k]):>5} 个窗口（上限 {args.max_per_tier}）")
    for k in range(1, 4):
        print(f"    连续冗余 >= {k} 步 : {len(red_tiers[k]):>5} 个窗口")

    print()
    print(f"  {'等级':<26}{'样本':>6}{'卡住均分':>13}{'里程碑均分':>14}{'里程碑负向?':>12}")
    print("  " + "-" * 74)

    rows = []
    for label, table in [("连续失败", tiers), ("连续冗余", red_tiers)]:
        keys = sorted(table)
        for k in keys:
            wins = table[k]
            if not wins:
                continue
            texts = [window_text(w) for w in wins]
            s_stuck = score("卡住监控器", texts)
            s_mile = score("里程碑监控器", texts)
            name = f"{label} = {k} 步"
            neg = "是 ←" if s_mile and mean(s_mile) < mean(score("里程碑监控器", [window_text(w) for w in tiers[0]])) else "否"
            print(f"  {name:<26}{len(wins):>6}{mean(s_stuck):>13.6f}{mean(s_mile):>14.6f}{neg:>12}")
            rows.append((name, len(wins), mean(s_stuck), mean(s_mile)))

    # ================= 第三部分：结论 =================
    print()
    print("=" * 80)
    print("  三、读法")
    print("=" * 80)
    print("  · 看「卡住均分」这一列随等级上升吗 —— 上升说明监控器能排程度")
    print("  · 看「里程碑均分」是否随等级**下降** —— 下降说明两个监控器互补")
    print("    （卡住监控器管局部循环，里程碑监控器管语义漂移，本来该呈反向）")
    print("  · 绝对量级仍要考虑：如果所有值都在 1e-4，秩上有序但阈值没法切")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
