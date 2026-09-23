"""把数据摊开看：所谓的「卡住窗口」到底长什么样。

起因是一个必要且合理的质疑——**我凭什么说这些窗口是卡住的？**
如果"卡住"的定义松了（比如把只有一步轻微重复的正常轨迹也算进来），
那模型测不出来就是我的问题，不是模型的问题。

本脚本不跑模型，只做一件事：**把真实的原始数据、标注、和最终喂给监控器的
那段文本，原封不动地打印出来**，让结论可以被人工核验。

同时用更严格的阶梯式标准重新划一遍：

  L1 宽松  窗口内任意一步 last_step_redundant == True
  L2 中等  窗口内连续 >= 2 步冗余
  L3 严格  窗口内连续 >= 3 步冗余，或同一个动作连续出现 >= 3 次
  L4 铁证  连续 >= 3 步冗余 **且** 这些步的动作代码完全相同

L1 到 L4 越往下越接近"真的卡死"。如果连 L4 的窗口模型都测不出来，
那"模型不行"这个结论就站得住了。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

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


def val(s: dict, key: str, default=None):
    return (s.get("value") or {}).get(key, default)


def act(s: dict) -> str:
    return val(s, "code", "") or ""


def thought(s: dict) -> str:
    return val(s, "thought", "") or ""


def is_red(s: dict) -> bool:
    return bool(val(s, "last_step_redundant", False))


def max_consecutive_red(win: list[dict]) -> int:
    best = cur = 0
    for s in win:
        cur = cur + 1 if is_red(s) else 0
        best = max(best, cur)
    return best


def max_consecutive_same_action(win: list[dict]) -> int:
    """同一个 pyautogui 动作连续出现的最大次数。"""
    best = cur = 0
    prev = None
    for s in win:
        a = act(s)
        if a and a == prev:
            cur += 1
        else:
            cur = 1
            prev = a
        best = max(best, cur)
    return best


def render_monitor_input(win: list[dict]) -> str:
    """监控器实际看到的文本（官方推理格式）。"""
    return "\n".join(
        f"Step {i}:\nResponse: {thought(s)}\nAction: {act(s)}\n"
        for i, s in enumerate(win, start=1)
    )


def show(win: list[dict], task: str, reason: str, score: float | None, title: str) -> None:
    print(f"\n{'─' * 76}")
    print(f"  {title}")
    print(f"{'─' * 76}")
    print(f"  任务: {task[:90]}")
    if reason:
        print(f"  数据集对整条轨迹的评语: {reason[:160]}")
    if score is not None:
        print(f"  ⭐ 监控器打分 P(卡住) = {score:.6f}")
    print()
    print("  逐步标注：")
    for i, s in enumerate(win, start=1):
        flag = "冗余" if is_red(s) else "  "
        corr = "对" if val(s, "last_step_correct", False) else "错"
        print(f"    [{i}] {flag} {corr}  {act(s)[:56]}")
    print()
    print("  ── 喂给监控器的原文（截断显示）──")
    text = render_monitor_input(win)
    for line in text.splitlines()[:12]:
        print(f"    {line[:100]}")
    if len(text.splitlines()) > 12:
        print(f"    …（共 {len(text)} 字符）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/agentnet_ubuntu_5k.jsonl")
    ap.add_argument("--scan", type=int, default=400)
    ap.add_argument("--examples", type=int, default=2)
    args = ap.parse_args()

    records = load(args.data, args.scan)

    print("=" * 78)
    print("  一、这份数据里的标注到底长什么样")
    print("=" * 78)

    n_steps = n_red = 0
    n_traj_with_red = 0
    complete = Counter()
    for rec in records:
        traj = rec.get("traj") or []
        complete[bool(rec.get("task_completed"))] += 1
        reds = sum(is_red(s) for s in traj)
        n_steps += len(traj)
        n_red += reds
        if reds:
            n_traj_with_red += 1

    print(f"  扫描轨迹数            : {len(records)}")
    print(f"  总步数                : {n_steps}")
    print(f"  标为冗余的步数         : {n_red}  ({n_red / n_steps * 100:.2f}%)")
    print(f"  含至少一步冗余的轨迹    : {n_traj_with_red}  ({n_traj_with_red / len(records) * 100:.1f}%)")
    print(f"  任务成功 / 失败        : {complete[True]} / {complete[False]}")

    # ---------- 阶梯式构造 ----------
    tiers: dict[str, list[dict]] = {f"L{i}": [] for i in range(1, 5)}
    for rec in records:
        traj = rec.get("traj") or []
        if len(traj) < WINDOW:
            continue
        for end in range(WINDOW, len(traj) + 1):
            win = traj[end - WINDOW : end]
            mred = max_consecutive_red(win)
            msame = max_consecutive_same_action(win)
            sample = {"win": win, "task": rec.get("instruction") or "",
                      "reason": rec.get("reason") or ""}
            if mred >= 1:
                tiers["L1"].append(sample)
            if mred >= 2:
                tiers["L2"].append(sample)
            if mred >= 3 or msame >= 3:
                tiers["L3"].append(sample)
            if mred >= 3 and msame >= 3:
                tiers["L4"].append(sample)

    print()
    print("=" * 78)
    print("  二、阶梯式标准：越往下越接近「真的卡死」")
    print("=" * 78)
    desc = {
        "L1": "窗口内 >=1 步冗余",
        "L2": "连续 >=2 步冗余",
        "L3": "连续 >=3 步冗余 或 同一动作连续 >=3 次",
        "L4": "连续 >=3 步冗余 且 同一动作连续 >=3 次",
    }
    for k in ["L1", "L2", "L3", "L4"]:
        print(f"  {k}  {desc[k]:<42} 命中 {len(tiers[k]):>6} 个窗口")

    # ---------- 展示具体样本 ----------
    print()
    print("=" * 78)
    print("  三、具体样本（请你亲眼判断这些算不算卡住）")
    print("=" * 78)

    if tiers["L4"]:
        print("\n  【最严格 L4 —— 连续 3 步以上冗余，且动作代码完全相同】")
        for s in tiers["L4"][: args.examples]:
            show(s["win"], s["task"], s["reason"], None, "L4 样本")
    else:
        print("\n  ⚠️ L4 没有任何样本！说明这份数据里几乎没有"
              "「连续重复同一个动作」的轨迹。")
        print("     这意味着我上一轮说的'卡住窗口'，大部分其实只是零星的冗余步，")
        print("     严格来讲**不能算真的卡住**。")

    if tiers["L3"]:
        print("\n  【L3 —— 连续 3 步冗余 或 同动作连续 3 次】")
        for s in tiers["L3"][: args.examples]:
            show(s["win"], s["task"], s["reason"], None, "L3 样本")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
