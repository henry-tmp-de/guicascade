"""在**官方同源格式**的真实轨迹上跑监控器。

## 为什么这份数据比之前所有测试集都好

`EvoCUATeam/EvoCUAFeedback` 里是 EvoCUA / Qwen3-VL 跑 OSWorld 的原始输出目录，
结构和 StepWise 评估脚本读的 `BERT_Training/evocua_8b/` **一模一样**：

    <model>/<task_id>/traj.jsonl     ← 每步一行，含 response / action / reward
    <model>/<task_id>/result.txt     ← 任务判分

而 `traj.jsonl` 的字段正好是官方读的那两个：

    {"step_num": 1,
     "response": "<上千字符的完整思维链>",
     "action": "pyautogui.click(1852, 223)",
     "reward": 0, "done": false}

**这是本项目能找到的、与监控器训练分布最接近的数据。** 之前所有测试
（合成探针、AgentNet）要么格式不对、要么模型家族不对，这份两个都对。

## 本脚本做三件事

1. **完全照搬官方窗口逻辑**（`_build_step_text_window`，window=6）构造输入，
   一行代码都不改，确保可比
2. 用**动作重复**独立判定哪些窗口真的卡住（连续同一个 pyautogui 调用）
3. 两个监控器都跑，看分数能不能把卡住/正常分开
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WINDOW = 6


def read_traj(path: Path) -> list[dict]:
    """照搬官方 _read_jsonl：按 step_num 排序。"""
    steps = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    steps.sort(key=lambda d: int(d.get("step_num", 0)))
    return steps


def build_window(steps: list[dict], end_idx: int, window: int = WINDOW) -> str:
    """照搬官方 _build_step_text_window，逐字符一致。"""
    start = max(0, end_idx - (window - 1))
    chunks = []
    for i in range(start, end_idx + 1):
        s = steps[i]
        step_num = int(s.get("step_num", i + 1))
        chunks.append(f"Step {step_num}:\nResponse: {s.get('response', '')}\n"
                      f"Action: {s.get('action', '')}\n")
    return "\n".join(chunks).strip() + "\n"


def repeated_streak(steps: list[dict], end_idx: int, window: int = WINDOW) -> int:
    """末尾连续相同动作的步数——独立于任何标注的"卡住"判据。"""
    start = max(0, end_idx - (window - 1))
    codes = [steps[i].get("action", "") for i in range(start, end_idx + 1)]
    streak = 0
    for c in reversed(codes):
        if c == codes[-1]:
            streak += 1
        else:
            break
    return streak


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/evocua_feedback")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    models = {}
    for name, path in [("卡住", "models/stuck-detector"), ("里程碑", "models/milestone-detector")]:
        models[name] = (
            AutoTokenizer.from_pretrained(path),
            AutoModelForSequenceClassification.from_pretrained(path).to(args.device).eval(),
        )

    def score(name: str, text: str) -> float:
        tok, mdl = models[name]
        enc = tok(text, truncation=True, max_length=2048,
                  padding="max_length", return_tensors="pt").to(args.device)
        with torch.no_grad():
            return torch.softmax(mdl(**enc).logits, -1)[0, 1].item()

    trajs = sorted(Path(args.root).rglob("traj.jsonl"))
    print(f"找到 {len(trajs)} 条真实轨迹\n")

    print("=" * 92)
    print("  逐条轨迹：每一步的监控器打分")
    print("=" * 92)

    stuck_scores, normal_scores = [], []

    for tp in trajs:
        steps = read_traj(tp)
        rel = tp.relative_to(args.root).parent
        result_file = tp.parent / "result.txt"
        result = result_file.read_text().strip() if result_file.exists() else "?"

        print(f"\n▌ {rel}   共 {len(steps)} 步   result={result}")
        print(f"  {'步':>4}  {'连续同动作':>10}  {'卡住P':>11}  {'里程碑P':>11}   动作")

        for i in range(len(steps)):
            if i < WINDOW - 1:
                continue
            text = build_window(steps, i)
            streak = repeated_streak(steps, i)
            p_stuck = score("卡住", text)
            p_mile = score("里程碑", text)
            action = (steps[i].get("action") or "")[:38]

            mark = ""
            if streak >= 3:
                stuck_scores.append(p_stuck)
                mark = "  ← 连续重复"
            elif streak == 1:
                normal_scores.append(p_stuck)

            print(f"  {i + 1:>4}  {streak:>10}  {p_stuck:>11.6f}  {p_mile:>11.6f}   {action}{mark}")

    print()
    print("=" * 92)
    print("  汇总")
    print("=" * 92)
    print(f"  正常窗口（动作不重复）  n={len(normal_scores):<4} 均分 = {mean(normal_scores):.6f}")
    print(f"  卡住窗口（连续重复>=3） n={len(stuck_scores):<4} 均分 = {mean(stuck_scores):.6f}")
    if stuck_scores and normal_scores:
        print(f"  差值 = {mean(stuck_scores) - mean(normal_scores):+.6f}")
    else:
        print("  ⚠️ 这份数据里没有「连续重复 >=3 步」的窗口。")
        print("     8 条轨迹都太短或太顺，没有真正卡死的情形——跟 AgentNet 一样的问题。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
