"""查清楚：官方监控器为什么打不出分？

三个待验证的假设，本脚本一次全测：

  H1  是我们的代码写错了
       -> 直接跑官方 `stuck_detector.py` 的原始类，和我们的结果对照

  H2  推理格式和训练格式对不上
       -> 官方训练脚本 `build_stuck_dataset.py` 用的是 `Step N:\\n{response}`，
          推理脚本 `stuck_detector.py` 用的是 `Step N:\\nResponse: ..\\nAction: ..`。
          两个格式各喂一遍，看分数是否变化

  H3  训练数据只有英文，模型对非英文没有泛化
       -> 在"正确的那个格式"下，比较英文轨迹 / 中文 / 乱码的分数

判读标准：**不看绝对分数，看"卡住的轨迹"和"正常的轨迹"能不能被分开。**
一个能用的监控器，卡住样本的分数必须显著高于正常样本；如果两者一样，
不管分数是多少，这个监控器在级联里都是废的。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 内容样本：卡住 vs 正常，用 OSWorld 风格的 pyautogui 动作（训练数据的真实形态）
# --------------------------------------------------------------------------

STUCK_REASONINGS = [
    "I need to click the save button to save the file.",
    "The file doesn't seem to have been saved. Let me click the save button again.",
    "Still nothing happened. I'll try clicking save once more.",
    "Maybe the click didn't register. Clicking save again.",
    "Let me click the save button again to make sure.",
    "Clicking save.",
]
STUCK_ACTIONS = ["pyautogui.click(1200, 45)"] * 6

NORMAL_REASONINGS = [
    "I need to open the browser first to search for the answer.",
    "The browser is open. I'll click on the address bar to type a search query.",
    "Now I'll type the search query into the address bar.",
    "The search results are showing. I'll click on the first result link.",
    "The page has loaded. I'll scroll down to read the relevant section.",
    "I found the information needed. Now I'll navigate back to the original tab.",
]
NORMAL_ACTIONS = [
    "pyautogui.click(30, 750)",
    "pyautogui.click(600, 80)",
    "pyautogui.typewrite('weather tomorrow')",
    "pyautogui.click(400, 320)",
    "pyautogui.scroll(-5)",
    "pyautogui.hotkey('ctrl', 'tab')",
]


def fmt_training(reasonings: list[str], actions: list[str]) -> str:
    """训练脚本的格式（build_stuck_dataset.py）。

    那边只写 `Step {n}:\\n{response}`，response 是 agent 的完整输出，
    也就是"理由 + Action 行"拼在一起。
    """
    chunks = []
    for i, (r, a) in enumerate(zip(reasonings, actions), start=1):
        chunks.append(f"Step {i}:\n{r}\nAction: {a}")
    return "\n".join(chunks)


def fmt_runtime(reasonings: list[str], actions: list[str]) -> str:
    """推理脚本的格式（stuck_detector.py）—— 多加了 Response:/Action: 标签。"""
    chunks = []
    for i, (r, a) in enumerate(zip(reasonings, actions), start=1):
        chunks.append(f"Step {i}:\nResponse: {r}\nAction: {a}\n")
    return "\n".join(chunks)


def fmt_reasoning_only(reasonings: list[str], actions: list[str]) -> str:
    """训练格式的另一种可能解读：response 里只有理由，不含动作。"""
    chunks = []
    for i, r in enumerate(reasonings, start=1):
        chunks.append(f"Step {i}:\n{r}")
    return "\n".join(chunks)


FORMATS = {
    "训练格式(理由+Action)": fmt_training,
    "推理格式(Response/Action)": fmt_runtime,
    "训练格式(只有理由)": fmt_reasoning_only,
}

# H3：在同一个格式下比较不同语言/内容
LANG_SAMPLES = {
    "英文GUI轨迹(正常)": fmt_runtime(NORMAL_REASONINGS, NORMAL_ACTIONS),
    "中文GUI轨迹": "".join(
        f"Step {i+1}:\nResponse: 我需要点击保存按钮来保存文件\nAction: pyautogui.click(1200, 45)\n\n"
        for i in range(6)
    ),
    "乱码": "AAAAAAAAAA" * 50,
    "空字符串": "",
    "纯英文散文": "The quick brown fox jumps over the lazy dog. " * 20,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/stuck-detector")
    ap.add_argument("--official-code", default="", help="官方 stuck_detector.py 的路径（测 H1）")
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("=" * 76)
    print("  H1：官方原始代码 vs 我们的复现")
    print("=" * 76)

    if args.official_code and Path(args.official_code).exists():
        sys.path.insert(0, str(Path(args.official_code).parent))
        try:
            from stuck_detector import StuckDetector  # type: ignore

            det = StuckDetector(model_path=args.model, device="cpu")
            for label, (rs, ac) in [
                ("卡住轨迹", (STUCK_REASONINGS, STUCK_ACTIONS)),
                ("正常轨迹", (NORMAL_REASONINGS, NORMAL_ACTIONS)),
            ]:
                _, prob, _ = det.check_if_stuck(rs, ac, current_step=6, max_history_steps=6)
                print(f"  官方代码 · {label:<14} P(卡住) = {prob:.6f}")
        except Exception as e:  # noqa: BLE001
            print(f"  官方代码运行失败：{type(e).__name__}: {e}")
    else:
        print("  （未提供 --official-code，跳过）")

    # ---- 我们自己的路径，用于对照 ----
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()

    def score(text: str) -> float:
        inputs = tokenizer(
            text, padding="max_length", truncation=True,
            max_length=args.max_length, return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.softmax(logits, dim=-1)[0, 1].item()

    print()
    for label, (rs, ac) in [
        ("卡住轨迹", (STUCK_REASONINGS, STUCK_ACTIONS)),
        ("正常轨迹", (NORMAL_REASONINGS, NORMAL_ACTIONS)),
    ]:
        print(f"  我们复现 · {label:<14} P(卡住) = {score(fmt_runtime(rs, ac)):.6f}")

    print()
    print("=" * 76)
    print("  H2：格式影响 —— 同一对内容，三种格式各喂一遍")
    print("=" * 76)
    print(f"  {'格式':<26}{'卡住':>12}{'正常':>12}{'差值':>12}  可分离?")
    print("  " + "-" * 72)

    for name, fmt in FORMATS.items():
        p_stuck = score(fmt(STUCK_REASONINGS, STUCK_ACTIONS))
        p_normal = score(fmt(NORMAL_REASONINGS, NORMAL_ACTIONS))
        gap = p_stuck - p_normal
        ok = "✅" if gap > 0.1 else ("🟡" if gap > 0.01 else "❌ 分不开")
        print(f"  {name:<26}{p_stuck:>12.6f}{p_normal:>12.6f}{gap:>12.6f}  {ok}")

    best_fmt = max(
        FORMATS.values(),
        key=lambda f: score(f(STUCK_REASONINGS, STUCK_ACTIONS))
        - score(f(NORMAL_REASONINGS, NORMAL_ACTIONS)),
    )

    print()
    print("=" * 76)
    print("  H3：语言/内容影响 —— 在推理格式下比较")
    print("=" * 76)
    for label, text in LANG_SAMPLES.items():
        print(f"  {label:<22} P(卡住) = {score(text):.6f}")

    print()
    print("=" * 76)
    print("  判读")
    print("=" * 76)
    print("  · 如果 H2 里三种格式都分不开卡住/正常 -> 不是格式问题，模型本身对目标概念无判别力")
    print("  · 如果某一种格式能分开 -> 是官方训练/推理格式不一致导致的，改格式就能救")
    print("  · 如果中文/乱码分数显著高于英文轨迹 -> 模型把'域外输入'当成了正类，")
    print("    说明它在训练分布内是退化的（对英文轨迹一律判负）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
