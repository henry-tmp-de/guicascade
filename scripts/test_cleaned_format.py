"""决定性对照：把 response 里的 XML 脚手架洗掉，分数会不会变？

## 背景

`response` 字段装的是 agent `predict()` 的**原始返回**（已从 StepWise 自己的
`lib_run_single.py:417-428` 确认）。我们手里的 EvoCUAFeedback 用的是 EvoCUA 的
scaffold，所以原始返回里混着：

    </think>
    Action: Click on a cell in the middle of the spreadsheet...
    <tool_call>
    {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [500, 291]}}
    </tool_call>

而 StepWise 训练时用的是它自家 Qwen3-VL agent 的输出，原始文本形态可能不同。
**如果形态不同，我们就是在用分布外的文本测它，分数低不能怪模型。**

## 本脚本

同一批窗口、同一条判据，只改 `response` 的清洗方式，跑四种：

  raw      原样（对照组）
  strip    去掉 <tool_call>...</tool_call>、</think> 这类标签
  reason   只保留 Action: 之前的那段自然语言理由
  noxml    去掉所有 XML/JSON 花括号内容，只留纯文本

如果某种清洗方式让"卡住窗口"的分数明显抬头，说明是格式没对上；
如果四种都一样贴着 0，那就是模型本身的问题。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

WINDOW = 6

TOOLCALL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)
XML_TAG = re.compile(r"</?[a-zA-Z_][\w\-]*>")
JSON_BLOB = re.compile(r"\{[^{}]*\}", re.DOTALL)
ACTION_LINE = re.compile(r"^\s*Action\s*:", re.MULTILINE | re.IGNORECASE)


def clean(text: str, mode: str) -> str:
    if mode == "raw":
        return text
    if mode == "strip":
        return XML_TAG.sub("", TOOLCALL_BLOCK.sub("", text)).strip()
    if mode == "reason":
        # 只保留第一个 "Action:" 之前的自然语言部分
        m = ACTION_LINE.search(text)
        head = text[: m.start()] if m else text
        return XML_TAG.sub("", TOOLCALL_BLOCK.sub("", head)).strip()
    if mode == "noxml":
        t = TOOLCALL_BLOCK.sub("", text)
        t = XML_TAG.sub("", t)
        t = JSON_BLOB.sub("", t)
        return re.sub(r"\n{2,}", "\n", t).strip()
    raise ValueError(mode)


def build_window(steps: list[dict], end_idx: int, mode: str, window: int = WINDOW) -> str:
    start = max(0, end_idx - (window - 1))
    chunks = [
        f"Step {int(steps[i].get('step_num', i + 1))}:\n"
        f"Response: {clean(steps[i].get('response', ''), mode)}\n"
        f"Action: {steps[i].get('action', '')}\n"
        for i in range(start, end_idx + 1)
    ]
    return "\n".join(chunks).strip() + "\n"


def repeat_streak(actions: list[str], end_idx: int, window: int = WINDOW) -> int:
    start = max(0, end_idx - (window - 1))
    codes = actions[start : end_idx + 1]
    n = 0
    for c in reversed(codes):
        if c == codes[-1]:
            n += 1
        else:
            break
    return n


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/evocua_feedback")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("models/stuck-detector")
    mdl = AutoModelForSequenceClassification.from_pretrained("models/stuck-detector")
    mdl = mdl.to(args.device).eval()

    def score(texts: list[str]) -> list[float]:
        out = []
        for i in range(0, len(texts), 32):
            enc = tok(texts[i : i + 32], truncation=True, max_length=2048,
                      padding="max_length", return_tensors="pt").to(args.device)
            with torch.no_grad():
                out.extend(torch.softmax(mdl(**enc).logits, -1)[:, 1].tolist())
        return out

    trajs = []
    for tp in sorted(Path(args.root).rglob("traj.jsonl")):
        steps = [json.loads(l) for l in tp.open(encoding="utf-8") if l.strip()]
        steps.sort(key=lambda d: int(d.get("step_num", 0)))
        trajs.append((tp.relative_to(args.root).parent, steps))

    print("=" * 86)
    print("  清洗方式对分数的影响")
    print("=" * 86)
    print(f"  {'清洗方式':<12}{'卡住窗口均分':>16}{'正常窗口均分':>16}{'卡住最大':>14}{'AUC':>10}")
    print("  " + "-" * 72)

    for mode in ["raw", "strip", "reason", "noxml"]:
        pos, neg = [], []
        for _, steps in trajs:
            acts = [s.get("action", "") for s in steps]
            for i in range(WINDOW - 1, len(steps)):
                txt = build_window(steps, i, mode)
                (pos if repeat_streak(acts, i) >= 2 else neg).append(txt)
        if not pos or not neg:
            continue
        sp, sn = score(pos), score(neg)
        print(f"  {mode:<12}{sum(sp)/len(sp):>16.6f}{sum(sn)/len(sn):>16.6f}"
              f"{max(sp):>14.6f}{auc(sp, sn):>10.4f}")

    print()
    print("=" * 86)
    print("  样本内容对照（看清洗到底改了什么）")
    print("=" * 86)
    if trajs:
        steps = trajs[0][1]
        i = min(WINDOW - 1, len(steps) - 1)
        for mode in ["raw", "reason", "noxml"]:
            t = build_window(steps, i, mode)
            first = t.split("Response:", 1)[-1][:220]
            print(f"\n  【{mode}】首 220 字符：")
            print(f"    {first!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
