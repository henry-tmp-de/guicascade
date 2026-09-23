"""分级探针：监控器的分数会不会随「卡住程度」单调上升？

## 为什么要做这个

如果项目要用这个监控器，那**我们的轨迹必须是它能分辨出来的**。
也就是说：prompt 怎么设计、上下文怎么组织、轨迹文本长什么样，
**都得先拿它测过**，确认它看得见，再去设计。反过来做就是白干。

这个脚本用一个**受控的**分级测试集回答三件事：

  1. 分数随「卡住步数」单调上升吗？（能分辨"程度"吗）
  2. 中文轨迹和英文轨迹表现一样吗？（我们两条都要支持）
  3. 换哪种"卡住"的写法，它还认得出？（重复动作 vs 动作不同但都失败）

## 测试集构造

窗口固定 6 步。设卡住程度 N = 0..4：
  · 前 (6-N) 步是正常推进的（动作各不相同）
  · 后 N 步是卡住的

两种卡住写法各来一遍：
  · A 型 重复：后 N 步的动作代码**完全相同**，理由说的是"没反应，再试一次"
  · B 型 失败：后 N 步的动作**各不相同**，但理由都写着"没成功"

每种 × 中文 / 英文，共 2 × 5 × 2 = 20 条，正好是你说的规模。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# --------------------------------------------------------------------------
# 正常推进的步骤（动作各不相同，且在朝目标前进）
# --------------------------------------------------------------------------

NORMAL_EN = [
    ("I need to open the Settings app. The gear icon should be in the launcher.", "pyautogui.click(x=0.92, y=0.08)"),
    ("The Settings window is open. I'll look for the Display category in the sidebar.", "pyautogui.click(x=0.12, y=0.35)"),
    ("The Display panel is showing. I need to find the brightness slider.", "pyautogui.scroll(-4)"),
    ("I can see the brightness section now. I'll click on the slider handle.", "pyautogui.click(x=0.55, y=0.62)"),
    ("The slider is selected. I'll drag it to increase the brightness.", "pyautogui.dragTo(x=0.78, y=0.62)"),
    ("Brightness looks right now. I'll close the Settings window to finish.", "pyautogui.click(x=0.98, y=0.03)"),
]

NORMAL_ZH = [
    ("需要打开设置应用，齿轮图标应该在启动器里。", "pyautogui.click(x=0.92, y=0.08)"),
    ("设置窗口已经打开了，我去左侧栏找「显示」分类。", "pyautogui.click(x=0.12, y=0.35)"),
    ("显示面板出来了，需要找到亮度滑块。", "pyautogui.scroll(-4)"),
    ("看到亮度区域了，我点击滑块手柄。", "pyautogui.click(x=0.55, y=0.62)"),
    ("滑块选中了，向右拖动提高亮度。", "pyautogui.dragTo(x=0.78, y=0.62)"),
    ("亮度已经合适，关闭设置窗口结束。", "pyautogui.click(x=0.98, y=0.03)"),
]

# --------------------------------------------------------------------------
# 卡住的步骤，两种写法
# --------------------------------------------------------------------------

# A 型：同一个动作反复点，理由承认没反应
STUCK_REPEAT_EN = ("The screen still looks the same. Nothing happened, let me click it once more.", "pyautogui.click(x=0.55, y=0.62)")
STUCK_REPEAT_ZH = ("屏幕还是老样子，没有任何变化，我再点一次。", "pyautogui.click(x=0.55, y=0.62)")

# B 型：动作每次都不一样，但每一步都失败
STUCK_FAIL_EN = [
    ("That didn't work. Let me try clicking a bit to the left instead.", "pyautogui.click(x=0.48, y=0.62)"),
    ("Still not working. Maybe I should try the menu at the top.", "pyautogui.click(x=0.30, y=0.05)"),
    ("That wasn't it either. Let me try right-clicking the area.", "pyautogui.rightClick(x=0.55, y=0.62)"),
    ("Nothing is responding. I'll try pressing Escape and retrying.", "pyautogui.press('escape')"),
]
STUCK_FAIL_ZH = [
    ("这样不行，我往左边一点再点一次试试。", "pyautogui.click(x=0.48, y=0.62)"),
    ("还是没反应，也许该试试顶部的菜单。", "pyautogui.click(x=0.30, y=0.05)"),
    ("也不对，我右键点一下这块区域试试。", "pyautogui.rightClick(x=0.55, y=0.62)"),
    ("什么都没响应，按 Esc 重来一次。", "pyautogui.press('escape')"),
]


def build_case(severity: int, lang: str, mode: str) -> dict:
    """构造一条样本。severity = 末尾卡住的步数。"""
    normal = NORMAL_EN if lang == "en" else NORMAL_ZH
    n_normal = 6 - severity
    steps = list(normal[:n_normal])

    if mode == "repeat":
        stuck = STUCK_REPEAT_EN if lang == "en" else STUCK_REPEAT_ZH
        steps += [stuck] * severity
    else:  # fail
        pool = STUCK_FAIL_EN if lang == "en" else STUCK_FAIL_ZH
        steps += [pool[i % len(pool)] for i in range(severity)]

    return {"severity": severity, "lang": lang, "mode": mode, "steps": steps}


def render(case: dict) -> str:
    """官方推理格式。"""
    return "\n".join(
        f"Step {i}:\nResponse: {r}\nAction: {a}\n"
        for i, (r, a) in enumerate(case["steps"], start=1)
    )


def spearman(xs: list[float], ys: list[float]) -> float:
    """秩相关。1.0 = 完全单调上升，0 = 无关。"""
    n = len(xs)
    if n < 3:
        return float("nan")

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/stuck-detector")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(args.device).eval()

    def score(text: str) -> float:
        enc = tok(text, truncation=True, max_length=2048,
                  padding="max_length", return_tensors="pt").to(args.device)
        with torch.no_grad():
            return torch.softmax(model(**enc).logits, -1)[0, 1].item()

    cases = [
        build_case(sev, lang, mode)
        for mode in ("repeat", "fail")
        for lang in ("en", "zh")
        for sev in range(5)
    ]

    print("=" * 80)
    print("  分级探针：20 条受控样本（卡住步数 0~4 × 中英 × 两种卡住写法）")
    print("=" * 80)

    for mode, mode_name in [("repeat", "A型：同一个动作反复点"), ("fail", "B型：动作不同但都失败")]:
        for lang, lang_name in [("en", "英文"), ("zh", "中文")]:
            group = [c for c in cases if c["mode"] == mode and c["lang"] == lang]
            scores = [score(render(c)) for c in group]
            sevs = [c["severity"] for c in group]

            print(f"\n  【{mode_name} · {lang_name}】")
            print(f"    {'卡住步数':<10}{'P(卡住)':>14}")
            for s, p in zip(sevs, scores):
                bar = "█" * int(p * 40) if p > 0 else ""
                print(f"    {s:<10}{p:>14.6f}  {bar}")

            rho = spearman([float(s) for s in sevs], scores)
            print(f"    秩相关 (越高=越单调) = {rho:+.3f}")

    # ---- 汇总 ----
    print()
    print("=" * 80)
    print("  汇总：能不能分辨？")
    print("=" * 80)

    results = {}
    for mode in ("repeat", "fail"):
        for lang in ("en", "zh"):
            group = [c for c in cases if c["mode"] == mode and c["lang"] == lang]
            sc = {c["severity"]: score(render(c)) for c in group}
            results[f"{mode}_{lang}"] = sc

            not_stuck = sc[0]
            stuck4 = sc[4]
            sep = stuck4 - not_stuck
            flag = "✅ 能分开" if sep > 0.3 else ("🟡 有一点" if sep > 0.05 else "❌ 分不开")
            print(f"  {mode:<7}{lang:<4} 卡住0步={not_stuck:.6f}  卡住4步={stuck4:.6f}  "
                  f"差值={sep:+.6f}  {flag}")

    print()
    print("  中英对比（同一设置下两条分数该接近）：")
    for mode in ("repeat", "fail"):
        en = results[f"{mode}_en"]
        zh = results[f"{mode}_zh"]
        diffs = [abs(en[s] - zh[s]) for s in range(5)]
        print(f"    {mode:<7} 最大差异 = {max(diffs):.6f}   "
              f"{'✅ 语言无关' if max(diffs) < 0.05 else '⚠️ 语言有影响'}")

    print()
    print("=" * 80)
    print("  结论")
    print("=" * 80)
    best = max(
        ((k, spearman([0.0, 1, 2, 3, 4], list(v.values()))) for k, v in results.items()),
        key=lambda kv: kv[1] if kv[1] == kv[1] else -9,
    )
    print(f"  单调性最好的设置：{best[0]}，秩相关 = {best[1]:+.3f}")
    if best[1] > 0.9:
        print("  ✅ 监控器能分辨卡住的「程度」——分数随卡住步数单调上升。")
        print("     说明它确实抓到了信号，阈值可以按需要的灵敏度去切。")
    elif best[1] > 0.6:
        print("  🟡 有单调趋势但不干净，阈值不好定。")
    else:
        print("  ❌ 分数与卡住程度基本无关。")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
