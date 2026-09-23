"""把喂给卡住监控器的**原始文件**和**实际输入文本**原样打印出来，供人工核验。

不做任何加工：文件路径、每行的原始 JSON、以及最终拼出来的那一段文本，
全部照原样输出。目的是让人能自己判断——数据本身有没有问题。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WINDOW = 6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/evocua_feedback/evocua-32b/"
                                     "1-Task-da52d699-e8d2-4dc5-9191-a2199e0b6a9b/traj.jsonl")
    ap.add_argument("--window-end", type=int, default=7,
                    help="要展开的窗口结束步（1-based）")
    args = ap.parse_args()

    path = Path(args.file)
    print("=" * 90)
    print("  一、文件本身")
    print("=" * 90)
    print(f"  路径: {path}")
    print(f"  大小: {path.stat().st_size:,} 字节")
    print(f"  行数: {sum(1 for _ in path.open(encoding='utf-8'))}")
    rd = path.parent / "result.txt"
    if rd.exists():
        print(f"  result.txt: {rd.read_text().strip()!r}")
    print()

    steps = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line))
    steps.sort(key=lambda d: int(d.get("step_num", 0)))

    print("=" * 90)
    print("  二、每一行的原始字段（response 太长，这里折叠；后面单独展开一个）")
    print("=" * 90)
    for s in steps:
        print(f"  step_num={s.get('step_num'):<3} "
              f"action={s.get('action', '')[:44]!r:<50} "
              f"response长度={len(s.get('response', '')):<5} "
              f"reward={s.get('reward')} done={s.get('done')}")
        print(f"         其它字段: {[k for k in s if k not in ('step_num','action','response','reward','done')]}")

    print()
    print("=" * 90)
    print(f"  三、第 {args.window_end} 步那一行的 response 全文（监控器真正读的就是它）")
    print("=" * 90)
    target = next((s for s in steps if int(s.get("step_num", 0)) == args.window_end), None)
    if target:
        print(f"  action = {target.get('action')!r}")
        print(f"  response 全文（{len(target.get('response',''))} 字符）：")
        print("  " + "-" * 86)
        for line in (target.get("response") or "").splitlines():
            print(f"  | {line}")
        print("  " + "-" * 86)

    print()
    print("=" * 90)
    print(f"  四、拼出来喂给监控器的完整文本（照搬官方 _build_step_text_window）")
    print("=" * 90)
    end = args.window_end - 1
    start = max(0, end - (WINDOW - 1))
    chunks = [
        f"Step {int(steps[i].get('step_num', i + 1))}:\n"
        f"Response: {steps[i].get('response', '')}\n"
        f"Action: {steps[i].get('action', '')}\n"
        for i in range(start, end + 1)
    ]
    text = "\n".join(chunks).strip() + "\n"
    print(f"  覆盖第 {start + 1} ~ {end + 1} 步，共 {len(text)} 字符")
    print("  " + "-" * 86)
    for line in text.splitlines():
        print(f"  | {line[:150]}{'…' if len(line) > 150 else ''}")
    print("  " + "-" * 86)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
