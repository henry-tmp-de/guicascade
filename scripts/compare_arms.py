"""把三形态的结果并成一张对照表。

    python scripts/compare_arms.py
    python scripts/compare_arms.py --group ours

## 为什么要单独一张表

三份 `suite_*.jsonl` 各说各的，**单看任何一份都答不了"级联到底有没有用"**——
那是个**配对**问题：同一个任务，三种跑法分别什么结果。

所以这里按任务对齐，只报三件真正能下结论的事：

    只有级联成功   ← 级联的价值所在（小模型不行、大模型也不行、级联行）
    级联没保住     ← 级联的代价（单独能成的，级联反而丢了）
    三边都失败     ← 任务本身太难，这一栏的多少决定了结论有多可信

再补一个**升级率**：级联有多少步叫了强模型。这个数太低说明监控器没触发，
太高说明基本一直在用大模型——**两种情况都意味着"级联"名不副实**。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[r["name"]] = r
    return out


def mark(v) -> str:
    return "✅" if v else ("❌" if v is False else "—")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="androidworld")
    args = ap.parse_args()

    arms = {
        "2B": load(ROOT / "results" / f"suite_{args.group}_small.jsonl"),
        "8B": load(ROOT / "results" / f"suite_{args.group}_large.jsonl"),
        "级联": load(ROOT / "results" / f"suite_{args.group}_cascade_repeat.jsonl"),
    }

    names = sorted(set().union(*[set(a) for a in arms.values()]))
    if not names:
        print("没有结果文件。先跑 python scripts/run_arms.py")
        return 1

    print(f"\n{'='*86}")
    print(f"  三形态对照 · {args.group} · 共 {len(names)} 个任务")
    print(f"{'='*86}\n")
    print(f"  {'任务':<44} {'2B':<5} {'8B':<5} {'级联':<5}  说明")
    print(f"  {'-'*82}")

    only_cascade, lost_by_cascade, all_fail, all_ok = [], [], [], []
    for n in names:
        a = arms["2B"].get(n, {}).get("ok")
        b = arms["8B"].get(n, {}).get("ok")
        c = arms["级联"].get(n, {}).get("ok")
        note = ""
        if c and not a and not b:
            note = "只有级联成功"
            only_cascade.append(n)
        elif (a or b) and c is False:
            note = "级联没保住"
            lost_by_cascade.append(n)
        elif not a and not b and not c:
            all_fail.append(n)
        elif a and b and c:
            all_ok.append(n)
        print(f"  {n[3:][:44]:<44} {mark(a):<5} {mark(b):<5} {mark(c):<5}  {note}")

    print(f"\n  {'-'*82}")
    for label, arm in arms.items():
        done = [r for r in arm.values() if r.get("ok") is not None]
        ok = [r for r in done if r["ok"]]
        steps = sum(r.get("steps", 0) for r in arm.values())
        esc = sum(r.get("escalated", 0) for r in arm.values())
        esc_s = f"   升级率 {round(esc/steps*100)}%" if label == "级联" and steps else ""
        print(f"  {label:<4} 完成 {len(done):>3}/{len(names):<3}  "
              f"成功 {len(ok):>2}  成功率 "
              f"{(str(round(len(ok)/len(done)*100))+'%') if done else '—':>4}{esc_s}")

    print(f"\n  ★ 只有级联成功  {len(only_cascade):>2} 个   ← 级联的价值")
    for n in only_cascade:
        print(f"      {n[3:]}")
    if lost_by_cascade:
        print(f"  ⚠ 级联没保住    {len(lost_by_cascade):>2} 个   ← 级联的代价")
        for n in lost_by_cascade:
            print(f"      {n[3:]}")
    print(f"  · 三边都失败    {len(all_fail):>2} 个   ← 这一栏越多，上面的结论越不可信")
    print(f"  · 三边都成功    {len(all_ok):>2} 个")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
