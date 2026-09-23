"""在**当前这个设备、当前这一屏**上验证 harness 的两条硬承诺。

## 为什么需要这个脚本

"我们的框架是通用的"是一句很容易说出口、却很难证伪的话。这个脚本把它
拆成两条**可当场检验、换任何设备任何界面都能重跑**的断言：

  承诺一：模型看到的 [N]，点下去就落在第 N 个元素上。
  承诺二：点下去真的有效果（界面变了）。

两条都不是靠读代码能确认的——本项目的索引错位 bug 就是"代码看着很对"、
跑起来每一步都点错地方。所以必须在真屏上验。

## 用法

    python scripts/verify_harness.py                    # 只验承诺一（不碰设备）
    python scripts/verify_harness.py --tap "Gmail"      # 顺带验承诺二
    python scripts/verify_harness.py --tap-index 3      # 按序号点

换设备/换界面直接重跑。**这个脚本本身不含任何设备特定的东西**——
元素是从无障碍树现读的，标签是现找的。
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guicascade.envs.android import (  # noqa: E402
    AndroidEnv,
    actionable_elements,
    render_elements,
)


def foreground(env: AndroidEnv) -> str:
    out = env._run("shell", "dumpsys", "window", "displays").decode("utf-8", "replace")
    m = re.search(r"mCurrentFocus=Window\{[^}]*?\s([\w\.]+)/", out)
    return m.group(1) if m else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="emulator-5554")
    ap.add_argument("--tap", default="", help="按标签点一个元素（模糊匹配）")
    ap.add_argument("--tap-index", type=int, default=-1, help="按序号点一个元素")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    env = AndroidEnv(serial=args.serial, capture_image=False, max_elements=args.limit)

    raw = env.ui_elements(fresh=True)
    shown = actionable_elements(raw, limit=args.limit)

    print("=" * 84)
    print("  当前屏幕")
    print("=" * 84)
    print(f"  无障碍树原始节点 {len(raw)} 个 -> 过滤后可操作 {len(shown)} 个")
    print()
    for line in render_elements(shown).splitlines()[:14]:
        print("   ", line[:100])
    if len(shown) > 14:
        print(f"    ...（还有 {len(shown) - 14} 个）")

    # ---------------- 承诺一：序号 -> 落点 ----------------
    print()
    print("=" * 84)
    print("  承诺一：模型看到的 [N]，点下去落在第 N 个元素上")
    print("=" * 84)

    bad = []
    for i, e in enumerate(shown):
        x, y = e["_click_xy"]
        x1, y1, x2, y2 = e["bounds"]
        inside_self = x1 <= x <= x2 and y1 <= y <= y2
        # 允许落在"最近的可点击祖先"里——那是刻意的设计（见 _resolve_click）
        inside_anc = any(
            raw[a]["bounds"][0] <= x <= raw[a]["bounds"][2]
            and raw[a]["bounds"][1] <= y <= raw[a]["bounds"][3]
            for a in e["_ancestors"]
        )
        if not (inside_self or inside_anc):
            bad.append((i, e, x, y))

    if bad:
        print(f"  ❌ {len(bad)} 个元素的落点不在自己（或可点击祖先）范围内：")
        for i, e, x, y in bad[:5]:
            print(f"     [{i}] {e['text'] or e['desc']!r} bounds={e['bounds']} -> 点到 ({x},{y})")
        return 1
    print(f"  ✅ 全部 {len(shown)} 个元素的落点都落在自己或可点击祖先的范围内")

    # 额外：落点必须在屏幕内，否则 tap 会打到屏幕外
    w, h = env.screen_size()
    outside = [(i, e["_click_xy"]) for i, e in enumerate(shown)
               if not (0 <= e["_click_xy"][0] < w and 0 <= e["_click_xy"][1] < h)]
    if outside:
        print(f"  ❌ {len(outside)} 个落点在屏幕外（屏幕 {w}x{h}）：{outside[:3]}")
        return 1
    print(f"  ✅ 全部落点都在屏幕范围内（{w}x{h}）")

    # ---------------- 承诺二：点下去有反应 ----------------
    target = None
    if args.tap_index >= 0:
        if args.tap_index >= len(shown):
            print(f"\n  ❌ 序号 {args.tap_index} 越界（只有 {len(shown)} 个）")
            return 1
        target = (args.tap_index, shown[args.tap_index])
    elif args.tap:
        needle = args.tap.lower()
        for i, e in enumerate(shown):
            if needle in (e["text"] or e["desc"]).lower():
                target = (i, e)
                break
        if target is None:
            print(f"\n  ⚠️ 当前屏幕上找不到标签含 {args.tap!r} 的元素，跳过承诺二")
            print("     （这不算失败——换一屏再试）")
            return 0

    if target is None:
        print()
        print("  （用 --tap 或 --tap-index 可以顺带验证承诺二）")
        return 0

    idx, el = target
    before = foreground(env)
    x, y = el["_click_xy"]
    print()
    print("=" * 84)
    print("  承诺二：点下去真的有效果")
    print("=" * 84)
    print(f"  目标   : [{idx}] {el['text'] or el['desc']!r}")
    print(f"  落点   : ({x}, {y})")
    print(f"  点之前 : {before}")
    env._run("shell", "input", "tap", str(x), str(y))
    time.sleep(2.5)
    after = foreground(env)
    print(f"  点之后 : {after}")
    if after == before:
        print("  ⚠️ 前台没变——可能这个元素本来就不换界面（比如选中一个开关），")
        print("     也可能点空了。人工看一眼截图再判断。")
    else:
        print("  ✅ 界面确实响应了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
