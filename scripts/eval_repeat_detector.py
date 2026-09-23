"""在真实安卓轨迹上扫「重复几次算卡住」，并给出诚实的评测口径。

## 先想清楚拿什么当真值

最容易犯的错，是拿"窗口末尾连续同一动作 >= 3 步"当真值，再去评一个
"连续同一动作 >= 3 步就报警"的检测器——**那是同义反复，AUC 必然接近 1.0，
但它什么都没证明。**

所以这里的真值只有一个：**`result.txt`，任务到底成没成**。
那是 `dumpsys` 读设备真实状态程序化判出来的，和检测器完全独立。

## 还有一个更容易忽略的坑

**"重复"不等于"卡住"。** 在本项目自采的数据里，动作重复在**成功**的轨迹里
出现得一点不比失败的少：

    成功  open_app(deskclock) x8    第1步就打开了，后面7步纯属不会收尾
    失败  click(index=7)      x8    真的在乱点

两条轨迹的动作序列在检测器眼里**一模一样**。差别在于"目标达成没有"，
而那需要看任务，卡住检测器看不到。

结论：**用 AUC 评这个检测器是用错尺子。** 它不是分类器，是升级触发器。
该问的是两个问题：

    1. 它能不能在轨迹废掉之前把大模型叫起来？   -> 失败轨迹的命中率 + 命中时机
    2. 每次误触发要花多少冤枉钱？               -> 成功轨迹上的触发率

这两个数才是三臂对照里真正影响成败的。本脚本就报这两个。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guicascade.monitors.repeat import RepeatMonitor  # noqa: E402


@dataclass
class MiniStep:
    """`traj.jsonl` 一行的最小还原，只带监控器需要的两样东西。"""

    step_num: int
    action: str
    reason: str

    @property
    def decision(self):
        return self


def load_groups(root: Path) -> dict[str, list[dict]]:
    """按 `results/` 下的顶层目录分组。

    **分组很要紧。** 这些目录是不同时期的采集：早的那几批跑在解析器和
    环境还没修好的版本上（最明显的是全部顶到 10 步上限），晚的那批
    （`final_*`）才是当前系统。混在一起报一个总数，等于把两个分布
    搅成一杯说不清的糊状物。
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for traj in sorted(root.rglob("traj.jsonl")):
        rows = [json.loads(x) for x in traj.open(encoding="utf-8") if x.strip()]
        if len(rows) < 2:
            continue
        rf = traj.parent / "result.txt"
        ok = rf.exists() and rf.read_text(encoding="utf-8").strip() == "1"
        rel = traj.relative_to(root)
        groups[rel.parts[0]].append({
            "name": f"{rel.parts[-2] if len(rel.parts) > 2 else rel.parts[1]}/{rel.parts[-2]}",
            "path": rel.as_posix(),
            "steps": [MiniStep(r.get("step_num", i + 1), r.get("action", ""), r.get("response", ""))
                      for i, r in enumerate(rows)],
            "success": ok,
        })
    return dict(groups)


def simulate(trajs: list[dict], mon: RepeatMonitor) -> dict:
    """把监控器在每条轨迹上**逐步**跑一遍，模拟真实路由时它会看到什么。

    注意模拟的时序：决定第 t+1 步时，监控器只能看到前 t 步。
    拿整条轨迹去打分再回头说"它早就该报警了"，是**事后诸葛**——
    上线时看不到未来的步骤。
    """
    fired_trajs = 0
    points = 0
    fires = 0
    first_fire: list[int] = []
    rows = []

    for tr in trajs:
        steps = tr["steps"]
        hit_at = None
        n_fires = 0
        n_points = 0
        for t in range(1, len(steps)):          # 决定第 t+1 步
            n_points += 1
            if mon.score(None, steps[:t]) >= mon.threshold:
                n_fires += 1
                if hit_at is None:
                    hit_at = t
        points += n_points
        fires += n_fires
        if hit_at is not None:
            fired_trajs += 1
            first_fire.append(hit_at)
        rows.append({**tr, "fired": hit_at is not None, "first_fire": hit_at,
                     "n_fires": n_fires, "n_points": n_points,
                     "finished": any("finish" in str(s.action) for s in steps)})

    return {
        "rows": rows,
        "fired_trajs": fired_trajs,
        "fires": fires,
        "points": points,
        "first_fire": first_fire,
    }


def summarize(trajs: list[dict], mon: RepeatMonitor) -> dict:
    """两个真正重要的数：抓到了多少失败、误伤了多少成功。"""
    r = simulate(trajs, mon)
    fails = [x for x in r["rows"] if not x["success"]]
    oks = [x for x in r["rows"] if x["success"]]
    caught = sum(x["fired"] for x in fails)
    noise = sum(x["fired"] for x in oks)
    # 分开数：已经正常收尾还被升级的才算真浪费；那些"重复到撞上限、
    # 全靠判分兜住"的轨迹本来就是有病的，升级它们不算误伤。
    real_noise = sum(x["fired"] and x["finished"] for x in oks)
    # 第一次报警时还剩多少步可救
    left = [len(x["steps"]) - x["first_fire"] for x in r["rows"] if x["first_fire"]]
    return {
        "n": len(r["rows"]),
        "n_fail": len(fails),
        "n_ok": len(oks),
        "caught": caught,
        "noise": noise,
        "real_noise": real_noise,
        "esc_rate": r["fires"] / r["points"] if r["points"] else 0.0,
        "mean_left": sum(left) / len(left) if left else 0.0,
        "rows": r["rows"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--mode", default=None, help="只跑某一种模式")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    groups = load_groups(root)

    print("=" * 90)
    print("  数据分组（各批采集的代码版本不同，必须分开看）")
    print("=" * 90)
    for g, trs in sorted(groups.items()):
        n_ok = sum(t["success"] for t in trs)
        lens = sorted(len(t["steps"]) for t in trs)
        print(f"  {g:<14} {len(trs):>2} 条   成功 {n_ok}/{len(trs)}   步数 {lens}")

    # ---- 扫描 ----
    print()
    print("=" * 90)
    print("  扫描：重复几次算卡住（真值 = result.txt，与检测器独立）")
    print("=" * 90)

    modes = [args.mode] if args.mode else ["streak", "count"]
    for mode in modes:
        print()
        print(f"  【模式 {mode}】" + ("  末尾连击" if mode == "streak" else "  窗口内出现次数（window=6）"))
        print(f"  {'N':>3} {'组':<14}{'失败命中':>10}{'成功触发':>10}{'其中真误伤':>12}"
              f"{'升级率':>9}{'剩余步数':>10}")
        print("  " + "-" * 94)
        for g, trs in sorted(groups.items()):
            for n in (2, 3, 4, 5):
                mon = RepeatMonitor(mode=mode, min_repeats=n)
                s = summarize(trs, mon)
                caught = f"{s['caught']}/{s['n_fail']}" if s["n_fail"] else "—"
                noise = f"{s['noise']}/{s['n_ok']}" if s["n_ok"] else "—"
                print(f"  {n:>3} {g:<14}{caught:>10}{noise:>10}{s['real_noise']:>12}"
                      f"{s['esc_rate']*100:>8.1f}%{s['mean_left']:>10.1f}")
        print()

    # ---- 选定 N 之后的逐条明细 ----
    print("=" * 90)
    print("  逐条明细（默认 N=3 / 连击）")
    print("=" * 90)
    for g, trs in sorted(groups.items()):
        mon = RepeatMonitor(mode="streak", min_repeats=3)
        s = summarize(trs, mon)
        print(f"\n  [{g}]")
        print(f"  {'轨迹':<26}{'结果':>6}{'步数':>5}{'首次报警':>9}{'报警次数':>9}   判定")
        print("  " + "-" * 72)
        for x in s["rows"]:
            ff = x["first_fire"] if x["first_fire"] else "-"
            if x["success"] and x["fired"] and not x["finished"]:
                # 这条"成功"是虚的：模型一遍遍重复，从来没调 finish，
                # 只因为目标 app 恰好开着、又撞上了步数上限，判分才通过。
                # 升级到这里**不是浪费**——大模型看一眼就会收尾，
                # 反而可能比原地空转更省。
                verdict = "空转后判分通过（升级=提前收尾，是收益）"
            elif x["success"] and x["fired"]:
                verdict = "真误伤（已正常收尾，白升级）"
            elif x["success"]:
                verdict = "正确不动"
            elif x["fired"]:
                verdict = f"抓到（还剩 {len(x['steps']) - x['first_fire']} 步可救）"
            else:
                verdict = "漏掉"
            print(f"  {x['name'][:25]:<26}{'✅' if x['success'] else '❌':>6}"
                  f"{len(x['steps']):>5}{str(ff):>9}{x['n_fires']:>9}   {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
