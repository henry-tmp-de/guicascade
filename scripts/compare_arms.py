"""把三条臂跑出来的结果汇成一张对比表。

## 这张表要回答的问题

    全小模型  ──  便宜，但会卡住
    全大模型  ──  稳，但每步都贵
    级联      ──  到底站在两者之间的哪个位置？

## 三个数必须分开看，别合成一个"综合得分"

    success_rate     任务成没成（程序化判分，不含主观成分）
    escalation_rate  强模型占了多大比例 —— **这是省钱的直接来源**
    latency_model_s  模型侧耗时 —— **级联唯一能省的就是这一项**
    latency_env_s    环境侧耗时 —— 截图 + adb 往返，级联一分都省不掉

最后一条是这类工作最常见的坑：把模型侧和环境侧加在一起报一个"总耗时"，
级联的收益会被环境开销稀释成看不出来，结论就废了。所以这里**坚持分开报**，
而且环境侧那列在三条臂上应该基本相等——**如果不相等，说明实验有问题**
（大概率是某条臂的步数差太多），比结论本身更值得先看。

## 为什么也报"墙钟"

模型跑在 GPU 服务器上、环境在本机，两者是**串行**的，所以墙钟 ≈ 模型侧 +
环境侧 + 网络开销。它是最贴近"用户等了多久"的数，但它**不适合用来证明
级联有效**——网络抖动会把信号淹掉。放着是为了交叉验证，
不是为了当主指标。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

ARMS = [
    ("small", "全小模型", "2B 打满全场（下界基线）"),
    ("large", "全大模型", "8B 打满全场（上界基线）"),
    ("cascade", "级联", "2B 打底，重复 3 次就换 8B"),
]


@dataclass
class ArmResult:
    name: str
    label: str
    note: str
    runs: list[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def n_ok(self) -> int:
        return sum(r["success"] for r in self.runs)

    @property
    def success_rate(self) -> float:
        return self.n_ok / self.n if self.n else 0.0

    @property
    def steps(self) -> int:
        return sum(r["steps"] for r in self.runs)

    @property
    def model_s(self) -> float:
        """模型侧总耗时。**级联要省的就是这个数。**"""
        return sum(r["latency_model_s"] for r in self.runs)

    @property
    def env_s(self) -> float:
        """环境侧总耗时。级联省不掉，三条臂应当基本相等。"""
        return sum(r["latency_env_s"] for r in self.runs)

    @property
    def escalated(self) -> int:
        return sum(r["escalated"] for r in self.runs)

    @property
    def esc_rate(self) -> float:
        return self.escalated / self.steps if self.steps else 0.0


def load_arm(key: str, label: str, note: str, root: Path) -> ArmResult:
    arm = ArmResult(key, label, note)
    for rf in sorted(root.rglob("result.txt")):
        d = rf.parent
        traj = d / "traj.jsonl"
        rows = [json.loads(x) for x in traj.open(encoding="utf-8") if x.strip()] \
            if traj.exists() else []
        arm.runs.append({
            "task": d.name,
            "success": rf.read_text(encoding="utf-8").strip() == "1",
            "steps": len(rows),
            "escalated": sum(bool(r.get("escalated")) for r in rows),
            "latency_model_s": sum(float(r.get("latency_model_s", 0)) for r in rows),
            "latency_env_s": sum(float(r.get("latency_env_s", 0)) for r in rows),
            "actions": [r.get("action", "") for r in rows],
        })
    return arm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--prefix", default="arm_")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root

    arms = []
    for key, label, note in ARMS:
        d = root / f"{args.prefix}{key}"
        if not d.exists():
            print(f"  ⚠️ 缺少 {d}，跳过")
            continue
        arms.append(load_arm(key, label, note, d))

    if not arms:
        print("没有可对比的结果")
        return 1

    # ---- 主表 ----
    print()
    print("=" * 92)
    print("  三臂对照")
    print("=" * 92)
    print(f"  {'臂':<12}{'成功率':>10}{'总步数':>9}{'强模型占比':>12}"
          f"{'模型侧耗时':>12}{'环境侧耗时':>12}{'每步模型耗时':>14}")
    print("  " + "-" * 88)
    for a in arms:
        per_step = a.model_s / a.steps if a.steps else 0.0
        print(f"  {a.label:<12}{a.n_ok}/{a.n:<8}{a.steps:>9}{a.esc_rate*100:>11.1f}%"
              f"{a.model_s:>11.1f}s{a.env_s:>11.1f}s{per_step:>13.3f}s")
    print()

    # ---- 环境侧一致性检查（比结论本身更该先看）----
    envs = [a.env_s for a in arms if a.steps]
    if len(envs) >= 2:
        spread = (max(envs) - min(envs)) / max(envs) if max(envs) else 0
        verdict = "✅ 基本一致" if spread < 0.35 else "⚠️ 差异偏大，先查原因再下结论"
        print(f"  环境侧耗时离散度 {spread*100:.0f}% —— {verdict}")
        print("  （三条臂的环境侧本该基本相等；差异主要来自步数不同，")
        print("    步数差异又是模型行为差异的**结果**，不是环境变了）")
    print()

    # ---- 逐任务明细 ----
    print("=" * 92)
    print("  逐任务")
    print("=" * 92)
    tasks = sorted({r["task"] for a in arms for r in a.runs})
    print(f"  {'任务':<20}" + "".join(f"{a.label+' 结果/步数/升级':>26}" for a in arms))
    print("  " + "-" * 88)
    for t in tasks:
        line = f"  {t[:19]:<20}"
        for a in arms:
            r = next((x for x in a.runs if x["task"] == t), None)
            if r is None:
                line += f"{'—':>26}"
            else:
                mark = "✅" if r["success"] else "❌"
                cell = f"{mark} {r['steps']} 步 / 升级 {r['escalated']}"
                line += f"{cell:>26}"
        print(line)

    # ---- 级联的账：省了多少、代价是什么 ----
    casc = next((a for a in arms if a.name == "cascade"), None)
    small = next((a for a in arms if a.name == "small"), None)
    large = next((a for a in arms if a.name == "large"), None)
    if casc and small and large:
        print()
        print("=" * 92)
        print("  级联站在哪")
        print("=" * 92)
        print(f"  成功率    : 全小 {small.success_rate*100:.0f}%  ->  "
              f"级联 {casc.success_rate*100:.0f}%  ->  全大 {large.success_rate*100:.0f}%")
        if small.model_s:
            print(f"  模型侧耗时: 全小 {small.model_s:.1f}s  ->  级联 {casc.model_s:.1f}s  ->  "
                  f"全大 {large.model_s:.1f}s")
            saved = 1 - casc.model_s / large.model_s
            print(f"              相对全大模型省了 {saved*100:.0f}%")
        print(f"  强模型占比: 级联 {casc.esc_rate*100:.1f}%（全大是 100%，全小是 0%）")
        print()
        print("  ⚠️ 样本量小（每个任务只跑一次、模型有随机性），这张表是**趋势**不是定论。")
        print("     要下结论得加大重复次数（--repeat）并报方差。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
