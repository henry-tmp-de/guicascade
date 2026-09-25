"""从 AndroidWorld 的名字表生成 `configs/apps_aw.yaml`（应用名 -> 包名）。

## 为什么需要它

`configs/apps.yaml` 是 `scan_apps.py` 扫出来的，而那次扫描发生在**官方
AndroidWorld 应用装上去之前**（9月23）。结果是设备上 34 个启动器应用里，
有 **17 个根本没登记** —— 恰好是官方任务要用的那批：

    com.dimowner.audiorecorder   录音机     AudioRecorderRecordAudio
    com.flauschcode.broccoli     菜谱       RecipeAddSingleRecipe ...
    org.videolan.vlc             VLC        VlcCreatePlaylist ...
    net.gsantner.markor          Markor     ...
    org.tasks / net.osmand / de.dennisguse.opentracks / ...

后果不是"少几个名字"这么轻：模型说 `open_app(app_name='Audio Recorder')`
→ 环境查表失败抛异常 → 异常被 `env.step` 吞进 `StepResult.error` →
**而那个 error 从来没进过给模型的提示词**（见 prompts.render_episode）→
模型不知道自己错了，看到一模一样的屏幕，下一轮说一模一样的话。
整条轨迹就在这一步空转，最后被记成"模型不行"。

## 为什么不直接改 apps.yaml

那份文件里有**手工维护的中文别名**和几条带解释的注释（比如"这台镜像上
没装计算器，别往里加"）。生成器会把那些冲掉。所以分开：

    configs/apps.yaml      手工 + 扫描，**优先**
    configs/apps_aw.yaml   本脚本生成，只补不覆盖

`load_apps()` 按这个顺序合并，前者已有的键不会被后者改写。

## 表的来源

AndroidWorld 自己的 `_PATTERN_TO_ACTIVITY`（正则 -> `包/Activity`）。
借它而不是自己写一份映射，理由和 `task_sets._app_to_package` 一样：
**那种表迟早和它的版本对不上。**

每条正则按 `|` 拆成若干别名分别登记（`google photos|gphotos|photos` 三个
名字都能查到同一个包），并且**只登记设备上真的装了的包** —— 表里有几十个
这台机器没有的应用（facebook / whatsapp / spotify…），登记它们只会让
模型以为可以打开一个不存在的应用。

## 用法

    python scripts/gen_apps.py                      # 生成到 configs/apps_aw.yaml
    python scripts/gen_apps.py --out /tmp/x.yaml
    python scripts/gen_apps.py --dry-run            # 只打印不写
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _sqlite_fts import ensure_fts  # noqa: E402

ensure_fts()

from guicascade.envs.android import find_adb, load_apps  # noqa: E402

AW_PATH = Path("D:/学习/code/_aw")

HEADER = """\
# 应用名 -> 包名。**本文件由 scripts/gen_apps.py 自动生成，不要手改。**
#
# 来源：AndroidWorld 的 `_PATTERN_TO_ACTIVITY`，并且只保留**设备上真装了**的包。
# 手写的映射放 configs/apps.yaml —— 那份优先，本文件只补它没有的键。
#
# 为什么要这张表、为什么它绝不能进提示词：见 configs/apps.yaml 开头的说明。
#
# 重新生成：python scripts/gen_apps.py
"""


def launcher_packages(adb: str, serial: str) -> set[str]:
    """设备上带启动器入口的包名。

    **必须有这个过滤。** AndroidWorld 的表里有 facebook / whatsapp / spotify
    这几十个这台机器根本没有的应用；把它们登记进去，模型会去打开一个不存在
    的应用，然后在那一步空转 —— 那正是我们要修的病，不能反过来制造它。
    """
    p = subprocess.run(
        [adb, "-s", serial, "shell", "cmd", "package", "query-activities",
         "-a", "android.intent.action.MAIN",
         "-c", "android.intent.category.LAUNCHER"],
        capture_output=True, timeout=90,
    )
    txt = p.stdout.decode("utf-8", "replace")
    return {
        ln.strip().split("=", 1)[1].strip()
        for ln in txt.splitlines()
        if ln.strip().startswith("packageName=")
    }


def build_table(installed: set[str]) -> dict[str, str]:
    """AndroidWorld 正则表 -> {别名: 包名}，只留设备上有的。"""
    if str(AW_PATH) not in sys.path:
        sys.path.insert(0, str(AW_PATH))
    from android_world.env.adb_utils import _PATTERN_TO_ACTIVITY  # noqa: PLC0415

    out: dict[str, str] = {}
    for pattern, activity in _PATTERN_TO_ACTIVITY.items():
        pkg = activity.split("/")[0].strip()
        if pkg not in installed:
            continue
        for alias in pattern.split("|"):
            alias = alias.strip()
            if alias:
                out[alias] = pkg
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="emulator-5554")
    ap.add_argument("--out", default=str(ROOT / "configs" / "apps_aw.yaml"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    adb = find_adb()
    installed = launcher_packages(adb, args.serial)
    print(f"设备上带启动器入口的包：{len(installed)} 个")

    table = build_table(installed)
    print(f"AndroidWorld 表里、且本机真实存在的映射：{len(table)} 条")

    curated = load_apps(str(ROOT / "configs" / "apps.yaml"))
    print(f"手工表 configs/apps.yaml：{len(curated)} 条（优先，不会被覆盖）")

    # 手工表已有的键不重复写 —— 两份文件里同一个键含义不同时，以手工那份为准
    merged = {k: v for k, v in sorted(table.items()) if k not in curated}
    print(f"本文件写入：{len(merged)} 条")

    dup = {k: (v, curated[k]) for k, v in table.items()
           if k in curated and curated[k] != v}
    if dup:
        print(f"⚠️  与手工表冲突 {len(dup)} 条（以手工表为准）：{dup}")

    lines = [HEADER, "apps:"]
    for k, v in merged.items():
        lines.append(f"  {k}: {v}")

    text = "\n".join(lines) + "\n"
    if args.dry_run:
        print("\n--- dry-run，未写文件 ---")
        print(text[:1200])
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\n✅ 已写入 {out}")

    # 立刻用合并后的表验一遍：这才是这条链路的真验收
    combined = load_apps()
    print(f"load_apps() 合并后共 {len(combined)} 条")
    env_index = {str(k).lower(): v for k, v in combined.items()}
    for probe in ("Audio Recorder", "Broccoli", "VLC", "Markor", "Tasks",
                  "OsmAnd", "Clock", "设置"):
        print(f"  {probe!r:18} -> {env_index.get(probe.lower(), '❌ 查不到')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
