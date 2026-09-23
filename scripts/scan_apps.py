"""扫出「应用显示名 -> 包名」的对照表，存进 `configs/apps.yaml`。

## 为什么要扫，而不是写死

"设置"对应哪个包，是**设备的事实**，不是模型能推出来的知识。所以这张表
必须存在。问题只在于**放在哪里**：

    放在提示词里  ->  模型不看屏幕，拿任务里的词去表里查，一步到位。
                      benchmark 退化成查表题。（踩过，见 envs/android.py）
    放在环境配置里 ->  模型只说"打开设置"，环境自己查。这才是对的。

放对环境之后，它就成了**设备相关数据**，换设备必须重扫——所以它属于
配置文件，不能进源码。

## 怎么扫（全程只用通用机制）

    1. 上滑打开应用抽屉，dump 无障碍树 -> 拿到所有应用名和它们的坐标
    2. 挨个点过去，读前台包名 -> 名字和包名就对上号了
    3. 存成 YAML

全程只有 `input swipe` / `input tap` / `uiautomator dump` / `dumpsys` 四样，
**在任何一个安卓设备上都能跑**，没有任何针对某台机器的假设。

## 用法

    python scripts/scan_apps.py                     # 扫，打印结果
    python scripts/scan_apps.py --out configs/apps.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guicascade.envs.android import AndroidEnv, _parse_ui_xml  # noqa: E402

# 抽屉里这些不是应用图标，是搜索框、设置入口之类，点了没有对应的包
_NOT_APPS = {"Search", "Google app", "Preferences", "Home", "Search apps, web and more"}


def foreground(env: AndroidEnv) -> str:
    out = env._run("shell", "dumpsys", "window", "displays").decode("utf-8", "replace")
    m = re.search(r"mCurrentFocus=Window\{[^}]*?\s([\w\.]+)/", out)
    return m.group(1) if m else ""


def launcher_packages(env: AndroidEnv) -> set[str]:
    """所有**带启动器入口**的包名。

    用来排除一类很常见的误判：不少 app 第一次启动会先弹权限框或登录页，
    这时读到的前台包名是对话框的宿主（`com.google.android.gms`、
    `permissioncontroller`），根本不是我们点开的那个 app。

    「是不是启动器应用」正好能把这批杂音滤掉，而且这个判据是系统给的、
    与设备无关——比自己猜"哪些包名看着像对话框"泛用得多。
    """
    out = env._run(
        "shell", "cmd", "package", "query-activities", "--brief",
        "-a", "android.intent.action.MAIN",
        "-c", "android.intent.category.LAUNCHER",
        timeout=60,
    ).decode("utf-8", "replace")

    pkgs: set[str] = set()
    for line in out.splitlines():
        line = line.strip()
        # 形如 com.android.settings/.Settings
        if "/" in line and " " not in line and "." in line.split("/")[0]:
            pkgs.add(line.split("/")[0])
    return pkgs


def open_drawer(env: AndroidEnv) -> None:
    env._run("shell", "input", "keyevent", "KEYCODE_HOME")
    time.sleep(1.5)
    env._run("shell", "input", "swipe", "540", "1600", "540", "400", "300")
    time.sleep(2.0)


def drawer_positions(env: AndroidEnv) -> dict[str, tuple[int, int]]:
    """当前抽屉里「应用名 -> 图标中心坐标」。

    每轮重新读，不缓存——上一步的操作可能让抽屉滚动，旧坐标就点空了。
    """
    env._run("shell", "uiautomator", "dump", "/sdcard/apps.xml", timeout=30)
    raw = env._run("shell", "cat", "/sdcard/apps.xml", timeout=30).decode("utf-8", "replace")
    out: dict[str, tuple[int, int]] = {}
    for e in _parse_ui_xml(raw[raw.find("<?xml"):]):
        label = e["text"] or e["desc"]
        if not label or label in _NOT_APPS:
            continue
        x1, y1, x2, y2 = e["bounds"]
        if x2 > x1 and y2 > y1:
            out.setdefault(label, ((x1 + x2) // 2, (y1 + y2) // 2))
    return out


_LAUNCH_TIMEOUT = 8.0


def wait_for_launcher_app(env: AndroidEnv, real: set[str]) -> str:
    """点完图标之后，轮询前台，等到一个**启动器应用**为止。

    为什么要轮询而不是等固定时长：Maps / 相机 / 电话这些重一点的 app 两秒
    根本起不来，固定等待会让一半的 app 扫不到。而且有些 app 首次启动先弹
    权限框，此时前台是对话框宿主——**继续等**，等它弹完了真正的界面才上来。

    等不到就返回空串（调用方跳过），不猜——猜错的名字比缺失更糟。
    """
    deadline = time.time() + _LAUNCH_TIMEOUT
    while time.time() < deadline:
        pkg = foreground(env)
        if pkg in real:
            return pkg
        time.sleep(0.7)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="emulator-5554")
    ap.add_argument("--out", default="", help="写到这个 YAML；留空只打印")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    env = AndroidEnv(serial=args.serial, capture_image=False)

    open_drawer(env)
    apps = drawer_positions(env)
    real = launcher_packages(env)
    print(f"系统报告 {len(real)} 个启动器入口；抽屉里发现 {len(apps)} 个候选应用。\n")

    mapping: dict[str, str] = {}
    for label in sorted(apps):
        if len(mapping) >= args.limit:
            break
        # 每轮重新 dump：抽屉可能因为上一步的操作滚动了，复用旧坐标会点空
        pos = drawer_positions(env).get(label)
        if pos is None:
            print(f"  ⚠️ {label:<14} -> 这一轮抽屉里没找到，跳过")
            continue

        env._run("shell", "input", "tap", str(pos[0]), str(pos[1]))
        pkg = wait_for_launcher_app(env, real)
        if pkg:
            mapping[label] = pkg
            print(f"  ✅ {label:<14} -> {pkg}")
        else:
            print(f"  ⚠️ {label:<14} -> 等了 {_LAUNCH_TIMEOUT:.0f}s 也没等到启动器应用，跳过")
        open_drawer(env)

    missed = sorted(set(real) - set(mapping.values()))
    if missed:
        print(f"\n  以下启动器应用没在抽屉里对上号（可能不在当前页、或被对话框挡了）：")
        for m in missed:
            print(f"     {m}")

    # 回桌面收尾
    env._run("shell", "input", "keyevent", "KEYCODE_HOME")

    print(f"\n共确认 {len(mapping)} 个应用。")
    if not args.out:
        print("（加 --out configs/apps.yaml 可以写进配置）")
        return 0

    # 手写 YAML，不用第三方库——这张表很扁，没必要为它引依赖
    lines = [
        "# 应用显示名 -> 包名。**这是设备相关的数据，不是源码。**",
        "#",
        "# 由 scripts/scan_apps.py 自动扫描生成，换设备/换镜像必须重扫：",
        "#     python scripts/scan_apps.py --out configs/apps.yaml",
        "#",
        "# ⚠️ 这张表**绝不能进提示词**。模型只负责说应用名，查表是环境的事。",
        "#    写进提示词的后果是模型不再看屏幕、直接查表，benchmark 退化成查表题。",
        "",
        "apps:",
    ]
    for k, v in sorted(mapping.items()):
        lines.append(f"  {k}: {v}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
