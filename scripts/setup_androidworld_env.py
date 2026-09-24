"""把 AndroidWorld 需要的 app 装到我们自己的模拟器上。

## 为什么需要这个

AndroidWorld 的任务分两类：

  1. **纯设备状态的**（开关 WiFi/蓝牙/亮度）—— 判分读系统开关，不需要预置数据
  2. **依赖它自带 app 和数据的** —— markor（17 个任务）、broccoli（13 个）、
     pro expense（9 个）…… 还有 `task.html` 这类预置文件

第二类的前置状态由任务的 `initialize_task()` 从 **app 快照** 还原，而快照来自
它**官方文档里让你自己建的那种 AVD**——不是下载来的镜像，是装完 app 之后
由它的 setup 脚本现场生成的。

我们之前的 AVD 只有系统自带 app，所以那类任务全部失败，而且**失败是静默的**：
`initialize_task` 只打一行 warning 就继续跑，Agent 去找一个不存在的文件，
必然失败，然后这笔账记到模型头上。

这个脚本补上"装 app"这一步。

## 为什么不用它自己的 run.py --perform_emulator_setup

那条路要把它的整套环境（`-grpc 8554` 的 accessibility forwarder、它家的
`AndroidWorldController`）都跑起来才走得到 setup。我们已经有自己的执行层，
没必要为了装几个 APK 把整套东西架起来。

这里直接：**下载 APK -> adb install -> 收工**。快照那部分先不做
（见文件末尾的说明）。

## 用法

    python scripts/setup_androidworld_env.py            # 下载并安装
    python scripts/setup_androidworld_env.py --list     # 只列出要装什么
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guicascade.envs.android import find_adb  # noqa: E402

BASE_URL = "https://storage.googleapis.com/gresearch/android_world/"
CACHE = Path("D:/tools/aw-apks")

APKS = [
    # 任务真正会用到的（其余是 information_retrieval / miniwob 用的，先不装）
    "net.gsantner.markor_146.apk",                        # markor，17 个任务
    "com.flauschcode.broccoli_1020600.apk",               # broccoli，13 个
    "com.arduia.expense_11.apk",                          # pro expense，9 个
    "com.simplemobiletools.calendar.pro_238.apk",         # simple calendar，8 个
    "com.simplemobiletools.smsmessenger_85.apk",          # sms，7 个
    "com.simplemobiletools.gallery.pro_396.apk",          # gallery，4 个
    "code.name.monkey.retromusic_10603.apk",              # retro music，4 个
    "org.videolan.vlc_13050408.apk",                      # vlc，3 个
    "com.simplemobiletools.draw.pro_79.apk",              # draw pro，1 个
    "org.tasks_130605.apk",                               # tasks
    "clipper.apk",                                        # clipper，3 个
    "net.cozic.joplin_2097740.apk",                       # joplin
    "com.dimowner.audiorecorder_926.apk",                 # audio recorder，2 个
    "de.dennisguse.opentracks_5705.apk",                  # opentracks
    "androidworld.apk",                                   # 它家的辅助 app
]

# ⚠️ 没装的两个，以及原因：
#   org.videolan.vlc_13050407.apk  —— 和 13050408 是同一 app 的两个架构版本，
#                                     装一个就够（x86_64 用后者）
#   miniwobapp.apk                 —— MiniWoB 网页 benchmark 用的，和安卓任务无关


def download(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / name
    if dst.exists() and dst.stat().st_size > 100_000:
        print(f"  已缓存 {name}（{dst.stat().st_size//1024//1024} MB）")
        return dst

    url = BASE_URL + name
    print(f"  下载 {name} …", end="", flush=True)
    t0 = time.time()
    # 直连能通（实测 HTTP 200），不走代理更快
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=180) as r, dst.open("wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:  # noqa: BLE001
        print(f" 失败：{type(e).__name__}: {e}")
        dst.unlink(missing_ok=True)
        raise
    print(f" 完成（{dst.stat().st_size//1024//1024} MB，{time.time()-t0:.0f}s）")
    return dst


def install(adb: str, serial: str, apk: Path) -> bool:
    print(f"  安装 {apk.name} …", end="", flush=True)
    p = subprocess.run(
        [adb, "-s", serial, "install", "-r", "-g", str(apk)],
        capture_output=True, timeout=300,
    )
    out = (p.stdout + p.stderr).decode("utf-8", "replace")
    if p.returncode == 0 and "Success" in out:
        print(" 成功")
        return True
    # 已经装过 / 版本冲突之类的，不算致命
    print(f" 失败：{out.strip()[:120]}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="emulator-5554")
    ap.add_argument("--list", action="store_true", help="只列出要装什么")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"要安装 {len(APKS)} 个 app：")
        for a in APKS:
            print("   ", a)
        return 0

    adb = find_adb()
    print(f"\n{'='*70}")
    print(f"  给 {args.serial} 安装 AndroidWorld 依赖（{len(APKS)} 个 app）")
    print(f"  APK 缓存目录：{CACHE}")
    print(f"{'='*70}\n")

    ok = fail = 0
    for name in APKS:
        print(f"[{ok+fail+1}/{len(APKS)}] {name}")
        try:
            apk = CACHE / name if args.skip_download else download(name)
            if not apk.exists():
                print("  跳过：文件不在缓存里")
                fail += 1
                continue
            if install(adb, args.serial, apk):
                ok += 1
            else:
                fail += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ {type(e).__name__}: {str(e)[:100]}")
            fail += 1

    print(f"\n{'='*70}")
    print(f"  成功 {ok} / 失败 {fail}")
    print(f"{'='*70}")
    print("""
  接下来要做的事（**这一步没做，任务仍然会失败**）：

  AndroidWorld 的前置状态不只是"装了这个 app"，还包括 app 内部的数据
  （预置的联系人、下载好的文件、造好的笔记……）。那些存在**快照**里：

      /data/data/android_world/snapshots/<app>/

  它自己的 setup 脚本在装完 app 之后会跑一遍 app、把状态存成快照。
  那一步要 root（读 /data/data），而且要用它家的 controller，
  我们还没做——**所以现在任务会用"装好的空 app"去跑，仍然可能缺数据**。

  先往下跑一轮看看哪些任务能过；把过不了的和"缺数据"对应起来，
  再决定要不要补快照。
""")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
