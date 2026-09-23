"""安卓任务定义：一句指令 + 一段程序化判分。

判分刻意不放进 `Environment`——**判分属于「任务」，不属于「环境」**。
一个安卓环境不该知道"设置页打开算不算成功"。这样同一个环境能服务任意任务集。

判分读的是设备的真实状态（`dumpsys`），不是让模型自己说"我完成了"。
**程序化判分不会被裁判模型的随机性污染**，这是选安卓环境的理由之一。

选任务的原则：**优先纯文本能完成的**（打开 app、系统设置这类）。
需要看图的（相册里挑一张特定照片）先不碰——那类任务对纯文本 agent
本来就不公平，会污染对级联机制的评估。
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Callable


@dataclass
class Task:
    name: str
    instruction: str
    check: Callable[[], bool]
    package: str = ""
    difficulty: str = "easy"


def _shell(adb: str, serial: str, *args: str) -> str:
    return subprocess.run(
        [adb, "-s", serial, "shell", *args], capture_output=True, timeout=30
    ).stdout.decode("utf-8", "replace")


def foreground_package(adb: str, serial: str) -> str:
    """当前前台的包名。用 dumpsys 读设备真实状态。"""
    text = _shell(adb, serial, "dumpsys", "window", "displays")
    if "mCurrentFocus" not in text:
        text += _shell(adb, serial, "dumpsys", "activity", "activities")
    m = re.search(r"mCurrentFocus=Window\{[^}]*?\s([\w\.]+)/", text)
    if not m:
        m = re.search(r"mFocusedApp.*?\s([\w\.]+)/", text)
    return m.group(1) if m else ""


def _opener(adb: str, serial: str, package: str) -> Callable[[], bool]:
    return lambda: foreground_package(adb, serial) == package


def _contains(adb: str, serial: str, needle: str) -> Callable[[], bool]:
    return lambda: needle in foreground_package(adb, serial)


def build_tasks(adb: str, serial: str) -> dict[str, Task]:
    """返回所有可用任务。每个任务独立判分，互不影响。"""
    # ⚠️ 包名是在 Pixel 6 / API 33 这个镜像上用 pm list packages 查出来的。
    # 换镜像必须重查 —— 第一版凭常识写的包名有两个在这台机器上不存在。
    tasks = [
        Task("open_settings", "打开系统设置应用（Settings）。",
             _opener(adb, serial, "com.android.settings"), "com.android.settings"),
        Task("open_contacts", "打开联系人应用（Contacts）。",
             _opener(adb, serial, "com.google.android.contacts"), "com.google.android.contacts"),
        Task("open_clock", "打开时钟应用（Clock）。",
             _opener(adb, serial, "com.google.android.deskclock"), "com.google.android.deskclock"),
        Task("open_camera", "打开相机应用（Camera）。",
             _contains(adb, serial, "camera"), "com.android.camera2"),
        Task("open_files", "打开文件管理应用（Files）。",
             _opener(adb, serial, "com.google.android.documentsui"),
             "com.google.android.documentsui"),
        Task("open_calendar", "打开日历应用（Calendar）。",
             _opener(adb, serial, "com.google.android.calendar"), "com.google.android.calendar"),
        Task("open_maps", "打开地图应用（Maps）。",
             _contains(adb, serial, "maps"), "com.google.android.apps.maps"),
        Task("open_messaging", "打开信息应用（Messages）。",
             _contains(adb, serial, "messaging"), "com.google.android.apps.messaging"),
    ]
    return {t.name: t for t in tasks}
