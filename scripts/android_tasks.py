"""安卓任务定义：一句指令 + 一段程序化判分。

## 判分为什么不放在 Environment 里

`env.success()` 恒返回 None。**判分属于「任务」，不属于「环境」**——
一个安卓环境不该知道"设置页打开算不算成功"。这样同一个环境能服务任意
任务集，而不需要为每个 benchmark 改环境代码。

判分读的是设备的**真实状态**（`settings get` / `content query` / `dumpsys` /
无障碍树文本），不是让模型自己说"我完成了"。程序化判分不会被裁判模型的
随机性污染，这是选安卓环境的理由之一。

## ⚠️ 任务必须"长"且"跨应用"——这是踩过的坑

第一版任务是 8 个「打开某个 app」。看起来没问题，实际是**一步就能做完**：

    模型：任务说"时钟" -> 环境/提示词里查到包名 -> open_app -> finish
    两步收尾，全程没看过屏幕一眼。

后果是整轮三臂对照的数据全部作废——三条臂测的都是"谁背表背得准"。
所以这里定两条硬规矩：

1. **每个任务至少跨 3 个界面**，元素序号每步都在变。不读屏就一步走不动。
2. **必须跨应用**——只在「设置」里打转的话，测出来的还是"会不会用设置"。
   现在覆盖：设置 / 浏览器 / 联系人 / 时钟 / 文件管理。

## 为什么判分只用 ASCII 输入

`adb shell input text` **不支持非 ASCII**（实测中文会让 input 命令直接抛
NullPointerException）。所以凡是要模型输入的字段一律用英文/数字，
否则任务在环境层就废了，测出来的不是模型能力。
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable

__all__ = ["Task", "build_tasks", "check_network", "cleanup_device",
           "foreground_package", "ui_text"]


@dataclass
class Task:
    name: str
    instruction: str
    check: Callable[[], bool]
    package: str = ""
    """任务起点的 app。`reset` 时会被 force-stop，保证起点干净。"""
    steps_hint: tuple[int, int] = (3, 8)
    """预期步数区间。用来事后判断任务是不是太短（一步做完 = 没有测量价值）。"""


def _shell(adb: str, serial: str, *args: str, timeout: float = 30) -> str:
    return subprocess.run(
        [adb, "-s", serial, "shell", *args], capture_output=True, timeout=timeout
    ).stdout.decode("utf-8", "replace")


def foreground_package(adb: str, serial: str) -> str:
    """当前前台的包名。读设备的真实状态。"""
    text = _shell(adb, serial, "dumpsys", "window", "displays")
    if "mCurrentFocus" not in text:
        text += _shell(adb, serial, "dumpsys", "activity", "activities")
    m = re.search(r"mCurrentFocus=Window\{[^}]*?\s([\w\.]+)/", text)
    if not m:
        m = re.search(r"mFocusedApp.*?\s([\w\.]+)/", text)
    return m.group(1) if m else ""


def ui_text(adb: str, serial: str) -> str:
    """当前屏幕上所有可见文字，拼成一段。

    用无障碍树而不是截图——判分不该依赖 OCR，也不该被渲染差异影响。
    """
    xml_path = "/sdcard/_guicascade_check.xml"
    _shell(adb, serial, "uiautomator", "dump", xml_path)
    raw = _shell(adb, serial, "cat", xml_path)
    return " ".join(re.findall(r'(?:text|content-desc)="([^"]+)"', raw))


def _setting(adb: str, serial: str, ns: str, key: str) -> str:
    return _shell(adb, serial, "settings", "get", ns, key).strip()


def cleanup_device(adb: str, serial: str, *, verbose: bool = True) -> None:
    """把一个任务留在设备上的痕迹擦掉，好让下一个任务从干净状态开始。

    ## 为什么非要这么做

    实测踩过：`settings_airplane_on` 排在第 1 个跑，把飞行模式打开了；
    它排第 3 个的 `chrome_search_and_open` 需要联网——**网是断的，必然失败**。
    那条失败当时被算在模型头上，其实是环境被前一个任务毁了。

    任务之间**不能有任何副作用关联**。做不到"这个任务本身无副作用"时，
    就必须在它跑完之后**把它改过的东西全部还原**。

    ## 清什么

    - 系统设置：飞行模式、亮度（这两个会实打实影响别的任务）
    - 测试建出来的联系人、闹钟（不清的话，下一个任务一开始就判成功）

    ## 为什么不用 `AndroidEnv.reset()`

    `reset()` 只做两件事：force-stop 目标任务 + 回桌面。它管的是**界面起点**，
    不管**设备状态**。这两件事得分开——界面回到桌面了，飞行模式该开还是开着。
    """
    log = (lambda m: print("     " + m)) if verbose else (lambda m: None)

    # 系统设置还原
    if _setting(adb, serial, "global", "airplane_mode_on") == "1":
        _shell(adb, serial, "settings", "put", "global", "airplane_mode_on", "0")
        _shell(adb, serial, "am", "broadcast", "-a",
               "android.intent.action.AIRPLANE_MODE", "--ez", "state", "false")
        _shell(adb, serial, "svc", "wifi", "enable")
        log("还原：飞行模式已关闭")

    # 测试数据清理：联系人和闹钟都是"建了就一直在"的东西，
    # 不清掉的话下一个任务在起点就会判成功——**白送分**。
    #
    # ⚠️ 清的是 **provider**（`com.android.providers.contacts`），不是联系人
    # 那个界面 app（`com.google.android.contacts`）。数据存在 provider 里，
    # 清 app 只是清了界面，数据一条不少。这个区别实测踩过：
    # 重启模拟器后 `contacts_create` 依然在起点返回 True，
    # 因为上一轮建的联系人还在库里躺着——**重启模拟器并不会清数据库**。
    #
    # 另一条路 `content delete --where "display_name='Zhangsan'"` 走不通：
    # 引号在 adb shell 里会被剥掉，SQL 变成 `WHERE display_name=Zhangsan`，
    # provider 报 "no such column: Zhangsan"。要转义好几层，不值得。
    _shell(adb, serial, "pm", "clear", "com.android.providers.contacts")
    _shell(adb, serial, "pm", "clear", "com.google.android.deskclock")

    # Chrome 也清：让它每次都回到"首次启动"状态。
    #
    # 为什么这么做：Chrome 首次启动会弹两层欢迎页。如果只清一次，第一条臂
    # 遇到引导页、后两条臂遇不到——**三条臂的环境就不一样了，没法比**。
    # 每次都清，保证三条臂面对的是同一个（更难的）起点。
    #
    # 而且这本来就是该测的能力：**看到和预期不一样的界面时，能不能自己
    # 判断"这是欢迎页，先处理掉再继续"，而不是以为动作没生效、原地重试。**
    # 早先我图省事手工把引导页点掉了，那等于替模型做了一部分工作——
    # 现在改成让模型自己处理，提示词里也加了对应的说明。
    _shell(adb, serial, "pm", "clear", "com.android.chrome")
    log("清理：联系人库/闹钟/Chrome 数据已重置")

    # 回到确定的起点
    _shell(adb, serial, "input", "keyevent", "KEYCODE_HOME")


def check_network(adb: str, serial: str) -> tuple[bool, str]:
    """跑测试**之前**确认模拟器网络是通的。返回 `(是否可用, 说明)`。

    ## 为什么必须有这一道

    实测踩过，代价是一整轮三臂对照的数据作废：

        模拟器 DNS 坏了 -> 浏览器任务加载不出结果页
        -> 模型在地址栏里反复重试 -> 任务失败
        -> **判分记成"模型不会用浏览器"**

    这条失败链上**没有任何一步报错**。没有自检的话，你只会看到一份
    "模型不行"的数据，而真实原因是环境坏了。

    ## 检查什么

    只检查**任务真正依赖的东西**，而且**只认真实行为**：

    - 飞行模式是不是开着（这个会实打实地断网）
    - 真去 TCP 连一次 `www.bing.com:80` —— 一条命令同时考了 DNS、
      路由和对外连通，而且这正是浏览器任务要做的那件事

    ⚠️ 这个环境的三个已知特点（写在这里，免得下次又当成模型问题）：
      1. **ICMP 永远不通，而且这是正常的**：模拟器走 QEMU 用户态网络
         （slirp），slirp 不转发 ICMP。**所以判据里绝不能用 ping。**
      2. **HTTPS 走不通**（TLS 握手被掐，ERR_CONNECTION_CLOSED），HTTP 正常
      3. **Google 被 DNS 污染**（解析到 185.45.5.35），bing / baidu 正常
    """
    problems: list[str] = []

    if _setting(adb, serial, "global", "airplane_mode_on") == "1":
        problems.append("飞行模式开着")

    # 判据是"**真的连一次 TCP**"，不是 ping。
    #
    # ⚠️ 这个环境里 **ICMP 永远不通，而且这是正常的**：模拟器走的是 QEMU 的
    # 用户态网络（slirp），slirp 不转发 ICMP。实测（2026-09-25）：
    #
    #     ping 8.8.8.8        -> 100% packet loss
    #     路由表              -> 只有 10.0.0.0/8，补了 default 也一样
    #     nc -w 6 www.baidu.com 80 -> rc=0        <- TCP 好好的
    #     nc -w 6 www.bing.com  80 -> rc=0
    #     nc -w 5 <不存在的域名> 80 -> rc=1       <- 这个测法有判别力
    #
    # 之前这里用 ping 当判据，后果是**每一条记录都带一句"网络不可用"**，
    # 而网络其实是好的。**一个永远触发的检查比没有检查更糟**：它会训练人
    # 忽略错误，真正出问题那天也没人看。
    #
    # 顺带说明为什么不用 `net.dns1`：那是 Android 10 之前的属性，现代 Android
    # 走 PrivateDns/resolv 那套，**这个属性空着是正常的**。拿"看起来该有的
    # 字段"当判据而不是拿真实行为当判据，就会在正常环境下报故障。
    net = _shell(adb, serial, "sh", "-c",
                 "command -v nc >/dev/null 2>&1 || { echo NO_NC; exit 9; }; "
                 "echo -n '' | nc -w 6 www.bing.com 80; echo rc=$?")
    if "NO_NC" not in net and "rc=0" not in net:
        # 镜像里没带 nc 时**不报故障**：那是"这一项没法测"，不是"网络坏"。
        # 拿工具缺失当故障判据，就又犯了"用间接指标代替真实行为"的老毛病。
        problems.append("连不上 www.bing.com:80（TCP 层就不通，浏览器任务必挂）")

    if problems:
        return False, "；".join(problems)
    return True, "网络正常"


def digits(s: str) -> str:
    """只留数字。

    ⚠️ 判分里**永远不要拿原始字符串比电话号码**。安卓的联系人会按本地格式
    重排号码，实测存进去是 `13800138000`，读出来是 `1 (380) 013-8000`——
    中间多了空格、括号、连字符。拿原串比会**冤枉一个明明做对了的模型**，
    而且这种假阴性极难发现：判分说失败，但设备上联系人好端端躺着。

    踩过：`contacts_create` 因此被误判为失败，而模型把姓名和电话都填对了。
    """
    return re.sub(r"\D", "", s or "")


def build_tasks(adb: str, serial: str) -> dict[str, Task]:
    """返回所有可用任务。每个任务独立判分，互不影响。

    ⚠️ 包名/界面文案都是**在这台镜像上核对过的**，不是凭常识写的。
    第一版凭常识写的 `com.android.contacts` 在这台机器上根本不存在，
    模型照抄之后一步都走不动。**换镜像必须重查。**
    """
    fg = lambda: foreground_package(adb, serial)
    text = lambda: ui_text(adb, serial)

    tasks = [
        # ---------------- 设置：跨多层菜单 ----------------
        # ⚠️ 这里原来是 `settings_airplane_on`（打开飞行模式），**已删除**。
        #
        # 它是个**有副作用**的任务：改了全局网络状态，而且跑完不还原。
        # 实测后果——它排在第 1 个跑，之后 `chrome_search_and_open` 排第 3 个，
        # 那时飞机模式还开着、网络是断的，Chrome 必然失败。
        # 那条失败被记到模型头上了，其实是环境被前一个任务毁掉了。
        #
        # 判据：**一个任务跑完后，不该在设备上留下任何影响别的任务的东西。**
        # 改状态的任务（飞行模式、亮度、静音）一律不要，除非跑完立刻还原。
        Task(
            name="settings_apps_list",
            instruction="打开系统设置，进入「Apps」（应用列表）页面。",
            package="com.android.settings",
            steps_hint=(4, 9),
            # ⚠️ 这一条**改过两次**，两次都是"凭想象写判分"而不是照着真实屏幕写。
            #
            # 第一版：要求屏幕上出现 ("Chrome","Camera","Clock","Maps") 里至少 2 个。
            #   想当然以为应用列表会列出所有应用——**它先显示的是"最近打开的应用"**，
            #   模型做对了（连点四次 Apps 也确实跳转成功了），判分却只匹配到 1 个，
            #   把一条正确的轨迹判成失败。
            #
            # 现在改用 AOSP 设置页的**结构性文案**做判据：这几个字符串只在
            # "应用列表"这一页出现，在设置主页、其他子页都不会有。
            # 比"数应用名"稳得多，也不依赖这台机器上恰好装了什么应用。
            check=lambda: "settings" in fg()
            and bool(re.search(r"(See all \d+ apps|Recently opened apps|Default apps|App info)",
                               text())),
        ),
        Task(
            name="settings_about_version",
            instruction="打开系统设置，进入「About phone」，找到这台设备的 Android 版本号。",
            package="com.android.settings",
            steps_hint=(4, 9),
            # 真值从系统属性现取，再看界面上有没有出现这个字符串。
            #
            # 上一版写得又绕又脆：用 `\b1[0-9]\b` 去正则屏幕上任意两位数，
            # 拿第一个匹配去比版本号——界面上一堆两位数（内存、年份、构建号），
            # 匹配到哪个全看运气。判分不该有这种随机性。
            #
            # ⚠️ 指令里的 "About phone" 是**英文**：`adb shell input text`
            # 不支持非 ASCII，写中文的话模型在设置里搜索时会直接卡死
            # （实测它真的试了 `input_text(text='关于手机')`，然后原地打转）。
            check=lambda: _shell(adb, serial, "getprop", "ro.build.version.release").strip()
            in text(),
        ),
        # ---------------- 浏览器：完全另一套交互 ----------------
        Task(
            name="chrome_search_and_open",
            instruction="打开 Chrome 浏览器，访问 www.bing.com 并搜索「android automation」，"
                        "让搜索结果页面显示出来。",
            package="com.android.chrome",
            steps_hint=(6, 12),
            # ⚠️ 指令里写明 bing.com 和 http，**这是环境约束不是给答案**：
            #
            #   1. 这个环境里 **HTTPS 被掐断**（TLS 握手失败，ERR_CONNECTION_CLOSED），
            #      只有 HTTP 能通。所以不能让它访问 https 站点。
            #   2. **Google 被 DNS 污染**（解析到 185.45.5.35，连不上）。
            #      搜索引擎里只有 bing / baidu 这类能通。
            #
            # 这两条是**设备/网络环境的事实**，和"任务要实现什么"无关——
            # 和"打开哪一个 app"是同一个性质。不写清楚的话，模型无论多聪明
            # 都会失败，而那失败不是它的错。
            #
            # ⚠️ 判分也重写了。**旧版是假阳性**：只要求屏幕上出现 "automation"，
            # 而**模型输入的内容本身就停在地址栏里**——页面根本没加载出来，
            # 判分照样通过。踩过：那条"✅"是白送的。
            #
            # 现在要求看到 **Bing 结果页的标题**（`<query> - Search`）。
            # 错误页（"This site can't be reached"）不会有这个标题。
            check=lambda: "chrome" in fg().lower()
            and bool(re.search(r"automation\s*[-–]\s*Search", text())),
        ),
        # ---------------- 联系人：要输入文字、要点保存 ----------------
        Task(
            name="contacts_create",
            instruction="打开联系人应用，新建一个联系人，姓名填 Zhangsan，电话填 13800138000，然后保存。",
            package="com.google.android.contacts",
            steps_hint=(6, 12),
            # ⚠️ 用 `digits()` 归一化再比，不能拿原串比——见 `digits()` 的说明。
            # 真值读的是联系人数据库，不是界面文本：界面可能只是"正在编辑"，
            # 而数据库里有才叫真的保存成功了。
            check=lambda: "13800138000" in digits(_shell(
                adb, serial, "content", "query",
                "--uri", "content://com.android.contacts/data/phones",
                "--projection", "data1",
            )),
        ),
        # ---------------- 时钟：要滚动、要点数字 ----------------
        Task(
            name="clock_add_alarm",
            instruction="打开时钟应用，新建一个早上 7:00 的闹钟。",
            package="com.google.android.deskclock",
            steps_hint=(5, 11),
            # ⚠️ 括号**必须**加。原写法是
            #     `"clock" in fg or "deskclock" in fg and "7:00" in text`
            # Python 里 `and` 优先级高于 `or`，实际等价于 `A or (B and C)`——
            # 于是**只要前台是时钟应用就判成功，闹钟设没设根本不看**。
            # 这是一条白送分的假阳性，而且看起来完全正常。
            #
            # 教训：判分表达式里混用 and/or 一律加括号，别赌优先级。
            check=lambda: ("clock" in fg().lower() or "deskclock" in fg().lower())
            and bool(re.search(r"\b0?7:00\b", text())),
        ),
        # ---------------- 文件管理：纯列表导航 ----------------
        Task(
            name="files_open_downloads",
            instruction="打开文件管理应用，进入 Downloads 文件夹。",
            package="com.google.android.documentsui",
            # ⚠️ **这一条是刻意留的简单题（1~2 步）。**
            #
            # 文件管理一打开就停在 Downloads，所以它只需要一个 open_app。
            # 本来想删掉，但任务集**需要难度梯度**：全是难题的话，小模型会
            # 全军覆没，级联除了"每步都升级"以外没有别的选择，也就测不出路由
            # 到底有没有判断力。留一条简单的，才看得出"该省的时候省没省"。
            #
            # 但**校准难度时不能把它算进去**——它和另外 5 条不在一个量级。
            steps_hint=(1, 3),
            check=lambda: "documentsui" in fg() and "download" in text().lower(),
        ),
    ]
    return {t.name: t for t in tasks}
