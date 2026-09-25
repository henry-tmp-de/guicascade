"""在**不碰系统目录**的前提下，给 Python 补上 SQLite 的 FTS3/FTS4。

## 为什么需要这个东西

AndroidWorld 有一批任务要读带 FTS 索引的 sqlite 库（broccoli 的菜谱库、
VLC 的播放列表库）。本机 Anaconda 自带的 `sqlite3.dll` 编译时**只开了 FTS5**，
一读就报：

    OperationalError: no such module: FTS4

这个错的隐蔽之处在于它发生在 `initialize_task` 里 —— 任务**在起点就废了**，
然后被记成"模型失败"。实测 59 个 AndroidWorld 任务里有 **15 个**
（13 个 Recipe + 2 个 Vlc）栽在这上面，分数上完全看不出来是谁的锅。

## 为什么不是"把 Anaconda 的 DLL 换掉"

试过，**在这台机器上走不通**：

    D:\\Anaconda\\Library\\bin   只给 BUILTIN\\Users  ReadAndExecute
    D:\\Anaconda\\DLLs          同样不可写

两处都要管理员权限。而且 sqlite.org 上能下到的官方 DLL 版本号还不一定
比 Anaconda 自带的新，换过去有降级风险。

## 实际用的办法：进程内预加载

Windows 加载器解析依赖时，**先按基名匹配已经加载进本进程的模块**。
所以只要在 `import sqlite3` **之前**先用 ctypes 把官方 DLL 拉进进程，
之后 `_sqlite3.pyd` 找 `sqlite3.dll` 时就会命中我们这份，而不是 Anaconda 那份。

    ctypes.WinDLL(r"...\\3530400\\sqlite3.dll")   # 先
    import sqlite3                                 # 后 —— 顺序不能反

**顺序反了就完全无效**：`_sqlite3.pyd` 已经把 Anaconda 的 DLL 绑死了，
再预加载也只是多加载一份，Python 用的还是旧的那份。实测确认过这一点，
所以下面才要在 `sys.modules` 里检查 `sqlite3` 有没有被抢先 import 过 ——
**宁可报"没生效"，也不要让人以为生效了。**

## 用法

    from _sqlite_fts import ensure_fts
    ensure_fts()          # 必须在 import sqlite3 / 任何碰 sqlite 的模块之前

DLL 不在就静默跳过：这台机器上放没放 DLL，不该影响其他任务的运行。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 官方 DLL 的位置。换机器时改这里，或设环境变量 GUICASCADE_SQLITE_DLL。
DEFAULT_DLL = Path("D:/tools/sqlite-fts/3530400/sqlite3.dll")

_checked = False


def ensure_fts(dll: Path | str | None = None, *, verbose: bool = True) -> bool:
    """把带 FTS3/FTS4 的 sqlite3.dll 预加载进本进程。返回是否成功。

    必须在**任何** `import sqlite3` 之前调用。已经晚了就返回 False 并出声 ——
    静默失败在这里是最坏的结果：任务照样在起点废掉，但你不会再看到那条错。
    """
    global _checked
    if _checked:                      # 同一进程里重复调用没意义，也不该重复加载
        return True
    _checked = True

    # 只有 Windows 需要这一套；别的平台直接用系统 sqlite，本来就有 FTS3/4
    if sys.platform != "win32":
        return True

    if "sqlite3" in sys.modules or "_sqlite3" in sys.modules:
        if verbose:
            print("[sqlite_fts] ⚠️ 调用太晚：sqlite3 已经被 import 过了，"
                  "预加载不会生效。ensure_fts() 必须放在最前面。", flush=True)
        return False

    p = Path(dll or __import__("os").environ.get("GUICASCADE_SQLITE_DLL") or DEFAULT_DLL)
    if not p.exists():
        if verbose:
            print(f"[sqlite_fts] 未找到 {p}，跳过（FTS 类任务可能仍在起点失败）",
                  flush=True)
        return False

    try:
        import ctypes
        ctypes.WinDLL(str(p))
    except Exception as e:  # noqa: BLE001 - 预加载失败不该让整个服务起不来
        if verbose:
            print(f"[sqlite_fts] 预加载失败：{type(e).__name__}: {e}", flush=True)
        return False

    if verbose:
        import sqlite3                # 此时才第一次 import，会命中刚加载的那份
        print(f"[sqlite_fts] 已预加载 {p.name}（sqlite {sqlite3.sqlite_version}）",
              flush=True)
    return True


def fts_report() -> dict[str, bool]:
    """FTS3/4/5 各自可用与否 —— 给 `python -m` 自检和排错用。"""
    import sqlite3
    out = {}
    for m in ("fts3", "fts4", "fts5"):
        try:
            c = sqlite3.connect(":memory:")
            c.execute(f"CREATE VIRTUAL TABLE t USING {m}(x)")
            c.execute("INSERT INTO t VALUES('hello world')")
            c.execute("SELECT * FROM t WHERE t MATCH 'hello'").fetchall()
            out[m] = True
        except Exception:  # noqa: BLE001
            out[m] = False
    return out


if __name__ == "__main__":
    ok = ensure_fts()
    rep = fts_report()
    import sqlite3
    print(f"sqlite 版本 {sqlite3.sqlite_version}")
    for k, v in rep.items():
        print(f"  {k}: {'✅' if v else '❌'}")
    raise SystemExit(0 if all(rep.values()) else 1)
