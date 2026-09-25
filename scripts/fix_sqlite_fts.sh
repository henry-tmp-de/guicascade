#!/usr/bin/env bash
# 给 Anaconda 的 sqlite3 补上 FTS3/FTS4 —— 换 sqlite.org 官方的 DLL。
#
# ## 为什么需要
#
# Anaconda 自带的 `sqlite3.dll` 编译时**没开 FTS3/FTS4**（只开了 FTS5）。
# 而 AndroidWorld 有一批任务要读带 FTS 索引的 sqlite 库（broccoli 的菜谱库、
# VLC 的播放列表库），本机一读就报：
#
#     OperationalError: no such module: FTS4
#
# 后果非常隐蔽：`initialize_task` 抛异常 -> 任务在起点就废 -> **被记成模型失败**。
# 实测 59 个 AndroidWorld 任务里有 **15 个**（13 个 Recipe + 2 个 Vlc）栽在这上面。
# 这不是模型的锅，是**本机 Python 的编译选项**，但分数上完全看不出来。
#
# ## 为什么要换 DLL 而不是装包
#
# `pysqlite3-binary` 没有 Windows 轮子（那是 Linux/macOS 专用）。
# sqlite.org 官方的 Windows DLL **默认就编了 FTS3/4/5**，直接换上去最省事。
# 注意：SQLite 里 **FTS4 是跟着 FTS3 一起编的**——看到 `ENABLE_FTS3`
# 就等于 FTS3 和 FTS4 都能用，不用去找单独的 `ENABLE_FTS4`。
#
# ## 用法（必须先把所有 python 进程停掉，DLL 被占用时换不了）
#
#     bash scripts/fix_sqlite_fts.sh
#
# 回滚：把 `sqlite3.dll.anaconda-backup` 改回 `sqlite3.dll` 即可。

set -u

BIN="/d/Anaconda/Library/bin"
SRC="/d/tools/sqlite-fts/sqlite3.dll"
DST="$BIN/sqlite3.dll"
BAK="$BIN/sqlite3.dll.anaconda-backup"

if [ ! -f "$SRC" ]; then
  echo "❌ 找不到官方 DLL：$SRC"
  echo "   下载：curl -sL -o /d/tools/sqlite-fts/sqlite-dll.zip \\"
  echo "         https://www.sqlite.org/2024/sqlite-dll-win-x64-3460100.zip"
  exit 1
fi

# 检查还有没有 python 占着 DLL
echo "→ 检查占用..."
if powershell.exe -NoProfile -Command \
   "if (Get-Process python -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }" 2>/dev/null; then
  echo "  python 进程已清空"
else
  echo "  ⚠️ 还有 python 在跑，DLL 会被占用导致替换失败。"
  echo "     先停掉服务器和跑测进程再来。"
  exit 2
fi

# 备份（只备份一次，反复跑不会用坏掉的版本覆盖好备份）
if [ -f "$BAK" ]; then
  echo "→ 备份已存在，保留不动：$BAK"
else
  cp -p "$DST" "$BAK" && echo "→ 已备份到 $BAK"
fi

cp "$SRC" "$DST" && echo "→ 已替换为官方 DLL"

echo "→ 验证..."
/d/Anaconda/python.exe -c "
import sqlite3
print('  sqlite 版本:', sqlite3.sqlite_version)
bad = []
for m in ('fts3', 'fts4', 'fts5'):
    try:
        c = sqlite3.connect(':memory:')
        c.execute(f'CREATE VIRTUAL TABLE t USING {m}(x)')
        c.execute(\"INSERT INTO t VALUES('hello world')\")
        r = c.execute(\"SELECT * FROM t WHERE t MATCH 'hello'\").fetchall()
        print(f'  {m}: ✅  查询返回 {r}')
    except Exception as e:
        print(f'  {m}: ❌ {e}')
        bad.append(m)
if bad:
    print()
    print('  ⚠️ 仍有不可用的：', bad)
    print('     回滚：cp $BAK $DST'.replace('\$BAK', '$BAK').replace('\$DST', '$DST'))
    raise SystemExit(1)
print()
print('  ✅ FTS3/FTS4 都可用了，可以补跑那 15 个任务')
"
