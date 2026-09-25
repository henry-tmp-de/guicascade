#!/usr/bin/env bash
# 下载带 FTS3/FTS4 的官方 sqlite3.dll，并验证 Python 侧确实用上了。
#
# ## 这个脚本**不**替换 Anaconda 的 DLL —— 有意为之
#
# 最早的版本是 `cp` 到 `D:\Anaconda\Library\bin\` 覆盖原 DLL。在这台机器上
# **走不通**：
#
#     D:\Anaconda\Library\bin   只给 BUILTIN\Users  ReadAndExecute
#     D:\Anaconda\DLLs          同样不可写
#
# 两处都要管理员权限。脚本会报 `Permission denied`，而且**验证那一步还会
# 打印"❌ 仍有不可用的"** —— 看上去像脚本坏了，其实是权限。
#
# 现在改用**进程内预加载**，不需要任何管理员权限，也不改系统里任何一个文件。
# 机制和理由见 `scripts/_sqlite_fts.py` 的 docstring。
#
# ## 所以这个脚本现在只干两件事
#
#   1. 把官方 DLL 下到 D:\tools\sqlite-fts\<版本>\ （仓库外，不入库）
#   2. 跑自检，确认 fts3/fts4/fts5 都可用
#
# ## 用法
#
#     bash scripts/fix_sqlite_fts.sh          # 已有 DLL 就跳过下载，直接自检
#
# 自检不过时先看它打印的 sqlite 版本号：如果还是 3.51.0，说明预加载没生效
# （通常是 ensure_fts() 被放在了 import sqlite3 之后）。

set -u

VER="3530400"                       # sqlite 3.53.4
BASE="/d/tools/sqlite-fts"
DIR="$BASE/$VER"
ZIP="$BASE/sqlite-dll-win-x64-$VER.zip"
URL="https://www.sqlite.org/2026/sqlite-dll-win-x64-$VER.zip"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "→ 官方 DLL：$DIR/sqlite3.dll"
if [ -f "$DIR/sqlite3.dll" ]; then
  echo "  已存在，跳过下载"
else
  mkdir -p "$DIR"
  echo "  下载 $URL"
  # 走本机 Clash 代理；直连 sqlite.org 在这台机器上会超时
  https_proxy="${https_proxy:-http://127.0.0.1:7897}" \
  http_proxy="${http_proxy:-http://127.0.0.1:7897}" \
    curl -sL --max-time 120 -o "$ZIP" "$URL" || { echo "  ❌ 下载失败"; exit 1; }
  ( cd "$DIR" && unzip -oq "$ZIP" ) || { echo "  ❌ 解压失败"; exit 1; }
  echo "  已解压到 $DIR"
fi

echo
echo "→ 自检（走 scripts/_sqlite_fts.py，和线上同一条路径）"
cd "$ROOT/scripts" || exit 1
/d/Anaconda/python.exe _sqlite_fts.py
rc=$?

echo
if [ $rc -eq 0 ]; then
  echo "✅ FTS3/FTS4 可用，那 15 个任务（13 Recipe + 2 Vlc）不再被误记为模型失败"
else
  echo "❌ 自检未通过 —— 看上面的 sqlite 版本号定位是「没下载到」还是「预加载没生效」"
fi
exit $rc
