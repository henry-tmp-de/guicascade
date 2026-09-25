#!/usr/bin/env bash
# 把 AndroidWorld 的官方快照备份到 /sdcard，开机后还原。
#
# ## 为什么需要这个
#
# 这台 AVD 的**应用数据分区不持久**：模拟器一重启，`/data/data` 就回到初始状态。
# 实测（写标记 -> 优雅关机 -> 重启 -> 读）：
#
#     /sdcard/_persist_sdcard.txt         → 还在    ✅ 独立 SD 卡镜像
#     /data/data/_persist_test/marker.txt → 没了    ❌ 数据分区是临时的
#
# 改了 `disk.dataPartition.path`（`<temp>` → `userdata-qemu.img`）**也没用**，
# 所以不折腾原因了，直接绕：`/sdcard` 是持久的，那就把快照放那儿。
#
# 代价对比很悬殊：
#
#     重跑官方 setup  约 40 分钟（24 个 app 逐个清空、启动、点引导页）
#     从 /sdcard 还原  约 5 秒（8MB 的目录复制）
#
# ## 用法
#
#     bash scripts/aw_snapshot_keep.sh save      # setup 跑完后：快照 -> /sdcard
#     bash scripts/aw_snapshot_keep.sh restore   # 每次开机后：/sdcard -> 快照位置
#     bash scripts/aw_snapshot_keep.sh status    # 看看两边各有多少个

set -u
ADB="D:/tools/android-sdk/platform-tools/adb.exe"
SERIAL="emulator-5554"
LIVE="/data/data/android_world/snapshots"
KEEP="/sdcard/_aw_snapshots"

sh() { "$ADB" -s "$SERIAL" shell "$@" 2>&1 | tr -d '\r'; }

# 读 /data/data 需要 root；模拟器重开后 adbd 会退回普通权限
if [ "$(sh whoami)" != "root" ]; then
  "$ADB" -s "$SERIAL" root >/dev/null 2>&1
  sleep 4
fi

case "${1:-status}" in
  save)
    n=$(sh "ls -1 $LIVE 2>/dev/null | wc -l")
    if [ "${n:-0}" -eq 0 ]; then
      echo "❌ $LIVE 里一个快照都没有，没什么可备份的。先跑官方 setup。"
      exit 1
    fi
    sh "rm -rf $KEEP"
    sh "cp -r $LIVE $KEEP"
    m=$(sh "ls -1 $KEEP 2>/dev/null | wc -l")
    echo "✅ 已备份 $m 个快照到 $KEEP（源目录 $n 个）"
    [ "$m" = "$n" ] || echo "⚠️ 数量对不上，检查一下"
    ;;
  restore)
    n=$(sh "ls -1 $KEEP 2>/dev/null | wc -l")
    if [ "${n:-0}" -eq 0 ]; then
      echo "❌ $KEEP 里没有备份。先跑一次官方 setup 再 save。"
      exit 1
    fi
    sh "mkdir -p /data/data/android_world"
    sh "rm -rf $LIVE"
    sh "cp -r $KEEP $LIVE"
    # 复制过来的文件属主/安全上下文会变，不修的话 app 读不到
    sh "restorecon -RD /data/data/android_world"
    sh "chmod 777 -R /data/data/android_world"
    m=$(sh "ls -1 $LIVE 2>/dev/null | wc -l")
    echo "✅ 已从 $KEEP 还原 $m 个快照"
    ;;
  *)
    echo "  实时 ($LIVE): $(sh "ls -1 $LIVE 2>/dev/null | wc -l") 个"
    echo "  备份 ($KEEP): $(sh "ls -1 $KEEP 2>/dev/null | wc -l") 个"
    ;;
esac
