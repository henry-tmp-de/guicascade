#!/usr/bin/env bash
# 逐个 app 建前置状态，每个带超时；某个挂死只损失它自己。
#
# ## 为什么不是一条命令跑完
#
# 官方 `setup_apps()` 是一条直线：24 个 app 顺序跑，哪个卡住整条停摆。
# 实测连续两轮都栽在这上面——第二轮跑到 OsmAnd 挂死在系统权限弹窗上，
# 二十多分钟没动静，而它前面建好的快照就那么等着，后面 9 个一个没动。
#
# **挂死的代价不是"少做一个 app"，是"整批停住"。** 无人值守的跑法里
# 这个代价不能接受。所以这里把粒度降到单个 app，外面套 `timeout`：
# 挂死就砍掉，下一个照跑。
#
# 每个 app 最多 HARD_TIMEOUT 秒。正常一个 app 约 30~60 秒，
# 给到 240 秒是很宽裕的——**超过这个数基本就是真挂了，不是在慢慢做**。
#
# ## 用法
#
#     bash scripts/aw_setup_loop.sh
#
# 可反复跑：已经有快照的 app 会自动跳过（脚本自己查设备）。

set -u
cd "$(dirname "$0")/.." || exit 1

HARD_TIMEOUT="${1:-240}"
LOG="results/aw_setup_loop.log"

# 全部 app 类名（官方 _APPS 的顺序），逐个单独跑
APPS=(
  AndroidWorldApp AudioRecorder CameraApp ChromeApp ClipperApp ClockApp
  ContactsApp DialerApp ExpenseApp FilesApp JoplinApp MarkorApp MiniWobApp
  OpenTracksApp OsmAndApp RecipeApp RetroMusicApp SettingsApp
  SimpleCalendarProApp SimpleDrawProApp SimpleGalleryProApp
  SimpleSMSMessengerApp TasksApp VlcApp
)

# 代理必须清掉：官方 setup 用 requests 下载 Google 存储，
# 而 requests 在 Windows 上除了读环境变量，还会**回落到注册表里的系统代理**
# （Clash Verge 会写那里）。所以光 unset 环境变量不够，要显式设 NO_PROXY。
export HTTP_PROXY= HTTPS_PROXY= ALL_PROXY=
export http_proxy= https_proxy= all_proxy=
export NO_PROXY="storage.googleapis.com,localhost,127.0.0.1"
export no_proxy="storage.googleapis.com,localhost,127.0.0.1"

echo "================================================================"
echo "  逐个建前置状态   单个超时 ${HARD_TIMEOUT}s"
echo "  日志 $LOG"
echo "================================================================"

ok=0; skip=0; timeout_n=0; fail=0
for A in "${APPS[@]}"; do
  {
    echo ""
    echo "[$(date '+%H:%M:%S')] ▶ $A"
  } | tee -a "$LOG"

  timeout -k 15 "$HARD_TIMEOUT" /d/Anaconda/python.exe scripts/aw_official.py \
      --apps "$A" >> "$LOG" 2>&1
  rc=$?

  case $rc in
    0)   verdict="✅ 完成";  ok=$((ok+1)) ;;
    124) verdict="⏱  超时被砍（跳过，继续下一个）"; timeout_n=$((timeout_n+1)) ;;
    *)   verdict="⚠️  退出码 $rc"; fail=$((fail+1)) ;;
  esac
  echo "[$(date '+%H:%M:%S')] $verdict  $A" | tee -a "$LOG"

  # 每轮之间让模拟器喘口气，也把残留的弹窗清掉
  ADB="D:/tools/android-sdk/platform-tools/adb.exe"
  "$ADB" -s emulator-5554 shell input keyevent KEYCODE_HOME >/dev/null 2>&1
  sleep 3
done

echo ""
echo "================================================================"
echo "  完成 $ok   超时 $timeout_n   失败 $fail"
ADB="D:/tools/android-sdk/platform-tools/adb.exe"
echo "  设备上快照数：$("$ADB" -s emulator-5554 shell 'ls -1 /data/data/android_world/snapshots/ | wc -l' 2>/dev/null | tr -d '\r')"
echo "================================================================"
