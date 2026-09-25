#!/usr/bin/env bash
# 三形态全量跑：小(2B) / 大(8B) / 级联，各跑一遍全部任务。
#
# ## 为什么要单独一个脚本，而不是手敲三遍 run_suite.sh
#
# 因为**幽灵进程**。这个项目里踩过一次：终端里 Ctrl-C 掉的那个批次，
# 其实子进程还活着，在后台疯狂重试打服务器。结果是新起的批次一直
# 拿到"服务忙"，一夜跑下来结果文件里全是 `start failed`——
# **看起来像是服务器坏了，实际上是自己人打自己人。**
#
# 所以这里开跑前先确认没有别的 run_suite 在跑，有就先清掉。
#
# ## 顺序是有讲究的
#
# 先小模型，再大模型，最后级联。级联要和小模型逐条比，
# 小模型的轨迹还热乎着，出问题当场就能对上；放最后比就凉了。
#
# ## 用法
#
#     bash scripts/run_three_arms.sh              # 两组 × 三配置，全跑
#     bash scripts/run_three_arms.sh androidworld # 只跑 AndroidWorld 那组
#
# 结果实时落盘：
#     results/suite_<组>_<配置>.jsonl   一行一个任务
#     results/web_records.jsonl         后端记录（前端读这个）

set -u
cd "$(dirname "$0")/.." || exit 1

GROUP="${1:-both}"
MAXSTEPS="${2:-12}"
STAMP=$(date '+%Y-%m-%d %H:%M:%S')

# 三份配置，顺序见上面说明
CONFIGS=(
  configs/android_small.yaml
  configs/android_large.yaml
  configs/android_cascade_repeat.yaml
)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

echo "================================================================"
echo "  三形态全量测试"
echo "  开始 $STAMP   步数上限 $MAXSTEPS"
echo "================================================================"

# ---- 0. 清幽灵 ----
# 「上一个批次没死干净」是这类长跑最常见的翻车方式，开跑前先确认。
#
# ⚠️ 匹配串要够具体：第一版写的是 `-match 'run_suite'`，结果**连自己都匹配上了**
# ——本脚本的命令行里就含 `run_suite.sh` 这几个字，一跑就会把自己杀掉。
# 改成匹配 `run_suite.sh <组名>`，只有真正在跑的批次才会中。
GHOSTS=$(powershell.exe -NoProfile -Command \
  "(Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -match 'run_suite\.sh (androidworld|ours)' }).ProcessId" \
  2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$' || true)
if [ -n "$GHOSTS" ]; then
  log "⚠️ 发现还在跑的 run_suite（PID: $(echo $GHOSTS | tr '\n' ' ')），清掉"
  for p in $GHOSTS; do powershell.exe -NoProfile -Command "Stop-Process -Id $p -Force" 2>/dev/null; done
  sleep 3
fi

# ---- 1. 服务器活着吗 ----
if ! curl -s --noproxy '*' --max-time 20 "http://127.0.0.1:8765/api/tasks" >/dev/null 2>&1; then
  echo "❌ 服务器没起来。先在另一个终端跑："
  echo "     python scripts/serve_web.py"
  exit 2
fi
log "服务器正常"

# ---- 2. 开跑 ----
for G in $([ "$GROUP" = "both" ] && echo "androidworld ours" || echo "$GROUP"); do
  for CFG in "${CONFIGS[@]}"; do
    echo ""
    echo "────────────────────────────────────────────────────────────"
    log "▶ $G  ×  $(basename "$CFG")"
    echo "────────────────────────────────────────────────────────────"
    # ⚠️ **不要再套 `| tail -40`。** 早先这么写过，本意是"只留结尾的汇总"，
    # 但 `tail` 要等输入结束才输出——一整轮（两小时）的进度全被缓冲住了，
    # 中途看日志是空的，会误以为卡死。实时进度比"日志干净"重要得多。
    bash scripts/run_suite.sh "$G" "$CFG" "$MAXSTEPS" 2>&1
    log "✔ 完成 $G × $(basename "$CFG")"
  done
done

echo ""
echo "================================================================"
echo "  全部跑完 $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
for f in results/suite_*.jsonl; do
  [ -f "$f" ] || continue
  /d/Anaconda/python.exe - "$f" <<'PY'
import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
done = [r for r in rows if r.get('ok') is not None]
ok = [r for r in done if r['ok']]
print(f"  {p.split('/')[-1]:<48} {len(ok):>3}/{len(done):<3} 成功"
      f"  ({len(rows)-len(done)} 个没跑出结果)")
PY
done
