#!/usr/bin/env bash
# 批量跑一组任务，用 curl 驱动前端同一套 HTTP 接口。
#
# ## 为什么是 bash + curl，而不是 Python 的 run_suite.py
#
# `run_suite.py` 用 urllib 读 SSE 流时卡死了（服务端能接受新任务，说明请求
# 根本没发出去）。curl 的 `-N`（禁用缓冲）是专门为流式响应设计的，
# 而且这套驱动方式在调前端时已经验证过很多次。
#
# **换个工具比调试 urllib 的流式行为快得多**——这类"工具本身不配合"的坑，
# 通常不值得深挖。run_suite.py 留着做接口示例，批量跑用这个。
#
# ## 用法
#     bash scripts/run_suite.sh androidworld
#     bash scripts/run_suite.sh ours
#
# 结果实时追加到 results/suite_<group>.jsonl，**一行一个任务，跑完即落盘**——
# 中途挂了也不会丢已完成的。

set -u
GROUP="${1:-androidworld}"
BASE="http://127.0.0.1:8765"
CONFIG="${2:-configs/android_cascade_repeat.yaml}"
MAXSTEPS="${3:-12}"
CFGKEY=$(basename "$CONFIG" .yaml | sed "s/android_//")
OUT="results/suite_${GROUP}_${CFGKEY}.jsonl"

mkdir -p results
: > "$OUT"

# 取任务列表，筛出这一组
TASKS=$(curl -s --noproxy '*' --max-time 60 "$BASE/api/tasks" | \
  /d/Anaconda/python.exe -c "
import sys, json
d = json.load(sys.stdin)
for t in d.get('tasks', []):
    if t.get('source') == '$GROUP':
        print(t['name'])
")
N=$(echo "$TASKS" | grep -c . || true)
echo "================================================================"
echo "  $GROUP · $N 个任务 · 配置 $CONFIG"
echo "  结果写入 $OUT（一行一个，跑完即落盘）"
echo "================================================================"

i=0
for NAME in $TASKS; do
  i=$((i+1))
  echo ""
  echo "[$i/$N] $NAME"
  T0=$(date +%s)

  # ⚠️ 必须重试。服务端一次只接一个任务，上一个还没收尾时会返回 409。
  # 60 轮里只要有**一轮**比别人慢，后面不重试就全部"起不来"——
  # 实测踩过：20 个任务里 19 个空跑，结果文件里全是 null。
  RID=""
  for try in 1 2 3 4 5 6 7 8 9 10; do
    RESP=$(curl -s --noproxy '*' --max-time 60 -X POST "$BASE/api/run" \
      -H "Content-Type: application/json" \
      -d "{\"task\":\"$NAME\",\"config\":\"$CONFIG\",\"max_steps\":$MAXSTEPS,\"capture_image\":false}")
    RID=$(printf '%s' "$RESP" | /d/Anaconda/python.exe -c "
import sys, json
try:
    print(json.load(sys.stdin).get('run_id') or '')
except Exception:
    print('')" 2>/dev/null)
    [ -n "$RID" ] && break
    echo "      …服务忙，等 20s 重试（第 $try 次）"
    sleep 20
  done

  if [ -z "$RID" ]; then
    echo "      ⚠️ 重试 10 次仍起不来"
    echo "{\"name\":\"$NAME\",\"ok\":null,\"note\":\"start failed\"}" >> "$OUT"
    continue
  fi

  # 接 SSE 直到服务器关闭连接。--max-time 兜底，防止某个任务真的挂死
  curl -s --noproxy '*' -N --max-time 900 "$BASE/api/stream/$RID" > ".sse_$$.txt" 2>&1

  DT=$(( $(date +%s) - T0 ))
  /d/Anaconda/python.exe -c "
import json, sys
steps = esc = 0
ok = None; errs = []
for line in open('.sse_$$.txt', encoding='utf-8'):
    line = line.strip()
    if not line.startswith('data: '): continue
    try: d = json.loads(line[6:])
    except Exception: continue
    t = d.get('type')
    if t == 'step':
        steps += 1
        if d.get('escalated'): esc += 1
    elif t == 'verdict': ok = d.get('success')
    elif t == 'error': errs.append(d.get('message','')[:200])
mark = '✅' if ok else ('❌' if ok is False else '⚠️')
print('      %s %d 步  强模型 %d/%d  %ds' % (mark, steps, esc, steps, $DT))
for e in errs[:2]: print('      · ' + e)
rec = {'name': '$NAME', 'ok': ok, 'steps': steps, 'escalated': esc,
       'seconds': $DT, 'errors': errs[:3]}
with open('$OUT', 'a', encoding='utf-8') as f:
    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
"
  rm -f ".sse_$$.txt"
done

echo ""
echo "================================================================"
echo "  $GROUP 汇总"
echo "================================================================"
/d/Anaconda/python.exe -c "
import json
rows = [json.loads(l) for l in open('$OUT', encoding='utf-8') if l.strip()]
done = [r for r in rows if r.get('ok') is not None]
ok = [r for r in done if r['ok']]
steps = sum(r.get('steps',0) for r in rows)
esc = sum(r.get('escalated',0) for r in rows)
print('  完成 %d/%d   成功 %d/%d   成功率 %s' % (
    len(done), len(rows), len(ok), len(done),
    ('%d%%' % round(len(ok)/len(done)*100)) if done else '—'))
if steps:
    print('  总步数 %d   平均 %.1f 步/任务   强模型占比 %d%%' % (
        steps, steps/len(rows), round(esc/steps*100)))
miss = [r['name'] for r in rows if r.get('ok') is None]
if miss:
    print('  ⚠️ %d 个没跑出结果（环境问题，不算模型失败）：' % len(miss))
    for m in miss: print('       ', m)
bad = [r['name'] for r in done if not r['ok']]
if bad:
    print('  失败：')
    for b in bad: print('       ', b)
"
