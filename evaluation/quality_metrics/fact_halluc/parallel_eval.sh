#!/usr/bin/env bash
set -euo pipefail

# ====================== 可自定义参数 ======================
NUM_JOBS=${1:-10}  # 并发度，可运行时传参：./run_eval_fact_parallel.sh 20
PY_SCRIPT="eval_fact_count_v3.py"

# 输入/输出路径（按你的单条命令默认值设置）
INPUT_JSON="/home/fmy/project/DPO-Summary/data/qwen/ruby/train/enhanced_data_full_0_1249.json"
OUTPUT_DIR="/home/fmy/project/DPO-Summary/evaluate/ProCon/fact_halluc/ruby/ds3.2_ruby_train_step5"

# 每批处理条数（对应 --limit）
# 每批处理条数（对应 --limit）
BATCH_LIMIT=50

# 可选：总条数（留空则自动探测；如有需要可手动指定以跳过探测）
TOTAL_ITEMS="1000"

# 额外透传参数（可为空，例如：EXTRA_ARGS='--device cuda:0 --foo bar'）
EXTRA_ARGS=""
# =========================================================

mkdir -p "$OUTPUT_DIR"

# ---------- 自动探测总条数（JSON 数组或 JSONL） ----------
if [[ -z "${TOTAL_ITEMS}" ]]; then
  set +e
  TOTAL_ITEMS=$(python - <<'PY'
import json, sys, os
p = os.environ.get('INPUT_JSON')
try:
    with open(p, 'rb') as f:
        first = f.read(1)
        f.seek(0)
        if first == b'[':
            data = json.load(f)
            print(len(data))
        else:
            # JSONL：按非空行计数
            n = 0
            for line in f:
                if line.strip():
                    n += 1
            print(n)
except Exception as e:
    # 探测失败时不输出任何内容，让外层处理
    pass
PY
)
  rc=$?
  set -e
  if [[ $rc -ne 0 || -z "${TOTAL_ITEMS}" ]]; then
    echo "⚠️ 自动探测总条数失败。请设置 TOTAL_ITEMS 后重试。" >&2
    exit 1
  fi
fi

# ---------- 展示参数 ----------
echo "💡 并行评测启动：总 ${TOTAL_ITEMS} 条，每批 ${BATCH_LIMIT} 条，并发 ${NUM_JOBS} 进程"
echo "Python脚本: $PY_SCRIPT"
echo "输入文件:   $INPUT_JSON"
echo "输出目录:   $OUTPUT_DIR"
[[ -n "${EXTRA_ARGS}" ]] && echo "透传参数:   ${EXTRA_ARGS}"
echo "--------------------------------------------------------"

export PY_SCRIPT INPUT_JSON OUTPUT_DIR BATCH_LIMIT TOTAL_ITEMS EXTRA_ARGS

# ---------- 生成 offset 序列并并发执行 ----------
# 例如：0, 100, 200, ... 直到 TOTAL_ITEMS-1
seq 0 "$BATCH_LIMIT" $(( TOTAL_ITEMS > 0 ? TOTAL_ITEMS-1 : 0 )) | \
  xargs -I{} -P "$NUM_JOBS" bash -c '
      START={}
      END=$(( START + BATCH_LIMIT - 1 ))
      if (( END >= TOTAL_ITEMS )); then END=$(( TOTAL_ITEMS - 1 )); fi
      CUR_LIMIT=$(( END - START + 1 ))
      OUTFILE="${OUTPUT_DIR}/resultscores_${START}_${END}.json"

      # 如结果已存在则跳过，避免重复计算（可按需删除此判断）
      if [[ -s "$OUTFILE" ]]; then
        echo "⏭️  [$$] 已存在，跳过：rows ${START}-${END} -> ${OUTFILE}"
        exit 0
      fi

      echo "▶️  [$$]  rows ${START}-${END} (limit=${CUR_LIMIT}) -> ${OUTFILE}"
      python "$PY_SCRIPT" \
          --input  "$INPUT_JSON" \
          --output "$OUTFILE" \
          --limit  "$CUR_LIMIT" \
          --offset "$START" \
          $EXTRA_ARGS
  '

echo "✅ 全部任务完成，结果文件位于: $OUTPUT_DIR/"
