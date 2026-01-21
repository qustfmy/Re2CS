DATA_FILE="./dataset/php/train.jsonl"      # 数据文件
OUT_DIR="./data/qwen/php/train"           # 输出目录
MAX_SAMPLES=${MAX_SAMPLES:-10}              # 每个进程处理多少条
NUM_JOBS=${NUM_JOBS:-2}                     # 并发进程数量
BASE_START=${BASE_START:-0}                 # ⭐新增：起始下标（默认 0）
MODEL_CHAT="DeepSeek-V3"                  # Steps 1-4 使用的模型
MODEL_REASONER="DeepSeek-R1"              # Step 5 使用的模型
LOG_DIR="./logs_enhanced"                   # 日志目录
PYTHON_BIN="python"                         # 或者 python3

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

echo "从下标 $BASE_START 开始"
echo "启动 $NUM_JOBS 个并发任务，每个任务处理 $MAX_SAMPLES 条..."

for ((i=0; i<NUM_JOBS; i++)); do
  START_INDEX=$((BASE_START + i * MAX_SAMPLES))
  LOG_FILE="${LOG_DIR}/job_${START_INDEX}.log"

  echo "  - 启动任务 $i: start-index=${START_INDEX}, max-samples=${MAX_SAMPLES}"

  # 后台启动一个进程
  $PYTHON_BIN enhanced_code_summary.py \
    --data "$DATA_FILE" \
    --output-dir "$OUT_DIR" \
    --start-index "$START_INDEX" \
    --max-samples "$MAX_SAMPLES" \
    --model_chat "$MODEL_CHAT" \
    --model_reasoner "$MODEL_REASONER" \
    > "$LOG_FILE" 2>&1 &
done

echo "所有任务已启动，等待全部结束..."
wait
echo "全部任务结束。日志在：$LOG_DIR"
