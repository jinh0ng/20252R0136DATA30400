#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-$HOME/20252R0136DATA30400/project_release/Amazon_products}"
REPO_DIR="${REPO_DIR:-$HOME/20252R0136DATA30400}"

EMB_SUBDIR="${EMB_SUBDIR:-embeddings_gte}"
PID_SUBDIR="${PID_SUBDIR:-embeddings_gte}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data_gte}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$REPO_DIR/outputs_gte}"

MODEL_NAME="${MODEL_NAME:-thenlper/gte-base}"

EMB_BATCH_SIZE="${EMB_BATCH_SIZE:-64}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"

TOP_K_CORE="${TOP_K_CORE:-3}"

GAT_EPOCHS="${GAT_EPOCHS:-10}"
GAT_BATCH_SIZE="${GAT_BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NUM_HEADS="${NUM_HEADS:-4}"
LR="${LR:-1e-3}"
SEED="${SEED:-42}"

ROUNDS="${ROUNDS:-2}"
EPOCHS_ST="${EPOCHS_ST:-3}"
ALPHA_EMA="${ALPHA_EMA:-0.99}"
T_POS="${T_POS:-0.9}"
T_NEG="${T_NEG:-0.1}"
EMA_PRED_BATCH="${EMA_PRED_BATCH:-1024}"

THRESHOLD="${THRESHOLD:-0.75}"
MIN_LABELS="${MIN_LABELS:-2}"
MAX_LABELS="${MAX_LABELS:-3}"
PRED_BATCH="${PRED_BATCH:-1024}"
OUT_NAME="${OUT_NAME:-submission_gte.csv}"

FORCE_EMB="${FORCE_EMB:-0}"

echo_section () {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

need_cmd () {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing command: $1"; exit 1; }
}

need_cmd python3

echo_section "ENV SUMMARY (GTE)"
echo "REPO_DIR    = $REPO_DIR"
echo "DATASET_DIR = $DATASET_DIR"
echo "DATA_DIR    = $DATA_DIR"
echo "OUTPUTS_DIR = $OUTPUTS_DIR"
echo "EMB_SUBDIR  = $EMB_SUBDIR"
echo "PID_SUBDIR  = $PID_SUBDIR"
echo "MODEL_NAME  = $MODEL_NAME"
mkdir -p "$DATA_DIR" "$OUTPUTS_DIR"

TRAIN_EMB="$DATASET_DIR/$EMB_SUBDIR/train_doc_mpnet.npy"
TEST_EMB="$DATASET_DIR/$EMB_SUBDIR/test_doc_mpnet.npy"
CLASS_EMB="$DATASET_DIR/$EMB_SUBDIR/class_name_mpnet.npy"
PID_TRAIN="$DATASET_DIR/$PID_SUBDIR/pid_list_train.npy"
PID_TEST="$DATASET_DIR/$PID_SUBDIR/pid_list_test.npy"

SILVER_Y="$DATA_DIR/silver/y_silver.npy"
ADJ_NPY="$DATA_DIR/graph/adj.npy"

echo_section "1) MAKE EMBEDDINGS (GTE) - SKIP IF EXISTS"
if [[ "$FORCE_EMB" == "1" ]]; then
  FORCE_FLAG="--force"
else
  FORCE_FLAG=""
fi

if [[ -f "$TRAIN_EMB" && -f "$TEST_EMB" && -f "$CLASS_EMB" && -f "$PID_TRAIN" && -f "$PID_TEST" && "$FORCE_EMB" != "1" ]]; then
  echo "Embeddings already exist. Skipping."
else
  python3 "$REPO_DIR/src/make_embeddings_mpnet.py" \
    --dataset_dir "$DATASET_DIR" \
    --emb_subdir "$EMB_SUBDIR" \
    --model_name "$MODEL_NAME" \
    --batch_size "$EMB_BATCH_SIZE" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    $FORCE_FLAG
fi

echo_section "2) BUILD SILVER LABELS + ADJ"
python3 "$REPO_DIR/src/build_silver.py" \
  --dataset_dir "$DATASET_DIR" \
  --data_dir "$DATA_DIR" \
  --outputs_dir "$OUTPUTS_DIR" \
  --emb_subdir "$EMB_SUBDIR" \
  --pid_subdir "$PID_SUBDIR" \
  --train_doc_name train_doc_mpnet.npy \
  --test_doc_name test_doc_mpnet.npy \
  --class_name_name class_name_mpnet.npy \
  --top_k_core "$TOP_K_CORE"

echo_section "3) TRAIN GAT"
python3 "$REPO_DIR/src/train_gat.py" \
  --dataset_dir "$DATASET_DIR" \
  --data_dir "$DATA_DIR" \
  --outputs_dir "$OUTPUTS_DIR" \
  --emb_subdir "$EMB_SUBDIR" \
  --pid_subdir "$PID_SUBDIR" \
  --train_doc_name train_doc_mpnet.npy \
  --test_doc_name test_doc_mpnet.npy \
  --class_name_name class_name_mpnet.npy \
  --y_silver_path "$SILVER_Y" \
  --adj_path "$ADJ_NPY" \
  --epochs "$GAT_EPOCHS" \
  --batch_size "$GAT_BATCH_SIZE" \
  --hidden_dim "$HIDDEN_DIM" \
  --num_heads "$NUM_HEADS" \
  --lr "$LR" \
  --seed "$SEED" \
  --ckpt_name gat.pt

echo_section "4) EMA SELF-TRAIN"
python3 "$REPO_DIR/src/self_train.py" \
  --dataset_dir "$DATASET_DIR" \
  --data_dir "$DATA_DIR" \
  --outputs_dir "$OUTPUTS_DIR" \
  --emb_subdir "$EMB_SUBDIR" \
  --pid_subdir "$PID_SUBDIR" \
  --train_doc_name train_doc_mpnet.npy \
  --test_doc_name test_doc_mpnet.npy \
  --class_name_name class_name_mpnet.npy \
  --y_silver_path "$SILVER_Y" \
  --adj_path "$ADJ_NPY" \
  --student_ckpt "$OUTPUTS_DIR/checkpoints/gat.pt" \
  --hidden_dim "$HIDDEN_DIM" \
  --num_heads "$NUM_HEADS" \
  --batch_size "$GAT_BATCH_SIZE" \
  --lr "$LR" \
  --seed "$SEED" \
  --rounds "$ROUNDS" \
  --epochs_st "$EPOCHS_ST" \
  --alpha_ema "$ALPHA_EMA" \
  --t_pos "$T_POS" \
  --t_neg "$T_NEG" \
  --ema_pred_batch "$EMA_PRED_BATCH"

echo_section "5) MAKE SUBMISSION (EMA TEACHER)"
python3 "$REPO_DIR/src/submission.py" \
  --dataset_dir "$DATASET_DIR" \
  --data_dir "$DATA_DIR" \
  --outputs_dir "$OUTPUTS_DIR" \
  --emb_subdir "$EMB_SUBDIR" \
  --pid_subdir "$PID_SUBDIR" \
  --test_doc_name test_doc_mpnet.npy \
  --class_name_name class_name_mpnet.npy \
  --adj_path "$ADJ_NPY" \
  --teacher_ckpt "$OUTPUTS_DIR/checkpoints/ema_teacher.pt" \
  --hidden_dim "$HIDDEN_DIM" \
  --num_heads "$NUM_HEADS" \
  --threshold "$THRESHOLD" \
  --min_labels "$MIN_LABELS" \
  --max_labels "$MAX_LABELS" \
  --pred_batch "$PRED_BATCH" \
  --seed "$SEED" \
  --out_name "$OUT_NAME"

echo_section "DONE (GTE)"
echo "Submission saved at:"
echo "  $OUTPUTS_DIR/submissions/$OUT_NAME"
