#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Config (edit if needed)
# ----------------------------
DATASET_DIR="${DATASET_DIR:-$HOME/new/project_release/Amazon_products}"
DATA_DIR="${DATA_DIR:-$HOME/new/data}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$HOME/new/outputs}"

EMB_SUBDIR="${EMB_SUBDIR:-embeddings_mpnet}"
PID_SUBDIR="${PID_SUBDIR:-embeddings_mpnet}"

# MPNet embedding outputs (under dataset_dir/embeddings_mpnet by default in our scripts)
TRAIN_EMB="$DATASET_DIR/$EMB_SUBDIR/train_doc_mpnet.npy"
TEST_EMB="$DATASET_DIR/$EMB_SUBDIR/test_doc_mpnet.npy"
CLASS_EMB="$DATASET_DIR/$EMB_SUBDIR/class_name_mpnet.npy"
PID_TRAIN="$DATASET_DIR/$PID_SUBDIR/pid_list_train.npy"
PID_TEST="$DATASET_DIR/$PID_SUBDIR/pid_list_test.npy"

# Make embeddings options
EMB_BATCH_SIZE="${EMB_BATCH_SIZE:-64}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"

# Build silver options
TOP_K_CORE="${TOP_K_CORE:-3}"

# Train GAT options (keep notebook-equivalent defaults unless overriding)
GAT_EPOCHS="${GAT_EPOCHS:-10}"
GAT_BATCH_SIZE="${GAT_BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NUM_HEADS="${NUM_HEADS:-4}"
LR="${LR:-1e-3}"
SEED="${SEED:-42}"

# Self-train options (notebook-equivalent defaults)
ROUNDS="${ROUNDS:-2}"
EPOCHS_ST="${EPOCHS_ST:-3}"
ALPHA_EMA="${ALPHA_EMA:-0.99}"
T_POS="${T_POS:-0.9}"
T_NEG="${T_NEG:-0.1}"
EMA_PRED_BATCH="${EMA_PRED_BATCH:-1024}"

# Submission options (notebook-equivalent defaults)
THRESHOLD="${THRESHOLD:-0.65}"
MIN_LABELS="${MIN_LABELS:-2}"
MAX_LABELS="${MAX_LABELS:-3}"
PRED_BATCH="${PRED_BATCH:-1024}"
OUT_NAME="${OUT_NAME:-submission.csv}"

# ----------------------------
# Helpers
# ----------------------------
echo_section () {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

need_cmd () {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1"
    exit 1
  }
}

# ----------------------------
# Pre-checks
# ----------------------------
need_cmd python3

echo_section "ENV SUMMARY"
echo "DATASET_DIR = $DATASET_DIR"
echo "DATA_DIR    = $DATA_DIR"
echo "OUTPUTS_DIR = $OUTPUTS_DIR"
echo "EMB_SUBDIR  = $EMB_SUBDIR"
echo "PID_SUBDIR  = $PID_SUBDIR"

mkdir -p "$DATA_DIR" "$OUTPUTS_DIR"

# ----------------------------
# 1) Make embeddings (skip if already present)
# ----------------------------
echo_section "1) MAKE EMBEDDINGS (MPNet) - SKIP IF EXISTS"
if [[ -f "$TRAIN_EMB" && -f "$TEST_EMB" && -f "$CLASS_EMB" && -f "$PID_TRAIN" && -f "$PID_TEST" ]]; then
  echo "Embeddings already exist. Skipping:"
  echo "  $TRAIN_EMB"
  echo "  $TEST_EMB"
  echo "  $CLASS_EMB"
  echo "  $PID_TRAIN"
  echo "  $PID_TEST"
else
  echo "Embeddings missing; running make_embeddings_mpnet.py"
  python3 src/make_embeddings_mpnet.py \
    --dataset_dir "$DATASET_DIR" \
    --batch_size "$EMB_BATCH_SIZE" \
    --max_seq_length "$MAX_SEQ_LENGTH"
fi

# ----------------------------
# 2) Build silver labels + adj
# ----------------------------
echo_section "2) BUILD SILVER LABELS + ADJ"
python3 src/build_silver.py \
  --dataset_dir "$DATASET_DIR" \
  --data_dir "$DATA_DIR" \
  --outputs_dir "$OUTPUTS_DIR" \
  --emb_subdir "$EMB_SUBDIR" \
  --pid_subdir "$PID_SUBDIR" \
  --train_doc_name train_doc_mpnet.npy \
  --test_doc_name test_doc_mpnet.npy \
  --class_name_name class_name_mpnet.npy \
  --top_k_core "$TOP_K_CORE" \
  --save_adj_pt

# ----------------------------
# 3) Train GAT
# ----------------------------
echo_section "3) TRAIN GAT"
python3 src/train_gat.py \
  --dataset_dir "$DATASET_DIR" \
  --data_dir "$DATA_DIR" \
  --outputs_dir "$OUTPUTS_DIR" \
  --emb_subdir "$EMB_SUBDIR" \
  --pid_subdir "$PID_SUBDIR" \
  --train_doc_name train_doc_mpnet.npy \
  --test_doc_name test_doc_mpnet.npy \
  --class_name_name class_name_mpnet.npy \
  --epochs "$GAT_EPOCHS" \
  --batch_size "$GAT_BATCH_SIZE" \
  --hidden_dim "$HIDDEN_DIM" \
  --num_heads "$NUM_HEADS" \
  --lr "$LR" \
  --seed "$SEED"

# ----------------------------
# 4) EMA self-training
# ----------------------------
echo_section "4) EMA SELF-TRAIN"
python3 src/self_train.py \
  --dataset_dir "$DATASET_DIR" \
  --data_dir "$DATA_DIR" \
  --outputs_dir "$OUTPUTS_DIR" \
  --emb_subdir "$EMB_SUBDIR" \
  --pid_subdir "$PID_SUBDIR" \
  --train_doc_name train_doc_mpnet.npy \
  --test_doc_name test_doc_mpnet.npy \
  --class_name_name class_name_mpnet.npy \
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

# ----------------------------
# 5) Make submission (EMA teacher)
# ----------------------------
echo_section "5) MAKE SUBMISSION (EMA TEACHER)"
python3 src/submission.py \
  --dataset_dir "$DATASET_DIR" \
  --data_dir "$DATA_DIR" \
  --outputs_dir "$OUTPUTS_DIR" \
  --emb_subdir "$EMB_SUBDIR" \
  --pid_subdir "$PID_SUBDIR" \
  --test_doc_name test_doc_mpnet.npy \
  --class_name_name class_name_mpnet.npy \
  --hidden_dim "$HIDDEN_DIM" \
  --num_heads "$NUM_HEADS" \
  --threshold "$THRESHOLD" \
  --min_labels "$MIN_LABELS" \
  --max_labels "$MAX_LABELS" \
  --pred_batch "$PRED_BATCH" \
  --seed "$SEED" \
  --out_name "$OUT_NAME"

echo_section "DONE"
echo "Submission saved at:"
echo "  $OUTPUTS_DIR/submissions/$OUT_NAME"
