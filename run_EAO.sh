#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# Default values
EMBEDS_ROOT="/home/jma/Documents/cryoSumin/CT_FM/data/embeddings/features_EAO_public_MultiStage"
LABELS_ROOT="/home/jma/Documents/cryoSumin/CT_FM/data/raw_data_classify/amos-clf-tr-val/labels"
OUT_ROOT="/home/jma/Documents/cryoSumin/CT_FM/data/results_cvpr26"

# Attention pooling hyperparameters
QUERY_NUM=2
NUM_HEADS=4
NUM_LAYERS=2
DROPOUT=0.0
FFN_MULT=1

# Training hyperparameters
BATCH_SIZE=8
EPOCHS=1000
PATIENCE=50
NUM_WORKERS=16
MONITOR_METRIC="balanced_acc"

# Logging
LOG_INTERVAL=10
TOP_K_CHECKPOINTS=3

# Optional: Enable wandb logging
USE_WANDB=""  # Set to "--use_wandb" to enable
WANDB_PROJECT="attn_pool_cvpr26"

# TARGET="fatty_liver"
TARGETS=(
  splenomegaly
  adrenal_hyperplasia
  fatty_liver
  cholecystitis
  liver_calcifications
  hydronephrosis
  gallstone
  liver_lesion
  kidney_stone
  liver_cyst
  renal_cyst
  atherosclerosis
  colorectal_cancer
  ascites
  lymphadenopathy
)
for TARGET in "${TARGETS[@]}"; do
    echo "Running Embedding Aggregation Optimization for ${TARGET} ..."
    # Run the training script
    python run_EAO.py \
        --embeds_root "$EMBEDS_ROOT" \
        --labels_root "$LABELS_ROOT" \
        --target "$TARGET" \
        --out_dir "$OUT_ROOT/$TARGET/results_EAO" \
        --query_num $QUERY_NUM \
        --num_heads $NUM_HEADS \
        --num_layers $NUM_LAYERS \
        --dropout $DROPOUT \
        --ffn_mult $FFN_MULT \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --patience $PATIENCE \
        --num_workers $NUM_WORKERS \
        --monitor_metric "$MONITOR_METRIC" \
        --log_interval $LOG_INTERVAL \
        --top_k_checkpoints $TOP_K_CHECKPOINTS \
        --wandb_project "$WANDB_PROJECT" \
        $USE_WANDB
done