#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Paths
ROOT="/path/to/dir/containing/all_diseases"
LABELS_ROOT="/path/to/amos-clf-tr-val/labels"

# Inference settings
SPLIT="val"
BATCH_SIZE=256
NUM_WORKERS=16

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
    echo "Running inference for ${TARGET} on ${SPLIT} split ..."
    python cvpr26_inference_LP.py \
        --embeds_root "$ROOT" \
        --labels_root "$LABELS_ROOT" \
        --target "$TARGET" \
        --split "$SPLIT" \
        --ckpt_dir "$ROOT/$TARGET/results" \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS
    echo ""
done
