#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=<your_wandb_api_key>

disease_list=(
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

for disease in "${disease_list[@]}"; do
    echo "Running linear probing for ${disease} ..."
    python run_linear_probe.py \
        --embeds_root /home/jma/Documents/cryoSumin/CT_FM/data/embeddings/features_ct_fm_public_MultiStage \
        --disease ${disease} \
        --use_wandb
done

