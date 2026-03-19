# CVPR 2026: Foundation Models for 3D Computed Tomography

This repository hosts evaluation of foundation model adaptability.

1. Linear probing: assessing the intrinsic discriminative power of pretrained representations under frozen backbones.

2. Embedding aggregation optimization: exploring effective adaptation strategies, including head design, learning schedules, and partial versus full parameter optimization.

It hosts scripts used for the following:
- Feature extraction
- Linear probing
- Inference & evaluation

`data_utils/` hosts:
- `get_fg_mask.py`: Foreground mask generation

## Installation
```bash
cd CVPR26-3DCTFMCompetition
# Create virtual environment with uv
uv venv --python 3.12
source .venv/bin/activate
# Install the package
uv pip install -e .
```


## Download classification data
```bash
bash download_data.sh
```

## Usage
The usage instruction is based on CT-NEXUS docker available [here for Task 1 (LP)](https://drive.google.com/file/d/126Z7bR25QWH6HDNyzn6z1zIkxZN3CiTU/view) and [here for Task 2 (EAO)](https://drive.google.com/file/d/1nPmeVhkda8rMXT6s-cXmlWD0L9lzzeLz/view).

### Download Docker Images
```bash
# Download Task 1 (LP) docker image
bash download_LP_image_example.sh

# Download Task 2 (EAO) docker image
bash download_EAO_image_example.sh
```

### 0. Test demo cases using Docker
Place [test_demo](https://huggingface.co/datasets/kmin06/CVPR26-3DCTFMCompetition/tree/main/AMOS-clf-tr-val/test_demo) in the current directory.

For Task 1 (LP):
```
ls test_demo
mkdir -p test_demo_outputs_ROI
mkdir -p test_demo_outputs_non-ROI
docker load -i ctnexus_lp.tar.gz

chmod -R 777 .

## for Non-ROI disease
docker container run --gpus "device=0" -m 32G --name ctnexus_lp --rm  -v $PWD/test_demo/:/workspace/inputs/ -v $PWD/test_demo_outputs_non-ROI/:/workspace/outputs/ ctnexus_lp:latest /bin/bash -c "sh extract_feat_LP.sh"
ls test_demo_outputs_non-ROI

## for ROI disease
docker container run --gpus "device=0" -m 32G --name ctnexus_lp --rm -e MASKS_DIR=/workspace/inputs/fg_masks/adrenal_hyperplasia -v $PWD/test_demo/:/workspace/inputs/ -v $PWD/test_demo_outputs_ROI/:/workspace/outputs/ ctnexus_lp:latest /bin/bash -c "sh extract_feat_LP.sh"
ls test_demo_outputs_ROI
```

For Task 2 (EAO):
```
ls test_demo
mkdir -p test_demo_outputs_ROI_eao
mkdir -p test_demo_outputs_non-ROI_eao
docker load -i ctnexus_eao.tar.gz

chmod -R 777 .

## for Non-ROI disease
docker container run --gpus "device=0" -m 32G --name ctnexus_eao --rm  -v $PWD/test_demo/:/workspace/inputs/ -v $PWD/test_demo_outputs_non-ROI_eao/:/workspace/outputs/ ctnexus_eao:latest /bin/bash -c "sh extract_feat_EAO.sh"
ls test_demo_outputs_non-ROI_eao

## for ROI disease
docker container run --gpus "device=0" -m 32G --name ctnexus_eao --rm -e MASKS_DIR=/workspace/inputs/fg_masks/adrenal_hyperplasia -v $PWD/test_demo/:/workspace/inputs/ -v $PWD/test_demo_outputs_ROI_eao/:/workspace/outputs/ ctnexus_eao:latest /bin/bash -c "sh extract_feat_EAO.sh"
ls test_demo_outputs_ROI_eao
```
In case of permission error, please use `chmod -R 777 .`
We will use `extract_feat_LP.sh` for **Task 1** linear probing, and `extract_feat_EAO.sh` for **Task 2** Embedding aggregation optimization.

### 1. Feature Extraction using Docker
Extracts embeddings from CT scans using foundation models packaged in Docker containers. The output embeddings are saved in `./path/to/results/target_disease/embeddings`

For non-ROI diseases:
```bash
python cvpr26_extract_feat_docker_LP.py \
    -i /path/to/amos-clf-tr-val/images/target \
    -o ./path/to/results_LP \
    -d ./path/to/docker/folder \
```

For ROI diseases:
```bash
python cvpr26_extract_feat_docker_LP.py \
    -i /path/to/amos-clf-tr-val/images/target \
    -m /path/to/foreground/fg_masks/target \
    -o ./path/to/results_LP \
    -d ./path/to/docker/folder \
```
We will use `cvpr26_extract_feat_docker_LP.py` for **Task 1** linear probing, and `cvpr26_extract_feat_docker_EAO.py` for **Task 2** Embedding aggregation optimization.


ROI diseases:
  splenomegaly,
  adrenal_hyperplasia,
  fatty_liver,
  cholecystitis,
  liver_calcifications,
  hydronephrosis,
  gallstone,
  liver_lesion,
  kidney_stone,
  liver_cyst,
  renal_cyst

Non-ROI diseases: 
  atherosclerosis,
  colorectal_cancer,
  ascites,
  lymphadenopathy

### 2. Linear Probing (LP)
Trains and evaluates a linear classifier on top of frozen foundation model embeddings.

```bash
python run_LP.py \
    --embeds_root /path/to/embeddings \
    --target ${disease} \
    --out_dir "$OUT_ROOT/${disease}/results" \
    --use_wandb
```
Please refer to [run_LP.sh](run_LP.sh) to run linear probing across all targets.

### 3. Embedding Aggregation Optimization (EAO)
Trains and evaluates a linear classifier on top of frozen foundation model embeddings.

```bash
python run_EAO.py \
    --embeds_root /path/to/embeddings \
    --target ${disease} \
    --out_dir "$OUT_ROOT/${disease}/results" \
    --use_wandb
```
Please refer to [run_EAO.sh](run_EAO.sh) to run embedding aggregation optimization across all targets.

### 4. Inference & evaluation
Runs inference using the trained linear probe head and evaluates performance on test or validation splits.

```bash
# For LP
python cvpr26_inference_LP.py \
    --embeds_root "/path/to/dir/containing/all_diseases" \
    --labels_root "/path/to/amos-clf-tr-val/labels" \
    --target "target_disease" \
    --split "val" \
    --ckpt_dir "/path/to/dir/containing/all_diseases/target_disease/results" \
    --batch_size 256 \
    --num_workers 4
```

```bash
# For EAO
python cvpr26_inference_EAO.py \
    --embeds_root "/path/to/dir/containing/all_diseases" \
    --labels_root "/path/to/amos-clf-tr-val/labels" \
    --target "target_disease" \
    --split "val" \
    --ckpt_dir "/path/to/dir/containing/all_diseases/target_disease/results" \
    --batch_size 256 \
    --num_workers 4
```
Please refer to [run_inference_LP_val.sh](run_inference_LP_val.sh) and [run_inference_EAO_val.sh](run_inference_EAO_val.sh) to run inference and evaluation across all targets.

**Outputs:**
- `{split}_metrics.csv`: Aggregate metrics including accuracy, F1, AUROC, average precision, sensitivity, specificity, and balanced accuracy
- `{split}_per_sample_predictions.csv`: Per-sample predictions with filename, true label, predicted label, logits, and class probabilities

### 5. Organize evaluation metrics and predictions
Organizes evaluation metrics and predictions from all targets into a single CSV file.

```bash
python cvpr26_organize_eval_metrics_and_predictions.py \
    --csv_root "Root/directory/containing/disease/subfolders" \
    --label_dir "/path/to/your/amos-clf-tr-val/labels"
```
