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
The usage instruction is based on CT-NEXUS docker available [here](https://drive.google.com/file/d/1ke8_l0KRzQEzCOvvf3t_9yfVhJWOtQbq/view?usp=drive_link).

### 0. Test demo cases using Docker
Place [test_demo](https://huggingface.co/datasets/kmin06/CVPR26-3DCTFMCompetition/tree/main/AMOS-clf-tr-val/test_demo) in the current directory
```
ls test_demo
mkdir -p test_demo_outputs_ROI
mkdir -p test_demo_outputs_non-ROI
docker load -i ctnexus.tar.gz

## for Non-ROI disease
docker container run --gpus "device=0" -m 32G --name ctnexus --rm  -v $PWD/test_demo/:/workspace/inputs/ -v $PWD/test_demo_outputs_non-ROI/:/workspace/outputs/ ctnexus:latest /bin/bash -c "sh extract_feat_LP.sh"
ls test_demo_outputs_non-ROI

## for ROI disease
docker container run --gpus "device=0" -m 32G --name ctnexus --rm -e MASKS_DIR=/workspace/inputs/fg_masks/adrenal_hyperplasia -v $PWD/test_demo/:/workspace/inputs/ -v $PWD/test_demo_outputs_ROI/:/workspace/outputs/ ctnexus:latest /bin/bash -c "sh extract_feat_LP.sh"
ls test_demo_outputs_ROI
```
In case of permission error, please use `chmod -R 777 .`
We will use `extract_feat_LP.sh` for **Task 1** linear probing, and `extract_feat_EAO.sh` for **Task 2** Embedding aggregation optimization.

### 1. Feature Extraction using Docker
Extracts embeddings from CT scans using foundation models packaged in Docker containers.

For non-ROI diseases:
```bash
python cvpr26_extract_feat_docker_LP.py \
    -i /path/to/amos-clf-tr-val/images/target \
    -o ./path/to/results \
    -d ./path/to/docker/folder \
```

For ROI diseases:
```bash
python cvpr26_extract_feat_docker_LP.py \
    -i /path/to/amos-clf-tr-val/images/target \
    -m /path/to/foreground/fg_masks/target \
    -o ./path/to/results \
    -d ./path/to/docker/folder \
```
We will use `cvpr26_extract_feat_docker_LP.py` for **Task 1** linear probing, and `cvpr26_extract_feat_docker_EAO.sh` for **Task 2** Embedding aggregation optimization.


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

- `-i, --images`: Input directory containing CT image files
- `-l, --labels`: Directory with CSV label files for the target disease (data split information is used)
- `-o, --output`: Output directory where extracted features will be saved
- `-m, --mask_root`: Path to the foreground mask for ROI disease only. Set to `None` for Non-ROI.
- `-d, --dockers`: Directory containing team Docker images with foundation models
- `--target`: Target disease for classification (e.g., splenomegaly, pneumonia).

### 2. Linear Probing (LP)
Trains and evaluates a linear classifier on top of frozen foundation model embeddings.

```bash
python run_LP.py \
    --embeds_root /path/to/embeddings/features_LP \
    --labels_root /path/to/labels \
    --disease splenomegaly \
    --monitor_metric balanced_acc \
    --use_wandb
```
- `--embeds_root`: Root directory containing extracted feature embeddings
- `--labels_root`: Directory with ground truth CSV labels
- `--disease`: Disease classification task (e.g., splenomegaly, pneumonia, emphysema)
- `--monitor_metric`: Metric to monitor during training (balanced_acc, auroc, f1, etc.)
- `--use_wandb`: Enable Weights & Biases logging for experiment tracking (optional)

### 3. Embedding Aggregation Optimization (EAO)
Trains and evaluates a linear classifier on top of frozen foundation model embeddings.

```bash
python run_LP.py \
    --embeds_root /path/to/embeddings/features_EAO \
    --labels_root /path/to/labels \
    --disease splenomegaly \
    --monitor_metric balanced_acc \
    --use_wandb
```
- `--embeds_root`: Root directory containing extracted feature embeddings
- `--labels_root`: Directory with ground truth CSV labels
- `--disease`: Disease classification task (e.g., splenomegaly, pneumonia, emphysema)
- `--monitor_metric`: Metric to monitor during training (balanced_acc, auroc, f1, etc.)
- `--use_wandb`: Enable Weights & Biases logging for experiment tracking (optional)

### 3. Inference & evaluation
Runs inference using the trained linear probe head and evaluates performance on test or validation splits.

```bash
python cvpr26_inference_linear_probe.py \
    --embeds /path/to/embeddings/features_ct_fm/disease_name \
    --labels_root /path/to/labels \
    --target disease_name \
    --split val \
    --ckpt_dir /path/to/embeddings/features_ct_fm/disease_name/results \
    --batch_size 256 \
    --num_workers 4
```
- `--embeds`: Path to embeddings directory containing train/val/test split subdirectories
- `--labels_root`: Root directory containing CSV files with ground truth labels
- `--target`: Target disease name (used to construct CSV filename: target.csv)
- `--split`: Data split to run inference on (default: test)
- `--ckpt_dir`: Directory containing checkpoints saved during training (automatically selects best checkpoint based on validation balanced accuracy)
- `--batch_size`: Batch size for inference (default: 256)
- `--num_workers`: Number of data loading workers (default: 4)

**Outputs:**
- `{split}_metrics.csv`: Aggregate metrics including accuracy, F1, AUROC, average precision, sensitivity, specificity, and balanced accuracy
- `{split}_per_sample_predictions.csv`: Per-sample predictions with filename, true label, predicted label, logits, and class probabilities



## Helper functions
### 4. Generate Foreground Masks (For ROI-disease only)
Creates foreground masks for region-of-interest based disease classification tasks.

```bash
python data_utils/get_fg_mask.py \
    --csv-dir /path/to/labels \
    --labels-dir /path/to/totalseg \
    --output-dir /path/to/fg_masks \
    --num-workers 16
```
- `--csv-dir`: Directory containing disease label CSV files
- `--labels-dir`: Directory with organ segmentation masks (e.g., from TotalSegmentator)
- `--output-dir`: Output directory where foreground masks will be saved
- `--num-workers`: Number of parallel workers for mask generation (default: 16)