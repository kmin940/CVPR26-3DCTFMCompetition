# CVPR26-3DCTFMCompetition
CVPR 2026: Foundation Models for 3D Computed Tomography
This repository contains evaluation scripts for task 1 and task 2 and linear probing scripts for task 3:

Task 1:

Task 2:

Task 3:

## Evaluation
The evaluation script `CVPR25_iter_eval.py` evaluates Docker submissions for the **CVPR25: Foundation Models for Interactive 3D Biomedical Image Segmentation Challenge** using an iterative refinement approach.

### Installation
Installation of packages for the evaluation script:
```
conda create -n cvpr_ctfm_eval python=3.11 -y
conda activate cvpr_ctfm_eval
pip install -r requirements.txt
```

Run the script as follows:

```bash
python CVPR26_eval.py --docker_folder path/to/docker_submissions --test_img_path path/to/test_images --save_path path/to/output --verbose
```

### Arguments
- `--docker_folder` : Path to the directory containing submitted Docker containers (`.tar.gz`).
- `--test_img_path` : Path to the directory containing `.npz` test images.
- `--save_path` : Directory to save segmentation outputs and evaluation metrics.
- `--verbose` *(optional)* : Enables detailed output, including generated click coordinates.
- `--validation_gts_path` Path to validation / test set GT files. This is needed to prevent label leakage (val/test) during the challenge.

### Evaluation Process
1. **Loads Docker submissions** and processes test images one by one.
2. **Initial Prediction:** Uses a bounding box prompt to generate the first segmentation.
3. **Iterative Refinement:** Simulates up to 5 refinement clicks based on segmentation errors.
4. **Performance Metrics:** Computes **Dice Similarity Coefficient AUC (DSC_AUC), Normalized Surface Dice AUC (NSD_AUC), Final DSC, Final NSD, and Inference Time**.
5. **Outputs results** as `.npz` files and a CSV summary.

### Output
- Segmentation results are saved in the specified output directory. 
    -   Final prediction in the `segs` key
    -   All the 6 intermediate predictions in the `all_segs` key
- Metrics for each test case are compiled into a CSV file.

For more details, refer to the challenge page: https://www.codabench.org/competitions/5263/

