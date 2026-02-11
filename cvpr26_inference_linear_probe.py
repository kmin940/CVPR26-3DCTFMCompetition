import argparse, os, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import h5py

from torchmetrics import (
    Accuracy,
    F1Score,
    AUROC,
    AveragePrecision,
    Recall,
    Specificity,
    MetricCollection,
)
from metrics.balanced_accuracy import BalancedAccuracy  # same as training
import numpy as np
import pandas as pd

class FeaturesDataset(Dataset):
    def __init__(self, paths, csv_path, target_column=None):
        self.paths = paths
        # Load label mapping from CSV file
        df = pd.read_csv(csv_path)

        # Create label mapping: filename -> label
        # Assuming CSV has columns: 'case_id' (or filename) and target_column
        self.label_mapping = {}
        for _, row in df.iterrows():
            # Extract filename without extension
            filename = row['case_id'].split('.nii.gz')[0] if '.nii.gz' in row['case_id'] else row['case_id']
            filename = filename.replace('.h5', '')  # Also handle .h5 extension
            self.label_mapping[filename] = int(row[target_column])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        # Extract filename without extension from path
        filename = os.path.basename(path).replace('.h5', '')

        # Get label from CSV mapping
        if filename not in self.label_mapping:
            raise ValueError(f"Filename {filename} not found in label mapping")
        y = torch.tensor(self.label_mapping[filename]).long()

        # Load features from h5 file
        with h5py.File(path, 'r') as hf:
            y_hat = torch.tensor(hf['y_hat'][:]).float()

        return y_hat, y

    def _get_num_classes(self):
        # Get unique labels from the CSV mapping
        all_labels = set(self.label_mapping.values())
        return len(all_labels)


def get_paths_in_split(base_dir, split):
    split_dir = os.path.join(base_dir, split)
    all_files = os.listdir(split_dir)
    all_paths = [os.path.join(split_dir, f) for f in all_files if f.endswith('.h5')]
    return all_paths


def load_head(ckpt_path, in_dim, num_classes, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]
    stripped = {k.split(".", 2)[-1]: v for k, v in sd.items()}
    head = nn.Linear(in_dim, num_classes)
    head.load_state_dict(stripped, strict=True)
    head.to(device)
    head.eval()
    return head, ckpt.get("head_name", None), ckpt.get("val_metrics", None)


def build_metrics(num_classes, device):
    base_metrics = MetricCollection({
        "acc": Accuracy(task="multiclass", num_classes=num_classes),
        "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        "auroc": AUROC(task="multiclass", num_classes=num_classes),
        "ap": AveragePrecision(task="multiclass", num_classes=num_classes),
        "sensitivity": Recall(task="multiclass", num_classes=num_classes, average="macro"),
        "specificity": Specificity(task="multiclass", num_classes=num_classes, average="macro"),
        "balanced_acc": BalancedAccuracy(num_classes=num_classes, task="multiclass"),
    }).to(device)
    return base_metrics

def select_best_ckpt(ckpt_dir):
    # select best checkpoint based on val balanced accuracy
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    ckpt_files_acc = ['_'.join(x.split('_')[2:]).split('balanced_acc')[-1].split('auroc')[-1].split('_')[0] for x in ckpt_files]
    # get the one with highest balanced acc
    # use argmax
    max_idx = np.argmax([float(x) for x in ckpt_files_acc])
    best_ckpt = ckpt_files[max_idx]
    print(f"Selected best checkpoint: {best_ckpt} based on val balanced accuracy")
    return best_ckpt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeds", default='/home/jma/Documents/cryoSumin/CT_FM/data/embeddings/features_ct_fm_public_MultiStage/fatty_liver',
                    help="Path to embeddings directory containing train/val/test splits")
    ap.add_argument("--labels_root", type=str,
                    default='/home/jma/Documents/cryoSumin/CT_FM/data/raw_data_classify/amos-clf-tr-val/test/labels',
                    help='Root directory containing CSV files for labels')
    ap.add_argument("--target", type=str, default='fatty_liver',
                    help='target name (used to construct CSV filename: target.csv)')
    ap.add_argument("--split", type=str, default="test",
                    help="Split to run inference on (default: test)")
    ap.add_argument("--ckpt_dir", default='/home/jma/Documents/cryoSumin/CT_FM/data/embeddings/features_ct_fm_public_MultiStage/fatty_liver/results3',
                    help="checkpoint saved by save_single_head(...) in training")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    ckpt = select_best_ckpt(args.ckpt_dir)
    ckpt = os.path.join(args.ckpt_dir, ckpt)

    # Construct CSV path from labels_root and target
    labels_csv = os.path.join(args.labels_root, args.target + '.csv')
    print(f'Loading labels from: {labels_csv}')

    # Load paths from the specified split
    paths = get_paths_in_split(args.embeds, args.split)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataset
    ds = FeaturesDataset(paths, labels_csv, target_column=args.target)
    num_classes = ds._get_num_classes()
    in_dim = ds[0][0].shape[0]

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    head, head_name, val_metrics_from_ckpt = load_head(ckpt, in_dim, num_classes, device)

    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []
    all_filenames = []

    metrics = build_metrics(num_classes, device)

    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(dl):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = head(xb)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1)

            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

            # Extract filenames for this batch
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + xb.size(0), len(paths))
            batch_paths = paths[batch_start:batch_end]
            batch_filenames = [os.path.basename(p).replace('.h5', '') for p in batch_paths]
            all_filenames.extend(batch_filenames)

            metrics.update(logits, yb)

    all_logits = torch.cat(all_logits, 0)
    all_probs = torch.cat(all_probs, 0)
    all_preds = torch.cat(all_preds, 0)
    all_labels = torch.cat(all_labels, 0)

    computed = metrics.compute()
    computed_cpu = {}
    for k, v in computed.items():
        computed_cpu[k] = v.item() if torch.is_tensor(v) else v

    out = {
        "logits": all_logits,
        "probs": all_probs,
        "preds": all_preds,
        "labels": all_labels,
        "head_name": head_name,
        "val_metrics_from_ckpt": val_metrics_from_ckpt,
        "metrics_inference": computed_cpu,
    }

    print("Inference metrics:")
    for k, v in computed_cpu.items():
        print(f"  {k}: {v:.4f}")

    # Save aggregate metrics as pandas dataframe
    df_metrics = pd.DataFrame([computed_cpu]).round(4)
    metrics_csv_path = os.path.join(args.ckpt_dir, f"{args.split}_metrics.csv")
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f'Saved aggregate metrics to {metrics_csv_path}')

    # Save per-sample predictions to CSV
    # Create dataframe with filename, label, prediction, and probabilities for each class
    per_sample_data = {
        'filename': all_filenames,
        'label': all_labels.numpy(),
        'prediction': all_preds.numpy(),
    }

    # Add logits for each class
    for class_idx in range(num_classes):
        per_sample_data[f'logit_class_{class_idx}'] = all_logits[:, class_idx].numpy()

    # Add probabilities for each class
    for class_idx in range(num_classes):
        per_sample_data[f'prob_class_{class_idx}'] = all_probs[:, class_idx].numpy()

    df_per_sample = pd.DataFrame(per_sample_data)
    per_sample_csv_path = os.path.join(args.ckpt_dir, f"{args.split}_per_sample_predictions.csv")
    df_per_sample.to_csv(per_sample_csv_path, index=False)
    print(f'Saved per-sample predictions to {per_sample_csv_path}')

    #os.makedirs(os.path.dirname(args.out_pt) or ".", exist_ok=True)

    # torch.save(out, os.path.join(args.ckpt_dir, "inference_outputs.pt"))


if __name__ == "__main__":
    main()
