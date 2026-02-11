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

# Import the attention pooling module
from models.attention_pooling_multilayers import MultiLayersCrossAttentionPooling


class SpatialFeaturesDataset(Dataset):
    """Dataset for spatial features extracted by ResEncoderPatchLatent"""
    def __init__(self, embeds_dir, csv_path, split, target_column=None):
        # Load CSV and filter by split
        df = pd.read_csv(csv_path)
        split_df = df[df['split'] == split].copy()

        # Build paths and label mapping
        self.paths = []
        self.label_mapping = {}

        for _, row in split_df.iterrows():
            # Extract filename without extension
            case_id = row['case_id']
            filename = case_id.split('.nii.gz')[0] if '.nii.gz' in case_id else case_id
            filename_base = filename.replace('.h5', '')  # Base filename for mapping

            # Construct path with .h5 extension
            h5_filename = filename_base + '.h5'
            path = os.path.join(embeds_dir, h5_filename)

            # Only add if file exists
            if os.path.exists(path):
                self.paths.append(path)
                self.label_mapping[filename_base] = int(row[target_column])
            else:
                print(f"Warning: File not found, skipping: {path}")

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

        # Load spatial features from h5 file
        with h5py.File(path, 'r') as hf:
            # Features are saved as [D, H, W, L] or flattened
            y_hat = torch.tensor(hf['y_hat'][:]).float()

        return y_hat, y

    def _get_num_classes(self):
        # Get unique labels from the CSV mapping
        all_labels = set(self.label_mapping.values())
        return len(all_labels)


def load_attention_head(ckpt_path, embed_dim, num_classes, query_num, num_heads,
                       dropout, num_layers, ffn_mult, device):
    """Load attention pooling head from checkpoint"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]

    # Strip the "heads.{head_name}." prefix from state dict keys
    # The checkpoint has keys like "heads.head_lr_1e_03.query_embeddings.weight"
    stripped = {}
    for k, v in sd.items():
        # Remove "heads.{head_name}." prefix
        if k.startswith("heads."):
            parts = k.split(".", 2)  # Split into ["heads", "head_lr_XXX", "remaining"]
            if len(parts) >= 3:
                stripped[parts[2]] = v

    # Create attention pooling head with the same hyperparameters
    head = MultiLayersCrossAttentionPooling(
        embed_dim=embed_dim,
        query_num=query_num,
        num_classes=num_classes,
        num_heads=num_heads,
        dropout=dropout,
        num_layers=num_layers,
        ffn_mult=ffn_mult
    )

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
    """Select best checkpoint based on val balanced accuracy"""
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]

    if not ckpt_files:
        raise ValueError(f"No checkpoint files found in {ckpt_dir}")

    # Extract balanced_acc scores from filenames
    # Format: best_overall_balanced_acc{score}_{head_name}_ep{epoch}.pth
    ckpt_files_acc = []
    for x in ckpt_files:
        try:
            # Extract the score after "balanced_acc" or other monitor metric
            if 'balanced_acc' in x:
                score_str = x.split('balanced_acc')[-1].split('_')[0]
            elif 'auroc' in x:
                score_str = x.split('auroc')[-1].split('_')[0]
            else:
                # Try to extract any float after "best_overall_"
                score_str = x.split('best_overall_')[-1].split('_')[0]
            ckpt_files_acc.append(float(score_str))
        except (ValueError, IndexError):
            print(f"Warning: Could not parse score from filename {x}, skipping")
            ckpt_files_acc.append(-1.0)

    # Get the one with highest score
    max_idx = np.argmax(ckpt_files_acc)
    best_ckpt = ckpt_files[max_idx]
    print(f"Selected best checkpoint: {best_ckpt} with score {ckpt_files_acc[max_idx]:.4f}")

    return best_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeds_root", default='/home/jma/Documents/cryoSumin/CT_FM/data/embeddings/features_ct_nexus_attn_pool',
                    help="Root directory for embeddings")
    ap.add_argument("--labels_root", type=str,
                    default='/home/jma/Documents/cryoSumin/CT_FM/data/raw_data_classify/amos-clf-tr-val/labels',
                    help='Root directory containing CSV files for labels')
    ap.add_argument("--target", type=str, default='splenomegaly',
                    help='target name (used to construct CSV filename: target.csv)')
    ap.add_argument("--split", type=str, default="test",
                    help="Split to run inference on (default: test)")
    ap.add_argument("--ckpt_dir", default=None,
                    help="Directory containing checkpoint (if None, will use embeds_root/target/attn_pool_results)")

    # Attention pooling hyperparameters (must match training)
    ap.add_argument("--query_num", type=int, default=2,
                    help='Number of learnable queries for attention pooling')
    ap.add_argument("--num_heads", type=int, default=4,
                    help='Number of attention heads')
    ap.add_argument("--num_layers", type=int, default=2,
                    help='Number of cross-attention layers')
    ap.add_argument("--dropout", type=float, default=0.0,
                    help='Dropout rate')
    ap.add_argument("--ffn_mult", type=int, default=1,
                    help='FFN hidden dimension multiplier')

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    # Set up paths
    embeds_dir = os.path.join(args.embeds_root, args.target)

    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(embeds_dir, 'attn_pool_results')

    print(f'Using checkpoint directory: {args.ckpt_dir}')

    ckpt = select_best_ckpt(args.ckpt_dir)
    ckpt_path = os.path.join(args.ckpt_dir, ckpt)

    # Construct CSV path from labels_root and target
    labels_csv = os.path.join(args.labels_root, args.target + '.csv')
    print(f'Loading labels from: {labels_csv}')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataset
    ds = SpatialFeaturesDataset(embeds_dir, labels_csv, split=args.split, target_column=args.target)

    if len(ds) == 0:
        raise ValueError(f"No samples found in {args.split} split")

    num_classes = ds._get_num_classes()

    # Get feature dimensions from first sample
    first_sample = ds[0][0]
    if first_sample.dim() == 4:  # [D, H, W, L]
        embed_dim = first_sample.shape[0]
        print(f'Detected spatial features with shape: {first_sample.shape}')
    elif first_sample.dim() == 2:  # [D, H*W*L]
        embed_dim = first_sample.shape[0]
        print(f'Detected flattened spatial features with shape: {first_sample.shape}')
    else:
        raise ValueError(f'Unexpected feature shape: {first_sample.shape}')

    print(f'Embed dimension: {embed_dim}, num_classes: {num_classes}')

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Load attention pooling head
    head, head_name, val_metrics_from_ckpt = load_attention_head(
        ckpt_path,
        embed_dim,
        num_classes,
        args.query_num,
        args.num_heads,
        args.dropout,
        args.num_layers,
        args.ffn_mult,
        device
    )

    print(f'Loaded head: {head_name}')
    if val_metrics_from_ckpt:
        print('Validation metrics from checkpoint:')
        for k, v in val_metrics_from_ckpt.items():
            print(f'  {k}: {v:.4f}')

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

            # Attention pooling forward pass
            logits = head(xb)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1)

            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

            # Extract filenames for this batch
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + xb.size(0), len(ds.paths))
            batch_paths = ds.paths[batch_start:batch_end]
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

    print(f"\nInference metrics on {args.split} split:")
    for k, v in computed_cpu.items():
        print(f"  {k}: {v:.4f}")

    # Save aggregate metrics as pandas dataframe
    df_metrics = pd.DataFrame([computed_cpu]).round(4)
    metrics_csv_path = os.path.join(args.ckpt_dir, f"{args.split}_metrics.csv")
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f'\nSaved aggregate metrics to {metrics_csv_path}')

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


if __name__ == "__main__":
    main()