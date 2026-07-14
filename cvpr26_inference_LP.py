import argparse, os, re, torch
from torch import nn
import torch.nn.functional as F
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
import pandas as pd


class FeaturesDataset(Dataset):
    def __init__(self, embeds_dir, csv_path, split, target_column=None, split_column='split'):
        # Load CSV and filter by split (split_column lets us pass fold0/fold1/...)
        df = pd.read_csv(csv_path)
        # Some split CSVs use 'case_id', autoPET uses 'id' — auto-detect.
        if 'case_id' in df.columns:
            id_col = 'case_id'
        elif 'id' in df.columns:
            id_col = 'id'
        else:
            raise ValueError(f"Neither 'case_id' nor 'id' column found in {csv_path}")
        split_df = df[df[split_column] == split].copy()

        self.paths = []
        self.label_mapping = {}
        # Cases in the split whose embedding file was not produced by the team. Tracked so the inference driver can score them as failed.
        self.missing = []  # list of (filename_base, label)

        for _, row in split_df.iterrows():
            case_id = row[id_col]
            filename = case_id.split('.nii.gz')[0] if '.nii.gz' in case_id else case_id
            filename_base = filename.replace('.h5', '')

            h5_filename = filename_base + '.h5'
            path = os.path.join(embeds_dir, h5_filename)

            if os.path.exists(path):
                self.paths.append(path)
                self.label_mapping[filename_base] = int(row[target_column])
            else:
                print(f"Warning: File not found, treating as failed case: {path}")
                self.missing.append((filename_base, int(row[target_column])))

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        filename = os.path.basename(path).replace('.h5', '')

        if filename not in self.label_mapping:
            raise ValueError(f"Filename {filename} not found in label mapping")
        y = torch.tensor(self.label_mapping[filename]).long()

        with h5py.File(path, 'r') as hf:
            y_hat = torch.tensor(hf['y_hat'][:]).float()

        return y_hat, y

    def _get_num_classes(self):
        all_labels = set(self.label_mapping.values())
        return len(all_labels)


def apply_norm(X, norm, feat_mean, feat_std):
    """Apply the input transform the head was trained under, recorded in its
    checkpoint by run_LP_decays_norm.py. Must match the 'norm' axis exactly:
      * 'raw' (or None for legacy ckpts) -> identity (features as-is).
      * 'l2'  -> L2-normalize to the unit sphere (F.normalize, dim=1).
      * 'std' -> z-score with the train-set per-dim mean/std stored in the ckpt.
      * 'ln'  -> per-sample LayerNorm across the feature dim, no learnable affine.
                 Stateless (like 'l2'), so the norm tag alone reproduces it.
    X is (N, D) on its current device; stats broadcast as (1, D).
    """
    if norm in (None, "raw"):
        return X
    if norm == "l2":
        return F.normalize(X, dim=1)
    if norm == "ln":
        return F.layer_norm(X, (X.size(1),))
    if norm == "std":
        if feat_mean is None or feat_std is None:
            raise ValueError(
                "Checkpoint has norm='std' but no feat_mean/feat_std were saved; "
                "cannot reproduce the standardization transform.")
        return (X - feat_mean.to(X)) / feat_std.to(X)
    raise ValueError(f"Unknown norm '{norm}' in checkpoint.")


def load_head(ckpt_path, in_dim, num_classes, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]
    stripped = {k.split(".", 2)[-1]: v for k, v in sd.items()}
    head = nn.Linear(in_dim, num_classes)
    head.load_state_dict(stripped, strict=True)
    head.to(device)
    head.eval()
    # Input-normalization variant + (for 'std') the train stats. 
    norm = ckpt.get("norm", "raw")
    feat_mean = ckpt.get("feat_mean", None)
    feat_std = ckpt.get("feat_std", None)
    # betas/eps are training-only AdamW params with no effect at inference; carried through purely as traceability tags. Absent on pre-betas/eps-sweep checkpoints.
    betas = ckpt.get("betas", None)
    eps = ckpt.get("eps", None)
    return (head, ckpt.get("head_name", None), ckpt.get("val_metrics", None),
            norm, feat_mean, feat_std, betas, eps)


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


# Checkpoint filenames written by run_LP.py are
#   best_<strategy>_<score>_<head>_ep<N>.pth
# where <score> is a signed %.4f float and <head> starts with "clf_lr_". 
# We parse the strategy out of the filename rather than matching against a hardcoded list, so any new checkpoint-selection strategy added to run_LP.py is discovered automatically here. 
_CKPT_RE = re.compile(
    r"^best_(?P<strategy>.+?)_(?P<score>-?\d+\.\d+)_(?P<head>clf_lr_.+)_ep(?P<epoch>\d+)\.pth$"
)


def _parse_ckpt(fn):
    """(strategy, epoch) parsed from a checkpoint filename, or None if it does not
    match the expected best_<strategy>_<score>_<head>_ep<N>.pth pattern."""
    m = _CKPT_RE.match(fn)
    if not m:
        return None
    return m.group("strategy"), int(m.group("epoch"))


def _epoch_of(fn):
    m = re.search(r"_ep(\d+)\.pth$", fn)
    return int(m.group(1)) if m else -1


def select_ckpts_by_strategy(ckpt_dir):
    """Map each checkpoint strategy present in ckpt_dir to its best .pth.

    Strategies are discovered by parsing filenames, so any strategy run_LP.py
    writes is picked up without maintaining a hardcoded list here. Training saves a
    checkpoint only when a strategy's metric improves, so within a strategy the
    highest-epoch file is the best one. Returns {strategy: filename}. Falls back to
    a single {'best': latest} entry for legacy (pre-strategy) dirs.
    """
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if not ckpt_files:
        raise FileNotFoundError(f"No .pth checkpoint files found in {ckpt_dir}")

    by_strat = {}  # strategy -> (best_epoch, filename)
    for f in ckpt_files:
        parsed = _parse_ckpt(f)
        if parsed is None:
            continue
        strat, ep = parsed
        if strat not in by_strat or ep > by_strat[strat][0]:
            by_strat[strat] = (ep, f)
    selected = {strat: fn for strat, (_, fn) in by_strat.items()}

    if not selected:
        legacy = max(ckpt_files, key=_epoch_of)
        print(f"No strategy-prefixed checkpoints found; using latest: {legacy}")
        selected["best"] = legacy

    for strat in sorted(selected):
        print(f"  [{strat}] -> {selected[strat]}")
    return selected


def load_features(ds, batch_size, num_workers):
    """Run the dataset's real samples once, returning (X, y, filenames) on CPU.

    X is (N, D) features; every strategy head is applied to this same tensor so we
    read each .h5 only once regardless of how many strategy checkpoints we score.
    Per-head input normalization is applied downstream in score_head (the cached
    X stays raw so heads with different norms can each transform it independently).
    """
    if len(ds) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long), []
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    xs, ys, filenames = [], [], []
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(dl):
            xs.append(xb)
            ys.append(yb)
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + xb.size(0), len(ds.paths))
            batch_paths = ds.paths[batch_start:batch_end]
            filenames.extend(os.path.basename(p).replace('.h5', '') for p in batch_paths)
    return torch.cat(xs, 0), torch.cat(ys, 0), filenames


def score_head(head, X, y, filenames, missing, num_classes, device,
               norm="raw", feat_mean=None, feat_std=None):
    """Apply one strategy's head to the cached features (+ failed missing cases).

    The cached X is raw; the head's recorded `norm` (raw/l2/std) is applied here so
    multiple heads with different norms can each transform the same tensor. Returns
    (computed_metrics, logits, probs, preds, labels, out_filenames) with the real
    samples first and the synthetic failed-missing rows appended, matching
    out_filenames order.
    """
    all_logits, all_probs, all_preds, all_labels = [], [], [], []
    out_filenames = list(filenames)

    if X.numel() > 0:
        with torch.no_grad():
            Xn = apply_norm(X.to(device, non_blocking=True), norm, feat_mean, feat_std)
            logits = head(Xn)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1)
        all_logits.append(logits.cpu())
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(y)

    # Missing test cases: treat as failed. For binary this means prob[GT]=0,
    # prob[1-GT]=1; generalized to multiclass by routing mass to (GT+1) % num_classes.
    n_missing = len(missing)
    if n_missing > 0:
        miss_labels = torch.tensor([lbl for _, lbl in missing], dtype=torch.long)
        miss_preds = (miss_labels + 1) % num_classes
        miss_probs = torch.zeros(n_missing, num_classes, dtype=torch.float)
        miss_probs[torch.arange(n_missing), miss_preds] = 1.0
        miss_logits = torch.full((n_missing, num_classes), -10.0)
        miss_logits[torch.arange(n_missing), miss_preds] = 10.0
        all_logits.append(miss_logits)
        all_probs.append(miss_probs)
        all_preds.append(miss_preds)
        all_labels.append(miss_labels)
        out_filenames = out_filenames + [fn for fn, _ in missing]

    all_logits = torch.cat(all_logits, 0) if all_logits else torch.empty(0, num_classes)
    all_probs = torch.cat(all_probs, 0) if all_probs else torch.empty(0, num_classes)
    all_preds = torch.cat(all_preds, 0) if all_preds else torch.empty(0, dtype=torch.long)
    all_labels = torch.cat(all_labels, 0) if all_labels else torch.empty(0, dtype=torch.long)

    metrics = build_metrics(num_classes, device)
    metrics.update(all_logits.to(device), all_labels.to(device))
    computed = metrics.compute()
    computed_cpu = {k: (v.item() if torch.is_tensor(v) else v) for k, v in computed.items()}

    return computed_cpu, all_logits, all_probs, all_preds, all_labels, out_filenames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeds_dir", type=str, required=True,
                    help="Directory of .h5 feature files for a single team")
    ap.add_argument("--csv_path", type=str, required=True,
                    help="Path to 5-fold split CSV (contains fold{N} columns with train/val/test values)")
    ap.add_argument("--target", type=str, required=True,
                    help="Target column name in the CSV")
    ap.add_argument("--fold", type=int, default=0,
                    help="Fold index. Used for the 'fold{fold}' split column and output tagging "
                         "unless --split_column is given.")
    ap.add_argument("--split_column", type=str, default=None,
                    help="CSV column to filter the split on. Defaults to 'fold{fold}'. "
                         "Pass 'split' for single-split (non-fold) test CSVs.")
    ap.add_argument("--ckpt_dir", type=str, required=True,
                    help="Per-fold checkpoint dir (e.g. <team>/results/fold{N}/)")
    ap.add_argument("--split", type=str, default="test",
                    help="Split value within the split column (default: test)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    fold_col = args.split_column if args.split_column else f"fold{args.fold}"
    print(f"=== Inference fold={args.fold} (column: {fold_col}, split={args.split}) ===")
    print(f"  embeds_dir: {args.embeds_dir}")
    print(f"  csv_path:   {args.csv_path}")
    print(f"  ckpt_dir:   {args.ckpt_dir}")

    # One checkpoint per strategy (val_balanced_acc / val_auroc / val_loss /
    # val_balacc_auroc); legacy dirs collapse to a single {'best': ...}.
    selected = select_ckpts_by_strategy(args.ckpt_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = FeaturesDataset(args.embeds_dir, args.csv_path, split=args.split,
                         target_column=args.target, split_column=fold_col)
    if len(ds) == 0 and not ds.missing:
        raise RuntimeError(f"No samples found for {fold_col}=={args.split} under {args.embeds_dir}")

    # Derive num_classes / in_dim from one checkpoint (architecture is identical
    # across strategies) so we stay consistent with training on tiny test splits.
    _peek = torch.load(os.path.join(args.ckpt_dir, next(iter(selected.values()))),
                       map_location="cpu")
    _sd = _peek["state_dict"]
    _w_key = next(k for k in _sd.keys() if k.endswith("weight"))
    num_classes, in_dim = _sd[_w_key].shape
    num_classes = int(num_classes); in_dim = int(in_dim)
    del _peek, _sd

    # Read every .h5 once; each strategy head is applied to the same feature tensor
    # (its own norm transform is applied per-head inside score_head).
    X, y, filenames = load_features(ds, args.batch_size, args.num_workers)
    n_real = len(ds)
    n_missing = len(ds.missing)
    if n_missing > 0:
        print(f"Including {n_missing} missing case(s) as failed predictions in metrics")

    for strat, ckpt_name in selected.items():
        # Legacy single-checkpoint dirs keep the original unsuffixed output filenames.
        suffix = "" if strat == "best" else f"_{strat}"
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
        head, head_name, _, norm, feat_mean, feat_std, betas, eps = load_head(
            ckpt_path, in_dim, num_classes, device)

        (computed_cpu, all_logits, all_probs, all_preds,
         all_labels, out_filenames) = score_head(
            head, X, y, filenames, ds.missing, num_classes, device,
            norm=norm, feat_mean=feat_mean, feat_std=feat_std)

        print(f"[{strat}] {ckpt_name} (norm={norm}) metrics:")
        for k, v in computed_cpu.items():
            print(f"  {k}: {v:.4f}")

        # Tags so the per-team aggregator can stitch results back together.
        computed_cpu["strategy"] = strat
        computed_cpu["norm"] = norm
        computed_cpu["betas"] = betas  # training-only tag; None on legacy ckpts
        computed_cpu["eps"] = eps      # training-only tag; None on legacy ckpts
        computed_cpu["fold"] = args.fold
        computed_cpu["checkpoint"] = ckpt_name
        computed_cpu["head_name"] = head_name
        computed_cpu["n_real"] = n_real
        computed_cpu["n_missing_failed"] = n_missing

        df_metrics = pd.DataFrame([computed_cpu])
        metrics_csv_path = os.path.join(args.ckpt_dir, f"{args.split}_metrics{suffix}.csv")
        df_metrics.to_csv(metrics_csv_path, index=False)
        print(f"Saved aggregate metrics to {metrics_csv_path}")

        per_sample_data = {
            'filename': out_filenames,
            'label': all_labels.numpy(),
            'prediction': all_preds.numpy(),
            'missing': [0] * n_real + [1] * n_missing,
        }
        for class_idx in range(num_classes):
            per_sample_data[f'logit_class_{class_idx}'] = all_logits[:, class_idx].numpy()
        for class_idx in range(num_classes):
            per_sample_data[f'prob_class_{class_idx}'] = all_probs[:, class_idx].numpy()

        df_per_sample = pd.DataFrame(per_sample_data)
        per_sample_csv_path = os.path.join(
            args.ckpt_dir, f"{args.split}_per_sample_predictions{suffix}.csv")
        df_per_sample.to_csv(per_sample_csv_path, index=False)
        print(f"Saved per-sample predictions to {per_sample_csv_path}")


if __name__ == "__main__":
    main()
