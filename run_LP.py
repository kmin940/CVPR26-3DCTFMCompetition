import argparse, os, random, shutil, torch
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; avoids Tk teardown SIGABRT with DataLoader workers
import matplotlib.pyplot as plt
from metrics.balanced_accuracy import BalancedAccuracy

import torch
import h5py
import wandb
import numpy as np
from torch.utils.data import WeightedRandomSampler, RandomSampler


from torchmetrics import (
    Accuracy,
    F1Score,
    AUROC,
    AveragePrecision,
    Recall,
    Specificity,
    MetricCollection,
)
import pandas as pd

def set_seed(seed: int):
    """Seed every RNG that affects training (weight init + per-epoch sampling)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FeaturesDataset(Dataset):
    def __init__(self, embeds_dir, csv_path, split, target_column=None, split_column='split'):
        # Load CSV and filter by split (split_column lets us pass custom split column name).
        df = pd.read_csv(csv_path)
        # Some split CSVs use 'case_id', autoPET uses 'id' — auto-detect.
        if 'case_id' in df.columns:
            id_col = 'case_id'
        elif 'id' in df.columns:
            id_col = 'id'
        else:
            raise ValueError(f"Neither 'case_id' nor 'id' column found in {csv_path}")
        split_df = df[df[split_column] == split].copy()

        # Build paths and label mapping
        self.paths = []
        self.label_mapping = {}

        for _, row in split_df.iterrows():
            # Extract filename without extension
            case_id = row[id_col]
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

    def __len__(self): return len(self.paths)

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
            y_hat = torch.tensor(hf['y_hat'][:]).float() # torch.Size([2048])

        return y_hat, y

    def _get_num_classes(self):
        # Get unique labels from the CSV mapping
        all_labels = set(self.label_mapping.values())
        return len(all_labels)


def compute_feature_stats(ds, num_workers=8, eps=1e-6):
    """
    Per-dimension mean/std over the training features, for the 'std' (z-score) normalization variant. 
    Computed in a single streaming pass in float64 over the raw (un-sampled) dataset order so it is deterministic. 
    eps clamps the std to avoid div-by-zero on constant feature dimensions. Returns float32 (mean, std).
    """
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=num_workers)
    n = 0
    s = sq = None
    for xb, _ in loader:
        xb = xb.double()
        if s is None:
            s = torch.zeros(xb.size(1), dtype=torch.double)
            sq = torch.zeros(xb.size(1), dtype=torch.double)
        s += xb.sum(0)
        sq += (xb * xb).sum(0)
        n += xb.size(0)
    if n == 0:
        raise RuntimeError("Empty dataset when computing feature stats.")
    mean = s / n
    var = (sq / n - mean * mean).clamp_min(0.0)
    std = var.sqrt().clamp_min(eps)
    return mean.float(), std.float()


class AllClassifiers(nn.Module):
    def __init__(self, in_dim, num_classes, lrs, weight_decays=(0.0,), norms=("raw",),
                 betas_list=((0.9, 0.999),), eps_list=(1e-8,),
                 feat_mean=None, feat_std=None):
        """One independent linear head per (lr, weight_decay, norm, betas, eps) combination.

        `weight_decay`, `betas` and `eps` are set per param-group. AdamW honours
        per-group `betas` and `eps` (SGD has neither — see build_optimizer, which
        strips them); both optimizers honour a per-group `weight_decay`, overriding
        the optimizer-level default. So a single optimizer trains the whole
        lr x wd x norm x betas x eps grid in parallel.

        `betas` is the AdamW (beta1, beta2) pair and `eps` is the AdamW numerical
        epsilon; sweeping either is AdamW-only (a multi-value betas/eps sweep with
        SGD is rejected in main()).

        `norm` selects which view of the (frozen) input each head consumes:
          * 'raw' -> features as-is (the original behavior).
          * 'l2'  -> L2-normalized to the unit sphere (F.normalize, dim=1).
          * 'std' -> z-scored with the train-set per-dim mean/std (feat_mean,
                     feat_std must be provided). This makes the lr/wd grid
                     comparable across foundation models whose embeddings live
                     at very different scales — the point of the norm sweep.
          * 'ln'  -> per-sample LayerNorm across the feature dim, no learnable
                     affine (stateless, like 'l2'/'raw'; needs no feat stats).
        All views are computed once per batch in forward (not per head), so the
        single summed-loss / single-optimizer structure is preserved.

        A suffix is only added to the head name when more than one value of that
        axis is swept ('_wd_<wd>', '_norm_<norm>', '_b1_<b1>_b2_<b2>', '_eps_<eps>'),
        keeping a single-wd / single-norm / single-betas / single-eps 'raw' run
        byte-identical to the previous `clf_lr_*` names. Init order is unchanged
        for the existing axes (outer lr, then wd, then norm, then betas), with eps
        as the innermost loop so single-eps runs draw weights identically to
        before this axis existed.

        Saved checkpoints carry the head's `norm` (and feat_mean/feat_std for
        'std') so inference can apply the matching input transform — see
        save_single_head.
        """
        super().__init__()
        if "std" in norms and (feat_mean is None or feat_std is None):
            raise ValueError("norms includes 'std' but feat_mean/feat_std were not provided.")
        self.clfs = nn.ModuleDict()
        self.param_groups = []
        self.head_view = {}          # head name -> norm variant key
        # head name -> {"lr", "wd", "norm"}; consumed by render_progress to lay
        # the heads out on a (weight_decay rows x lr cols) grid, one figure per norm.
        self.head_meta = {}
        self.norm_set = set(norms)   # which views to materialize in forward
        if feat_mean is not None and feat_std is not None:
            # (1, in_dim) so it broadcasts over the batch dimension.
            self.register_buffer("feat_mean", feat_mean.view(1, -1))
            self.register_buffer("feat_std", feat_std.view(1, -1))
        else:
            self.feat_mean = None
            self.feat_std = None
        include_wd = len(weight_decays) > 1
        include_norm = len(norms) > 1
        include_betas = len(betas_list) > 1
        include_eps = len(eps_list) > 1
        for lr in lrs:
            for wd in weight_decays:
                for norm in norms:
                    for betas in betas_list:
                        for eps in eps_list:
                            name = f"clf_lr_{lr:.0e}".replace("-", "_").replace(".", "_")
                            if include_wd:
                                name += (f"_wd_{wd:.0e}"
                                         .replace("-", "_").replace("+", "_").replace(".", "_"))
                            if include_norm:
                                name += f"_norm_{norm}"
                            if include_betas:
                                # (beta1, beta2) -> '_b1_0_90_b2_0_999'; enough digits to
                                # keep distinct pairs distinct, '.' -> '_' for filename safety.
                                name += (f"_b1_{betas[0]:.3f}_b2_{betas[1]:.4f}"
                                         .replace(".", "_"))
                            if include_eps:
                                # e.g. 1e-8 -> '_eps_1e_08'; '-'/'+'/'.' -> '_' for
                                # filename safety, mirroring the wd suffix scheme.
                                name += (f"_eps_{eps:.0e}"
                                         .replace("-", "_").replace("+", "_").replace(".", "_"))
                            m = nn.Linear(in_dim, num_classes)
                            nn.init.normal_(m.weight, 0.0, 0.01)
                            nn.init.zeros_(m.bias)
                            self.clfs[name] = m
                            self.head_view[name] = norm
                            self.head_meta[name] = {"lr": lr, "wd": wd, "norm": norm,
                                                    "betas": tuple(betas), "eps": eps}
                            # betas/eps are always set on the group
                            self.param_groups.append(
                                {"params": m.parameters(), "lr": lr, "weight_decay": wd,
                                 "betas": tuple(betas), "eps": eps})

    def _make_views(self, x):
        # Only materialize the views that some head actually consumes.
        views = {"raw": x}
        if "l2" in self.norm_set:
            views["l2"] = F.normalize(x, dim=1)
        if "std" in self.norm_set:
            views["std"] = (x - self.feat_mean) / self.feat_std
        if "ln" in self.norm_set:
            views["ln"] = F.layer_norm(x, (x.size(1),))
        return views

    def forward(self, x):
        views = self._make_views(x)
        return {k: m(views[self.head_view[k]]) for k, m in self.clfs.items()}

def acc_top1(logits, y):  # "top-1"
    return (logits.argmax(1) == y).float().mean().item()

def build_optimizer(name, param_groups, weight_decay=0.0):
    """Build the optimizer (adapted from dinov3 OptimizerType). `weight_decay` is
    applied to both options (set to 0 to disable, e.g. for a frozen backbone).

    Per-group `betas`/`eps` (set by AllClassifiers for the betas/eps sweeps) are
    honoured by AdamW; SGD has neither, so we strip the keys from the groups before
    handing them to SGD. A multi-value betas/eps sweep with SGD is rejected upstream
    in main().
    """
    if name == "sgd":
        for g in param_groups:
            g.pop("betas", None)
            g.pop("eps", None)
        return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(name, optimizer, param_groups, epoch_length, epochs, warmup_iters):
    """Build a per-iteration LR scheduler (adapted from dinov3 SchedulerType).

    Stepped once per optimizer step (not per epoch), so the LR is parameterized in
    iterations: max_iter = epochs * epoch_length.

    * 'none'             -> constant LR (current/default behavior).
    * 'cosine_annealing' -> CosineAnnealingLR(eta_min=0), optionally prefixed by a
                            linear warmup over `warmup_iters` iterations (the source
                            has no warmup; we add it via SequentialLR when requested).
    * 'one_cycle'        -> OneCycleLR with per-head max_lr; warmup is built in
                            (pct_start), so `warmup_iters` is ignored here.
    """
    if name == "none":
        return None
    max_iter = epochs * epoch_length
    if name == "one_cycle":
        if warmup_iters > 0:
            print("  Note: --warmup_epochs is ignored for one_cycle "
                  "(OneCycleLR has built-in warmup via pct_start).")
        lr_list = [g["lr"] for g in param_groups]
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr_list, steps_per_epoch=epoch_length, epochs=epochs)
    if name == "cosine_annealing":
        if warmup_iters > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_iters)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, max_iter - warmup_iters), eta_min=0)
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters])
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0)
    raise ValueError(f"Unknown scheduler: {name}")


def train_one_epoch(model, loader, opt, crit, device, scheduler=None):
    model.train()
    tot_loss, n_batches = 0.0, 0
    tot_samples = 0
    per_clf_losses = {name: 0.0 for name in model.clfs.keys()}
    for xb, yb in loader: # xb: (B, D), yb: (B, 1)
        batch_size = xb.size(0)
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        out = model(xb)
        loss_heads = [crit(o, yb) for o in out.values()]
        per_clf_losses = {name: per_clf_losses[name] + l.item() * batch_size for name, l in zip(model.clfs.keys(), loss_heads)}
        loss = torch.stack(loss_heads).sum()  # sum over heads
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # Per-iteration LR schedule step (matches dinov3 linear.py).
        if scheduler is not None:
            scheduler.step()
        tot_loss += loss.item() * batch_size
        tot_samples += batch_size
        n_batches += 1
    for name in per_clf_losses.keys():
        per_clf_losses[name] /= tot_samples
    return tot_loss / tot_samples, per_clf_losses


@torch.no_grad()
def evaluate(model, loader, crit, device, num_classes, monitor_metric, monitor_metric_mode="max"):
    base_metrics = MetricCollection({
        "acc": Accuracy(task="multiclass", num_classes=num_classes),
        "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        "auroc": AUROC(task="multiclass", num_classes=num_classes),
        "ap": AveragePrecision(task="multiclass", num_classes=num_classes),
        "sensitivity": Recall(task="multiclass", num_classes=num_classes, average="macro"),
        "specificity": Specificity(task="multiclass", num_classes=num_classes, average="macro"),
        "balanced_acc": BalancedAccuracy(num_classes=num_classes, task="multiclass"),
    })

    model.eval()
    names = list(model.clfs.keys())

    metrics = {name: base_metrics.clone().to(device) for name in names}
    losses = {name: 0.0 for name in names}
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        outputs = model(xb)  # {clf_name: logits}
        batch_size = xb.size(0)
        total_samples += batch_size

        for name, logits in outputs.items():
            losses[name] += crit(logits, yb).item() * batch_size
            metrics[name].update(logits, yb)

    for name in names:
        losses[name] /= total_samples

    computed = {name: metrics[name].compute() for name in names}
    for name in names:
        for k, v in computed[name].items():
            computed[name][k] = v.item()
        computed[name]["loss"] = losses[name]
        # Composite selection metric: average of balanced accuracy and AUROC.
        # Exposed here so it can be used as --monitor_metric (early stopping,
        # logging, curves) in addition to the val_balacc_auroc checkpoint strategy.
        computed[name]["val_balacc_auroc"] = (
            0.5 * computed[name]["balanced_acc"] + 0.5 * computed[name]["auroc"]
        )
        computed[name]["val_balacc_loss"] = (
            0.5 * computed[name]["balanced_acc"] - 0.5 * computed[name]["loss"]
        )

    select = min if monitor_metric_mode == "min" else max
    best_name = select(names, key=lambda n: computed[n][monitor_metric])
    best_acc = computed[best_name][monitor_metric]

    return best_name, best_acc, computed



def save_single_head(model, name, path, results):
    sd = {f"clfs.{name}."+k: v for k, v in model.clfs[name].state_dict().items()}
    payload = {"state_dict": sd, "head_name": name, "val_metrics": results,
               "norm": model.head_view[name],
               # betas/eps have no effect at inference (training-only AdamW params);
               # saved purely for traceability of which variant produced this head.
               "betas": model.head_meta[name]["betas"],
               "eps": model.head_meta[name]["eps"]}
    # 'std' heads need the train-set stats to reproduce the input transform at
    # inference; 'l2'/'raw' are parameter-free so the norm tag alone suffices.
    if model.head_view[name] == "std" and model.feat_mean is not None:
        payload["feat_mean"] = model.feat_mean.detach().cpu()
        payload["feat_std"] = model.feat_std.detach().cpu()
    torch.save(payload, path)


def render_progress(out_dir, head_meta, lrs, weight_decays, norms_order, betas_order,
                    eps_order, ep,
                    train_losses_over_epochs, val_losses_over_epochs,
                    monitor_metric_over_epochs, monitor_metric, monitor_metric_mode,
                    val_loss_ema_over_epochs=None, converged_epoch_per_head=None,
                    star_ckpt=None):
    """Write one progress figure per (norm, betas, eps), laid out as a (weight_decay
    rows x lr cols) grid of per-head curves.

    Each cell shows Train Loss / Val Loss (and Val <monitor_metric> when the
    monitor metric is not itself the loss) for the head at that (lr, wd, norm),
    with min/max annotations. A gold star marks the actually-saved checkpoint
    passed in ``star_ckpt`` -- placed on the head/epoch it was saved from,
    labelled with the saving strategy, and its subplot outlined in red. When
    ``star_ckpt`` is None it falls back to the raw best of the monitor metric
    within the norm group.

    Filenames: a single (norm, betas, eps) group writes ``progress.png`` (unchanged
    from the old single-figure behaviour); a swept norm adds a ``_<norm>`` tag, a
    swept betas adds a ``_b1_<b1>_b2_<b2>`` tag, and a swept eps adds a ``_eps_<eps>``
    tag, e.g. ``progress_std_b1_0_95_b2_0_999.png``.
    """
    # Reverse map (lr, wd, norm, betas, eps) -> head name. lrs/weight_decays/betas/eps
    # carry the same objects stored in head_meta, so equality holds for the lookup.
    meta_to_name = {(m["lr"], m["wd"], m["norm"], m["betas"], m["eps"]): name
                    for name, m in head_meta.items()}
    nrows, ncols = len(weight_decays), len(lrs)
    multi_norm = len(norms_order) > 1
    multi_betas = len(betas_order) > 1
    multi_eps = len(eps_order) > 1
    for norm in norms_order:
      for betas in betas_order:
        betas = tuple(betas)
        for eps in eps_order:
          group = [(wd, lr, meta_to_name[(lr, wd, norm, betas, eps)])
                   for wd in weight_decays for lr in lrs
                   if (lr, wd, norm, betas, eps) in meta_to_name]

          # Star: the actually-saved checkpoint (star_ckpt) when its head lives in
          # this norm group -- placed at that head's saved epoch and labelled with
          # the saving strategy. Before any checkpoint exists (star_ckpt is None)
          # we fall back to the raw best of the monitor metric within the group.
          group_names = {name for _, _, name in group}
          star_head, star_idx, star_val, star_label = None, None, None, None
          if star_ckpt is not None:
              sh, sep = star_ckpt.get("head"), star_ckpt.get("ep")
              if sh in group_names and sep is not None:
                  arr = np.asarray(monitor_metric_over_epochs.get(sh, []),
                                   dtype=float)
                  si = int(sep) - 1
                  if 0 <= si < arr.size and not np.isnan(arr[si]):
                      star_head, star_idx, star_val = sh, si, float(arr[si])
                      star_label = star_ckpt.get("label", "best")
          else:
              for _, _, name in group:
                  arr = np.asarray(monitor_metric_over_epochs[name], dtype=float)
                  if arr.size == 0 or np.all(np.isnan(arr)):
                      continue
                  bidx = int(np.nanargmax(arr) if monitor_metric_mode == 'max'
                             else np.nanargmin(arr))
                  bval = float(arr[bidx])
                  if star_head is None or (
                          (monitor_metric_mode == 'max' and bval > star_val) or
                          (monitor_metric_mode == 'min' and bval < star_val)):
                      star_head, star_idx, star_val = name, bidx, bval
              if star_head is not None:
                  star_label = monitor_metric

          fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                                  squeeze=False)
          for r, wd in enumerate(weight_decays):
              for c, lr in enumerate(lrs):
                  ax = axs[r][c]
                  name = meta_to_name.get((lr, wd, norm, betas, eps))
                  if name is None:
                      ax.set_visible(False)
                      continue
                  curves = [
                      ("Train Loss", train_losses_over_epochs[name]),
                      ("Val Loss", val_losses_over_epochs[name]),
                  ]
                  # Skip the monitor curve when it *is* the val loss
                  # (monitor_metric="loss"): plotting it again duplicates the
                  # "Val Loss" line and adds a second val-loss legend entry.
                  if monitor_metric != "loss":
                      curves.append(
                          (f"Val {monitor_metric}", monitor_metric_over_epochs[name]))
                  for label, ys in curves:
                      # x is per-curve, not a shared range(1, ep+1): a head that has
                      # early-stopped has fewer points (its curves are frozen at its
                      # convergence epoch) than heads still training.
                      line, = ax.plot(range(1, len(ys) + 1), ys, label=label)
                      # Overlay the EMA-smoothed val loss -- the signal that actually
                      # drives the patience counter -- as a dashed line in the same
                      # color as the raw Val Loss it smooths.
                      if label == "Val Loss" and val_loss_ema_over_epochs is not None:
                          ema_ys = val_loss_ema_over_epochs.get(name, [])
                          if len(ema_ys):
                              ax.plot(range(1, len(ema_ys) + 1), ema_ys,
                                      label="Val Loss (EMA)", linestyle="--",
                                      color=line.get_color(), alpha=0.8)
                      if len(ys) == 0:
                          continue
                      ys_arr = np.asarray(ys, dtype=float)
                      imax = int(np.nanargmax(ys_arr))
                      imin = int(np.nanargmin(ys_arr))
                      col = line.get_color()
                      ax.annotate(f"max {ys_arr[imax]:.3f}", (imax + 1, ys_arr[imax]),
                                  textcoords="offset points", xytext=(0, 6),
                                  fontsize=8, color=col, ha="center")
                      ax.annotate(f"min {ys_arr[imin]:.3f}", (imin + 1, ys_arr[imin]),
                                  textcoords="offset points", xytext=(0, -12),
                                  fontsize=8, color=col, ha="center")
                  if name == star_head and star_idx is not None:
                      ax.plot(star_idx + 1, star_val, marker="*", markersize=18,
                              color="gold", markeredgecolor="black",
                              markeredgewidth=1.0, linestyle="None", zorder=10,
                              label=f"{star_label} ckpt (ep{star_idx + 1}, "
                                    f"{monitor_metric}={star_val:.3f})")
                      ax.annotate(f"{star_label} ckpt ep{star_idx + 1}",
                                  (star_idx + 1, star_val),
                                  textcoords="offset points", xytext=(0, 14),
                                  fontsize=9, fontweight="bold", color="black",
                                  ha="center")
                      # Outline the starred (saved-checkpoint) subplot in red so
                      # it stands out from the rest of the lr x wd grid.
                      for spine in ax.spines.values():
                          spine.set_edgecolor("red")
                          spine.set_linewidth(2.5)
                  ax.set_xlabel("Epoch")
                  title = f"lr={lr:.0e}  wd={wd:.0e}"
                  if name == star_head and star_idx is not None:
                      title = f"★ {star_label}  " + title
                  conv_ep = (converged_epoch_per_head or {}).get(name)
                  if conv_ep is not None:
                      title += f"  [converged ep{conv_ep}]"
                  ax.set_title(title)
                  ax.set_ylabel("Value")
                  ax.legend(fontsize=7)
                  ax.relim()
                  ax.autoscale_view()
                  ax.margins(y=0.1)
          title_bits = []
          if multi_norm:
              title_bits.append(f"norm={norm}")
          if multi_betas:
              title_bits.append(f"betas={betas}")
          if multi_eps:
              title_bits.append(f"eps={eps}")
          if title_bits:
              fig.suptitle("  ".join(title_bits))
          fig.tight_layout()
          # Preserve the old filenames: single norm+betas+eps -> progress.png; a
          # swept norm keeps the ``_<norm>`` tag; a swept betas appends
          # ``_b1_.._b2_..`` and a swept eps appends ``_eps_..``.
          suffix = ""
          if multi_norm:
              suffix += f"_{norm}"
          if multi_betas:
              suffix += (f"_b1_{betas[0]:.3f}_b2_{betas[1]:.4f}".replace(".", "_"))
          if multi_eps:
              suffix += (f"_eps_{eps:.0e}"
                         .replace("-", "_").replace("+", "_").replace(".", "_"))
          fname = f"progress{suffix}.png"
          fig.savefig(os.path.join(out_dir, fname))
          plt.close(fig)


def extract_labels_from_dataset(ds: FeaturesDataset):
    labels = []
    for p in ds.paths:
        fn = os.path.basename(p).replace(".h5", "")
        if fn not in ds.label_mapping:
            raise ValueError(f"{fn} missing from label mapping")
        labels.append(int(ds.label_mapping[fn]))
    return np.asarray(labels, dtype=int)


def compute_sample_weights(ds: FeaturesDataset):
    labels = extract_labels_from_dataset(ds)
    if labels.size == 0:
        raise RuntimeError("Empty label array when computing sample weights.")

    n_samples = len(labels)
    num_classes = int(labels.max()) + 1
    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts[class_counts == 0] = 1  # avoid div by zero for missing classes

    # Matches sklearn's "balanced" formula exactly
    class_weights = n_samples / (num_classes * class_counts)
    sample_weights = class_weights[labels]

    return torch.as_tensor(sample_weights, dtype=torch.float)

def resolve_paths(args):
    """Resolve embeddings dir, labels CSV, and split column from the CLI, supporting
    both layouts:
      * AMOS single-split: --labels_root + --target -> <labels_root>/<target>.csv,
        --embeds_root + --target -> <embeds_root>/<target>/embeddings, split column 'split'.
      * 5-fold: --csv_path + --fold -> split column 'fold{fold}'; --embeds_dir directly.
    Explicit --embeds_dir / --csv_path / --split_column always win over the
    *_root convenience options.
    """
    # Embeddings dir
    if args.embeds_dir:
        embeds_dir = args.embeds_dir
    elif args.embeds_root and args.target:
        embeds_dir = os.path.join(args.embeds_root, args.target, "embeddings")
    else:
        raise ValueError("Provide --embeds_dir, or --embeds_root together with --target.")

    # Labels CSV
    if args.csv_path:
        labels_csv = args.csv_path
    elif args.labels_root and args.target:
        labels_csv = os.path.join(args.labels_root, args.target + ".csv")
    else:
        raise ValueError("Provide --csv_path, or --labels_root together with --target.")

    # Split column: --fold selects fold{N} (5-fold mode); otherwise --split_column.
    if args.fold is not None:
        split_column = f"fold{args.fold}"
    else:
        split_column = args.split_column

    return embeds_dir, labels_csv, split_column


def main():
    ap = argparse.ArgumentParser(
        description="Unified linear-probing trainer. Supports the AMOS single-split "
                    "layout and the 3-dataset 5-fold layout through the same training "
                    "pipeline.")
    # --- Embeddings location (one of) ---
    ap.add_argument("--embeds_dir", type=str, default=None,
                    help="Directory of .h5 feature files for a single team. Takes "
                         "precedence over --embeds_root.")
    ap.add_argument("--embeds_root", type=str, default=None,
                    help="Root dir; embeddings resolved as <embeds_root>/<target>/embeddings.")
    # --- Labels location (one of) ---
    ap.add_argument("--csv_path", type=str, default=None,
                    help="Path to split CSV. Takes precedence over --labels_root.")
    ap.add_argument("--labels_root", type=str, default=None,
                    help="Root dir; CSV resolved as <labels_root>/<target>.csv.")
    # --- Split / target selection ---
    ap.add_argument("--target", type=str, required=True,
                    help='Target column name in the CSV (also used to build paths '
                         'from the *_root options).')
    ap.add_argument("--fold", type=int, default=None,
                    help="5-fold mode: select column 'fold{fold}'. Omit for "
                         "single-split mode (uses --split_column).")
    ap.add_argument("--split_column", type=str, default="split",
                    help="Column used to select train/val rows when --fold is not "
                         "given (single-split mode). Default: 'split'.")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory for this run.")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs",     type=int, default=150)
    ap.add_argument(
                        "--lrs",
                        type=float,
                        nargs="+",
                        default=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1], #[5e-4, 5e-3, 5e-2], #[5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2], #[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
                        help="List of learning rates"
                    )
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--optimizer", type=str, default="adamw",
                    choices=["adamw", "sgd"],
                    help="Optimizer for the linear heads. 'sgd' uses momentum=0.9; "
                         "weight decay is controlled by --weight_decay.")
    ap.add_argument("--weight_decays", type=float, nargs="+",
                    default=[1e-5, 1e-4, 1e-3, 1e-2],
                    help="Sweep multiple weight decays in parallel")
    ap.add_argument("--norms", type=str, nargs="+", default=["raw", "std"],
                    choices=["raw", "l2", "std", "ln"],
                    help="Sweep input-normalization variants in parallel, one independent head per (lr, weight_decay, norm) combination. ")
    ap.add_argument("--betas", type=str, nargs="+",
                    default=["0.9,0.999"],
                    help="Sweep AdamW (beta1,beta2) pairs in parallel")
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-8], 
                    help="Sweep AdamW numerical-epsilon values in parallel")
    ap.add_argument("--scheduler", type=str, default="cosine_annealing",
                    choices=["none", "cosine_annealing", "one_cycle"],
                    help="Per-iteration LR schedule. 'none' keeps a constant LR (default/current behavior). 'cosine_annealing' decays to 0 over training; 'one_cycle' uses each head's LR as max_lr.")
    ap.add_argument("--warmup_epochs", type=float, default=5.0,
                    help="Linear LR warmup length, in epochs, prepended to cosine_annealing. Ignored for one_cycle (built-in warmup) and when --scheduler none.")
    ap.add_argument("--monitor_metric", type=str, default="loss",
                    help="selection criterion.")
    ap.add_argument("--monitor_metric_mode", type=str, default='min')
    ap.add_argument("--top_k_checkpoints", type=int, default=1)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="linear_probe_cvpr26")
    ap.add_argument("--wandb_run_name", type=str, default=None,
                    help="Optional W&B run name; defaults to basename(out_dir)")
    ap.add_argument("--patience", type=int, default=10,
                help="Early stopping patience based on best head")
    ap.add_argument("--es_ema_decay", type=float, default=0.9,
                help="EMA decay for the early-stopping monitor metric. The patience counter is driven by an exponential moving average ema = decay*ema + (1-decay)*current, so noisy single-epoch dips/spikes don't reset or prematurely trip patience. 0 disables it.")
    ap.add_argument("--min_save_epoch", type=int, default=6,
                help="Only save/select checkpoints from this epoch onward (inclusive)")
    ap.add_argument("--seed", type=int, default=42,
                help="Random seed for weight init and per-epoch sampling")
    ap.add_argument("--gap_penalty_weight", type=float, default=0.5,
                help="Weight on |train_loss - val_loss| subtracted from the base score in the 'val_balacc_auroc_gap' checkpoint strategy. Penalizes overfitting (large train/val loss gap).")
    args = ap.parse_args()

    set_seed(args.seed)

    embeds_dir, labels_csv, split_column = resolve_paths(args)
    out_dir = args.out_dir
    target_col = args.target

    if os.path.isdir(out_dir):
        print(f"  Clearing existing out_dir: {out_dir}")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    use_wandb = args.use_wandb
    if use_wandb:
        wandb_dir = os.path.join(out_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
        wandb_run_name = args.wandb_run_name or os.path.basename(os.path.normpath(out_dir))
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=vars(args),
        )

    mode_desc = (f"fold={args.fold} (column: {split_column})"
                 if args.fold is not None
                 else f"single-split (column: {split_column})")
    print(f"=== {mode_desc} ===")
    print(f"  embeds_dir: {embeds_dir}")
    print(f"  labels_csv: {labels_csv}")
    print(f"  out_dir:    {out_dir}")

    ds_tr = FeaturesDataset(embeds_dir, labels_csv, split='train',
                            target_column=target_col, split_column=split_column)
    if len(ds_tr) == 0:
        print(f"  WARNING: no training samples for {mode_desc}; exiting")
        if use_wandb:
            wandb.finish()
        return

    embed_dim = ds_tr[0][0].shape[0]
    num_classes = ds_tr._get_num_classes()
    print(f'  Feature dimension: {embed_dim}, num_classes: {num_classes}')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dedicated generator so per-epoch sampling is reproducible and decoupled from the order of weight-init draws on the global RNG.
    sampler_gen = torch.Generator().manual_seed(args.seed)

    # Raw inverse-frequency weights (1 / class_count per sample). Used by WeightedRandomSampler
    sample_weights = compute_sample_weights(ds_tr)

    print(
        f"  Sample weight statistics (inverse-frequency): "
        f"Max: {sample_weights.max().item():.6f}, "
        f"Min: {sample_weights.min().item():.6f}, "
        f"Mean: {sample_weights.mean().item():.6f}, "
        f"Max/Min ratio: {(sample_weights.max() / sample_weights.min()).item():.4f}"
    )

    if torch.isclose(sample_weights.max(), sample_weights.min()):
        print("  Warning: All samples have the same weight. Using uniform sampling instead.")
        sampler = RandomSampler(
            ds_tr,
            replacement=True,
            num_samples=len(ds_tr),
            generator=sampler_gen,
        )
    else:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(ds_tr),
            replacement=True,
            generator=sampler_gen,
        )

    # Clamp batch size to dataset size and disable drop_last when the train set is smaller than the requested batch (5-fold splits can be tiny, e.g. 87).
    effective_bs = min(args.batch_size, len(ds_tr))
    drop_last_tr = len(ds_tr) >= args.batch_size
    if effective_bs != args.batch_size or not drop_last_tr:
        print(f"  Adjusting train DataLoader: batch_size={effective_bs} (requested {args.batch_size}), drop_last={drop_last_tr}, len(ds_tr)={len(ds_tr)}")
    dl_tr = DataLoader(
        ds_tr,
        batch_size=effective_bs,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last_tr,
    )

    ds_va = FeaturesDataset(embeds_dir, labels_csv, split='val',
                            target_column=target_col, split_column=split_column)
    if len(ds_va) == 0:
        print(f"  WARNING: no val samples for {mode_desc} (all val embeddings missing for this team); exiting")
        if use_wandb:
            wandb.finish()
        return
    dl_va = DataLoader(ds_va, batch_size=args.batch_size,
                       shuffle=False, num_workers=args.num_workers, pin_memory=True)

    lrs = args.lrs
    weight_decays = args.weight_decays
    norms = args.norms
    # Parse --betas tokens ("b1,b2") into (float, float) tuples for the betas sweep.
    betas_list = []
    for tok in args.betas:
        parts = tok.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"--betas expects comma-separated 'b1,b2' pairs; got '{tok}'")
        betas_list.append((float(parts[0]), float(parts[1])))
    # betas is an AdamW-only parameter; a multi-value sweep is meaningless for SGD so throws error.
    if args.optimizer == "sgd" and len(betas_list) > 1:
        raise ValueError(
            "--betas sweep (more than one pair) requires --optimizer adamw; "
            "SGD has no betas.")
    eps_list = list(args.eps)
    # eps is AdamW-only; a multi-value sweep with SGD would be meaningless so throws error.
    if args.optimizer == "sgd" and len(eps_list) > 1:
        raise ValueError(
            "--eps sweep (more than one value) requires --optimizer adamw; "
            "SGD has no eps.")
    # The 'std' variant needs train-set per-dim mean/std (computed over the raw, un-sampled train order, not the weighted sampler, for determinism).
    feat_mean = feat_std = None
    if "std" in norms:
        feat_mean, feat_std = compute_feature_stats(ds_tr, num_workers=args.num_workers)
        print(f"  Standardization stats (train): "
              f"mean[min/mean/max]={feat_mean.min():.4f}/{feat_mean.mean():.4f}/{feat_mean.max():.4f}, "
              f"std[min/mean/max]={feat_std.min():.4f}/{feat_std.mean():.4f}/{feat_std.max():.4f}")
    # One head per (lr, weight_decay, norm, betas, eps) combination.
    model = AllClassifiers(embed_dim, num_classes, lrs, weight_decays, norms,
                           betas_list=betas_list, eps_list=eps_list,
                           feat_mean=feat_mean, feat_std=feat_std).to(device)
    opt = build_optimizer(args.optimizer, model.param_groups, weight_decay=0.0)
    print(f"  lrs={lrs}")
    print(f"  weight_decays={weight_decays}")
    print(f"  betas={betas_list}")
    print(f"  eps={eps_list}")
    print(f"  norms={norms} "
          f"({len(lrs)}x{len(weight_decays)}x{len(norms)}x{len(betas_list)}x{len(eps_list)}="
          f"{len(lrs) * len(weight_decays) * len(norms) * len(betas_list) * len(eps_list)} heads)")
    # Scheduler is stepped per-iteration, so it's parameterized in iterations: epoch_length = number of optimizer steps per epoch (= len(dl_tr)).
    epoch_length = len(dl_tr)
    warmup_iters = int(round(args.warmup_epochs * epoch_length))
    scheduler = build_scheduler(
        args.scheduler, opt, model.param_groups,
        epoch_length=epoch_length, epochs=args.epochs, warmup_iters=warmup_iters,
    )
    print(f"  optimizer={args.optimizer}, scheduler={args.scheduler}, "
          f"epoch_length={epoch_length}, warmup_iters={warmup_iters}")
    crit = nn.CrossEntropyLoss()

    monitor_metric = args.monitor_metric
    monitor_metric_mode = args.monitor_metric_mode

    # End-of-run "raw best" tracker: best single (head, epoch) by --monitor_metric,
    # un-smoothed, for the final report only. Early stopping is per-head (see the
    # per-head state set up just before the epoch loop).
    init_met_val = -1.0 if monitor_metric_mode == "max" else np.inf
    best_met_overall = init_met_val
    best_head_overall = None

    # --- Checkpointing strategies ---
    # Each strategy independently selects the best (head, epoch) by its own criterion and saves its own checkpoint. Early stopping below still uses --monitor_metric.
    checkpoint_strategies = [
        {"name": "val_balanced_acc", "mode": "max",
         "score": lambda m: m["balanced_acc"]},
        {"name": "val_auroc", "mode": "max",
         "score": lambda m: m["auroc"]},
        {"name": "val_loss", "mode": "min",
         "score": lambda m: m["loss"]},
        {"name": "val_balacc_auroc", "mode": "max",
         "score": lambda m: 0.5 * m["balanced_acc"] + 0.5 * m["auroc"]},
        {"name": "val_balacc_loss", "mode": "max",
         "score": lambda m: 0.5 * m["balanced_acc"] - 0.5 * m["loss"]},
    ]
    # Per-strategy running state.
    ckpt_state = {
        s["name"]: {
            "best_score": -np.inf if s["mode"] == "max" else np.inf,
            "best_head": None,
            "best_ep": None,
            "best_all_met": None,
            "ckpt_name": None,
            "saved": [],  # (score, ep, filename) kept for top-k pruning
        }
        for s in checkpoint_strategies
    }

    head_names = list(model.clfs.keys())
    num_heads = len(head_names)

    train_losses_over_epochs = {name: [] for name in head_names}
    val_losses_over_epochs = {name: [] for name in head_names}
    monitor_metric_over_epochs = {name: [] for name in head_names}
    # EMA-smoothed val loss per head, recorded each epoch for the progress plot so the figure shows the raw val loss and the smoothed signal together.
    val_loss_ema_over_epochs = {name: [] for name in head_names}
    val_loss_ema_per_head = {name: None for name in head_names}
    # Epoch at which each head early-stopped (None until it converges); used to
    # annotate and freeze that head's curves in the progress plot.
    converged_epoch_per_head = {name: None for name in head_names}

    # --- Per-head early-stopping state ---
    # Each head is an independent probe on frozen features so every head gets its own EMA-smoothed monitor metric and its own patience counter. 
    # A head that plateaus is added to `converged_heads`: it is dropped from checkpoint selection and stops counting. The whole run stops once every head has converged.
    patience = args.patience
    ema_met_per_head = {name: None for name in head_names}
    best_ema_per_head = {name: init_met_val for name in head_names}
    epochs_no_improve_per_head = {name: 0 for name in head_names}
    converged_heads = set()
    for ep in range(1, args.epochs + 1):
        tr_loss, per_clf_losses = train_one_epoch(model, dl_tr, opt, crit, device, scheduler)
        best_name, best_met, all_met = evaluate(model, dl_va, crit, device, num_classes, monitor_metric, monitor_metric_mode)
        for name in head_names:
            all_met[name]["train_loss"] = per_clf_losses[name]

        # Heads already frozen at the start of this epoch: their progress curves stop here (a newly-converged head still records this epoch's final point).
        converged_at_epoch_start = set(converged_heads)

        # Per-head early-stopping monitor (EMA of --monitor_metric) Update each still-active head's smoothed metric and patience counter.
        # Counting only begins at --min_save_epoch (early epochs are noisy), so a head can never converge before its best checkpoint had a chance to save.
        for name in head_names:
            if name in converged_heads:
                continue
            cur = all_met[name][monitor_metric]
            if ema_met_per_head[name] is None:
                ema_met_per_head[name] = cur
            else:
                ema_met_per_head[name] = (
                    args.es_ema_decay * ema_met_per_head[name]
                    + (1.0 - args.es_ema_decay) * cur
                )
            ema = ema_met_per_head[name]
            improved = (
                (monitor_metric_mode == 'max' and ema > best_ema_per_head[name]) or
                (monitor_metric_mode == 'min' and ema < best_ema_per_head[name])
            )
            if improved:
                best_ema_per_head[name] = ema
                epochs_no_improve_per_head[name] = 0
            elif ep >= args.min_save_epoch:
                epochs_no_improve_per_head[name] += 1

            if ep >= args.min_save_epoch and epochs_no_improve_per_head[name] >= patience:
                converged_heads.add(name)
                converged_epoch_per_head[name] = ep
                print(f"  head '{name}' converged at epoch {ep} "
                      f"(no EMA improvement in {patience} epochs)")

        best_bal_acc = all_met[best_name]['balanced_acc']
        best_auroc = all_met[best_name]['auroc']
        if use_wandb:
            log_dict = {
                "epoch": ep,
                "train/loss": tr_loss,
                "train/lr": opt.param_groups[0]["lr"],
                "val/best_head": best_name,
                f"val/{monitor_metric}_best": best_met,
                "val/balanced_acc_best": best_bal_acc,
                "val/auroc_best": best_auroc,
                "es/num_converged": len(converged_heads),
            }

            for name in head_names:
                log_dict[f"train/{name}/loss"] = per_clf_losses[name]
                log_dict[f"val/{name}/loss"] = all_met[name]["loss"]
                log_dict[f"val/{name}/{monitor_metric}"] = all_met[name][monitor_metric]
                log_dict[f"val/{name}/{monitor_metric}_ema"] = ema_met_per_head[name]
                log_dict[f"es/{name}/no_improve"] = epochs_no_improve_per_head[name]
                log_dict[f"es/{name}/converged"] = int(name in converged_heads)

            wandb.log(log_dict, step=ep)

        for name in head_names:
            # Freeze the curves of heads that early-stopped in a previous epoch.
            if name in converged_at_epoch_start:
                continue
            train_losses_over_epochs[name].append(per_clf_losses[name])
            vloss = all_met[name]["loss"]
            val_losses_over_epochs[name].append(vloss)
            monitor_metric_over_epochs[name].append(all_met[name][monitor_metric])
            prev = val_loss_ema_per_head[name]
            ema = (vloss if prev is None
                   else args.es_ema_decay * prev + (1.0 - args.es_ema_decay) * vloss)
            val_loss_ema_per_head[name] = ema
            val_loss_ema_over_epochs[name].append(ema)

        # Per-strategy checkpoint selection (only from --min_save_epoch onward) ---
        active_heads = [n for n in head_names if n not in converged_heads]
        if ep >= args.min_save_epoch and active_heads:
            for s in checkpoint_strategies:
                sname, smode, sscore = s["name"], s["mode"], s["score"]
                pick = max if smode == "max" else min
                # Best still-active (non-converged) head for this strategy.
                head = pick(active_heads, key=lambda n: sscore(all_met[n]))
                score = sscore(all_met[head])
                st = ckpt_state[sname]
                improved = (
                    (smode == "max" and score > st["best_score"]) or
                    (smode == "min" and score < st["best_score"])
                )
                if not improved:
                    continue
                st["best_score"] = score
                st["best_head"] = head
                st["best_ep"] = ep
                st["best_all_met"] = dict(all_met[head])

                ckpt_name = f"best_{sname}_{score:.4f}_{head}_ep{ep}.pth"
                save_single_head(model, head, os.path.join(out_dir, ckpt_name), all_met[head])
                st["ckpt_name"] = ckpt_name
                st["saved"].append((score, ep, ckpt_name))

                # Prune to top_k_checkpoints: best scores first, newer epoch wins ties.
                if len(st["saved"]) > args.top_k_checkpoints:
                    sign = -1.0 if smode == "max" else 1.0
                    st["saved"].sort(key=lambda t: (sign * t[0], -t[1]))
                    keep, drop = (st["saved"][:args.top_k_checkpoints],
                                  st["saved"][args.top_k_checkpoints:])
                    st["saved"] = keep
                    for _, _, fn in drop:
                        if fn == ckpt_name:
                            continue  # never delete the one we just saved
                        fp = os.path.join(out_dir, fn)
                        if os.path.exists(fp):
                            os.remove(fp)

        # Star the saved val_loss checkpoint (the head/epoch it was selected from) on the progress figure
        _vl = ckpt_state.get("val_loss")
        star_ckpt = ({"head": _vl["best_head"], "ep": _vl["best_ep"],
                      "label": "val_loss"}
                     if _vl and _vl["best_head"] is not None else None)
        all_converged = len(converged_heads) == num_heads
        if ep % args.log_interval == 0 or ep == args.epochs or all_converged:
            # One figure per norm; each is a (weight_decay rows x lr cols) grid. Multiple norms -> separate progress_<norm>.png files.
            render_progress(
                out_dir, model.head_meta, lrs, weight_decays, norms, betas_list,
                eps_list, ep,
                train_losses_over_epochs, val_losses_over_epochs,
                monitor_metric_over_epochs, monitor_metric, monitor_metric_mode,
                val_loss_ema_over_epochs, converged_epoch_per_head,
                star_ckpt=star_ckpt)

        # Track the raw best (head + value) across all heads/epochs for the end-of-run report
        if ((monitor_metric_mode == 'max' and best_met > best_met_overall) or
                (monitor_metric_mode == 'min' and best_met < best_met_overall)):
            best_met_overall = best_met
            best_head_overall = best_name

        print(f"  epoch {ep} | train_loss {tr_loss:.4f} | "
              f"val_best {best_name}={best_met:.4f} | "
              f"converged {len(converged_heads)}/{num_heads}")

        # Run-level early stop: every independent head has plateaued.
        if len(converged_heads) == num_heads:
            print(f"  Early stopping at epoch {ep} "
                  f"(all {num_heads} heads converged)")
            break

    print(f"  Overall best classifier for {mode_desc}:", {
        "name": best_head_overall,
        'val_' + monitor_metric: best_met_overall,
    })
    if use_wandb:
        wandb.finish()


    # One row per checkpoint strategy, reporting the val metrics of its selected head.
    report_rows = {}
    for s in checkpoint_strategies:
        st = ckpt_state[s["name"]]
        if st["best_all_met"] is None:
            continue
        row = dict(st["best_all_met"])
        row["head"] = st["best_head"]
        row["norm"] = model.head_view[st["best_head"]]
        row["betas"] = model.head_meta[st["best_head"]]["betas"]
        row["selected_epoch"] = st["best_ep"]
        row["checkpoint"] = st["ckpt_name"]
        report_rows[s["name"]] = row

    if not report_rows:
        print(f"  WARNING: no checkpoint was selected for {mode_desc} "
              f"(min_save_epoch={args.min_save_epoch}); skipping val_report.csv")
    else:
        df_report = pd.DataFrame.from_dict(report_rows, orient="index")
        df_report.to_csv(os.path.join(out_dir, "val_report.csv"))


if __name__ == "__main__":
    main()
