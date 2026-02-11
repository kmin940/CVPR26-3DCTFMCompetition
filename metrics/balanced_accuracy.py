import torch
from torchmetrics import Metric
from torchmetrics.functional.classification import stat_scores

import torch
from torchmetrics import Metric
from torchmetrics.functional.classification import stat_scores
from typing import Optional


class MultiLabelBalancedAccuracy(Metric):
    def __init__(self, num_labels, threshold=0.5):
        super().__init__()
        self.num_labels = num_labels
        self.threshold = threshold
        self.add_state("tp", default=torch.zeros(num_labels), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_labels), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(num_labels), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_labels), dist_reduce_fx="sum")

    def update(self, probs, target):
        preds = (probs >= self.threshold).long()
        target = target.long()

        for i in range(self.num_labels):
            mask = target[:, i] != -1
            if mask.sum() == 0:
                continue

            p = preds[mask, i]
            t = target[mask, i]

            self.tp[i] += ((p == 1) & (t == 1)).sum()
            self.fp[i] += ((p == 1) & (t == 0)).sum()
            self.tn[i] += ((p == 0) & (t == 0)).sum()
            self.fn[i] += ((p == 0) & (t == 1)).sum()

    def compute(self):
        recall = self.tp / (self.tp + self.fn + 1e-8)
        spec = self.tn / (self.tn + self.fp + 1e-8)
        return ((recall + spec) / 2).mean()

class BalancedAccuracy(Metric):
    def __init__(
        self,
        task: str = "multiclass",
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        dist_sync_on_step: bool = False,
        **kwargs
    ):
        assert task in {
            "binary",
            "multiclass",
            "multilabel",
        }, "Only 'binary', 'multiclass', and 'multilabel' tasks are supported."
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.task = task
        self.threshold = threshold
        self.ignore_index = ignore_index

        # Determine the number of state elements needed
        if task == "binary":
            num_state_elements = 1
            if num_classes is not None and num_classes != 2:
                pass
            self.num_classes = 2
        elif task == "multiclass":
            if not isinstance(num_classes, int) or num_classes < 2:
                raise ValueError(f"`num_classes` must be an integer >= 2 for task '{task}'.")
            num_state_elements = num_classes
            self.num_classes = num_classes
        elif task == "multilabel":
            if not isinstance(num_classes, int) or num_classes < 1:
                 raise ValueError(f"`num_labels` must be an integer >= 1 for task '{task}'.")
            num_state_elements = num_classes
            self.num_classes = num_classes
        else:
            raise ValueError(f"Task {task} not supported!")

        self.add_state("tp", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        target = target.to(torch.long)
        stats = None

        if self.task == "binary":
            if preds.ndim > 1 and preds.size(-1) == 1:
                preds = preds.squeeze(-1)
            
            if preds.max() > 1.0 or preds.min() < 0.0:
                preds = torch.sigmoid(preds)
                
            hard_preds = (preds >= self.threshold).long()

            stats = stat_scores(
                preds=hard_preds,
                target=target,
                task="binary",
                threshold=self.threshold,
                average="none",
            )

        elif self.task == "multilabel":
            if preds.max() > 1.0 or preds.min() < 0.0:
                preds = torch.sigmoid(preds)

            hard_preds = (preds >= self.threshold).long()

            # Handle ignore_index by computing stats manually per label
            if self.ignore_index is not None:
                # Create mask for valid entries (not ignored)
                valid_mask = (target != self.ignore_index)  # (B, num_labels)
                
                # Compute stats per label, only counting valid entries
                tp = torch.zeros(self.num_classes, device=preds.device)
                fp = torch.zeros(self.num_classes, device=preds.device)
                tn = torch.zeros(self.num_classes, device=preds.device)
                fn = torch.zeros(self.num_classes, device=preds.device)
                
                for label_idx in range(self.num_classes):
                    mask = valid_mask[:, label_idx]  # (B,)
                    if mask.sum() == 0:
                        continue
                    
                    pred_label = hard_preds[:, label_idx][mask]  # valid preds for this label
                    target_label = target[:, label_idx][mask]    # valid targets for this label
                    
                    tp[label_idx] = ((pred_label == 1) & (target_label == 1)).sum()
                    fp[label_idx] = ((pred_label == 1) & (target_label == 0)).sum()
                    tn[label_idx] = ((pred_label == 0) & (target_label == 0)).sum()
                    fn[label_idx] = ((pred_label == 0) & (target_label == 1)).sum()
                
                self.tp += tp
                self.fp += fp
                self.tn += tn
                self.fn += fn
                return  # Early return since we've already updated states
            
            else:
                stats = stat_scores(
                    preds=hard_preds,
                    target=target,
                    task="multilabel",
                    num_labels=self.num_classes,
                    average=None,
                )
        
        elif self.task == "multiclass":
            if preds.ndim == 2 and preds.size(1) == self.num_classes:
                preds = torch.argmax(preds, dim=1)

            stats = stat_scores(
                preds=preds,
                target=target,
                task="multiclass",
                num_classes=self.num_classes,
                average=None,
            )
        
        if stats.ndim == 1:
            stats = stats.unsqueeze(0)

        tp, fp, tn, fn, _ = stats.unbind(dim=1)
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self) -> torch.Tensor:
        """Compute the final Balanced Accuracy score."""
        recall = self.tp / (self.tp + self.fn + 1e-8)
        specificity = self.tn / (self.tn + self.fp + 1e-8)
        balanced_acc = (recall + specificity) / 2
        return balanced_acc.mean()
# class BalancedAccuracy(Metric):
#     def __init__(
#         self,
#         task: str = "multiclass", # Required to determine logic
#         num_classes: Optional[int] = None, # Optional if task is 'binary'
#         threshold: float = 0.5,
#         dist_sync_on_step: bool = False,
#         **kwargs
#     ):
#         # 1. Input Validation and Task Determination
#         assert task in {
#             "binary",
#             "multiclass",
#             "multilabel",
#         }, "Only 'binary', 'multiclass', and 'multilabel' tasks are supported."
#         super().__init__(dist_sync_on_step=dist_sync_on_step)

#         self.task = task
#         self.threshold = threshold

#         # 2. State Initialization based on Task
        
#         # Determine the number of state elements needed
#         if task == "binary":
#             # For binary, we only need a single element for the positive class metrics.
#             # We enforce num_classes = 1 for state storage simplicity, even though the problem has 2 classes.
#             num_state_elements = 1
#             if num_classes is not None and num_classes != 2:
#                 # Issue a warning or handle mismatch if num_classes is explicitly set but wrong
#                 pass
#             self.num_classes = 2 # Store 2, but use 1 state element for binary
#         elif task == "multiclass":
#             if not isinstance(num_classes, int) or num_classes < 2:
#                 raise ValueError(f"`num_classes` must be an integer >= 2 for task '{task}'.")
#             num_state_elements = num_classes
#             self.num_classes = num_classes
#         elif task == "multilabel":
#             if not isinstance(num_classes, int) or num_classes < 1:
#                  raise ValueError(f"`num_labels` must be an integer >= 1 for task '{task}'.")
#             num_state_elements = num_classes
#             # Using num_classes for num_labels consistency
#             self.num_classes = num_classes
#         else:
#             raise ValueError(f"Task {task} not supported!") # Should be caught by assert

#         # Initialize state: use a vector of size num_state_elements
#         self.add_state("tp", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")
#         self.add_state("fp", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")
#         self.add_state("tn", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")
#         self.add_state("fn", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")

#     def update(self, preds: torch.Tensor, target: torch.Tensor):
#         target = target.to(torch.long)
#         stats = None

#         if self.task == "binary":
#             # Flatten to (N) if it's (N, 1)
#             if preds.ndim > 1 and preds.size(-1) == 1:
#                 preds = preds.squeeze(-1)
            
#             # Auto-detect logits vs probs (must be done before stat_scores call if not passed hard preds)
#             if preds.max() > 1.0 or preds.min() < 0.0:
#                 preds = torch.sigmoid(preds)
                
#             # Convert to hard predictions before passing to stat_scores for binary
#             hard_preds = (preds >= self.threshold).long()

#             stats = stat_scores(
#                 preds=hard_preds,
#                 target=target,
#                 task="binary",
#                 threshold=self.threshold,
#                 average="none", # Return [tp, fp, tn, fn, support] for the positive class
#             )

#         elif self.task == "multilabel":
#             # Auto-detect logits vs probs
#             if preds.max() > 1.0 or preds.min() < 0.0:
#                 preds = torch.sigmoid(preds)

#             preds = (preds >= self.threshold).long()

#             stats = stat_scores(
#                 preds=preds,
#                 target=target,
#                 task="multilabel",
#                 num_labels=self.num_classes,
#                 average=None,
#             )
        
#         elif self.task == "multiclass":
#             # If input is probabilities/logits (N, C), apply argmax to get hard preds (N)
#             if preds.ndim == 2 and preds.size(1) == self.num_classes:
#                 preds = torch.argmax(preds, dim=1)

#             stats = stat_scores(
#                 preds=preds,
#                 target=target,
#                 task="multiclass",
#                 num_classes=self.num_classes,
#                 average=None,
#             )
        
#         # Stat scores returns shape (C, 5) for multi/multi-label, or (5) for binary (average=None)
#         if stats.ndim == 1:
#             stats = stats.unsqueeze(0)  # make it (1, 5) to unbind along dim=1

#         tp, fp, tn, fn, _ = stats.unbind(dim=1)
#         self.tp += tp
#         self.fp += fp
#         self.tn += tn
#         self.fn += fn

#     def compute(self) -> torch.Tensor:
#         """Compute the final Balanced Accuracy score."""
        
#         # Recall (True Positive Rate) = TP / (TP + FN)
#         recall = self.tp / (self.tp + self.fn + 1e-8)
        
#         # Specificity (True Negative Rate) = TN / (TN + FP)
#         specificity = self.tn / (self.tn + self.fp + 1e-8)
        
#         balanced_acc = (recall + specificity) / 2
        
#         # Return the mean of the per-class/per-label balanced accuracies
#         return balanced_acc.mean()

# class BalancedAccuracy(Metric):
#     def __init__(
#             self,
#             num_classes: int,
#             task: str = "multiclass",
#             threshold: float = 0.5,
#             dist_sync_on_step=False,
#     ):
#         assert task in {
#             "multiclass",
#             "multilabel",
#         }, "Only 'multiclass' and 'multilabel' tasks are supported."
#         super().__init__(dist_sync_on_step=dist_sync_on_step)

#         self.num_classes = num_classes
#         self.task = task
#         self.threshold = threshold

#         self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
#         self.add_state("fp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
#         self.add_state("tn", default=torch.zeros(num_classes), dist_reduce_fx="sum")
#         self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx="sum")

#     def update(self, preds: torch.Tensor, target: torch.Tensor):

#         target = target.to(torch.long)
#         if self.task == "multilabel":

#             # Auto-detect logits vs probs
#             if preds.max() > 1.0 or preds.min() < 0.0:
#                 preds = torch.sigmoid(preds)

#             preds = (preds >= self.threshold).long()

#             stats = stat_scores(
#                 preds=preds,
#                 target=target,
#                 task="multilabel",
#                 num_labels=self.num_classes,
#                 average=None,
#             )
#         elif self.task == "multiclass":
#             if preds.ndim == 2 and preds.size(1) == self.num_classes:
#                 preds = torch.argmax(preds, dim=1)

#             stats = stat_scores(
#                 preds=preds,
#                 target=target,
#                 task="multiclass",
#                 num_classes=self.num_classes,
#                 average=None,
#             )

#         if stats.ndim == 1:
#             stats = stats.unsqueeze(0)  # make it 2D to unbind along dim=1

#         tp, fp, tn, fn, _ = stats.unbind(dim=1)
#         self.tp += tp
#         self.fp += fp
#         self.tn += tn
#         self.fn += fn

#     def compute(self):
#         recall = self.tp / (self.tp + self.fn + 1e-8)
#         specificity = self.tn / (self.tn + self.fp + 1e-8)
#         balanced_acc = (recall + specificity) / 2
#         return balanced_acc.mean()
