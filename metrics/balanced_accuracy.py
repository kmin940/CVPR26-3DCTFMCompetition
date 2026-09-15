import torch
from torchmetrics import Metric
from torchmetrics.functional.classification import stat_scores

import torch
from torchmetrics import Metric
from torchmetrics.functional.classification import stat_scores
from typing import Optional

class BalancedAccuracy(Metric):
    def __init__(
        self,
        task: str = "multiclass", # Required to determine logic
        num_classes: Optional[int] = None, # Optional if task is 'binary'
        threshold: float = 0.5,
        dist_sync_on_step: bool = False,
        **kwargs
    ):
        # 1. Input Validation and Task Determination
        assert task in {
            "binary",
            "multiclass",
            "multilabel",
        }, "Only 'binary', 'multiclass', and 'multilabel' tasks are supported."
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.task = task
        self.threshold = threshold

        # 2. State Initialization based on Task
        
        # Determine the number of state elements needed
        if task == "binary":
            # For binary, we only need a single element for the positive class metrics.
            # We enforce num_classes = 1 for state storage simplicity, even though the problem has 2 classes.
            num_state_elements = 1
            if num_classes is not None and num_classes != 2:
                # Issue a warning or handle mismatch if num_classes is explicitly set but wrong
                pass
            self.num_classes = 2 # Store 2, but use 1 state element for binary
        elif task == "multiclass":
            if not isinstance(num_classes, int) or num_classes < 2:
                raise ValueError(f"`num_classes` must be an integer >= 2 for task '{task}'.")
            num_state_elements = num_classes
            self.num_classes = num_classes
        elif task == "multilabel":
            if not isinstance(num_classes, int) or num_classes < 1:
                 raise ValueError(f"`num_labels` must be an integer >= 1 for task '{task}'.")
            num_state_elements = num_classes
            # Using num_classes for num_labels consistency
            self.num_classes = num_classes
        else:
            raise ValueError(f"Task {task} not supported!") # Should be caught by assert

        # Initialize state: use a vector of size num_state_elements
        self.add_state("tp", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_state_elements), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        target = target.to(torch.long)
        stats = None

        if self.task == "binary":
            # Flatten to (N) if it's (N, 1)
            if preds.ndim > 1 and preds.size(-1) == 1:
                preds = preds.squeeze(-1)
            
            # Auto-detect logits vs probs (must be done before stat_scores call if not passed hard preds)
            if preds.max() > 1.0 or preds.min() < 0.0:
                preds = torch.sigmoid(preds)
                
            # Convert to hard predictions before passing to stat_scores for binary
            hard_preds = (preds > self.threshold).long()

            stats = stat_scores(
                preds=hard_preds,
                target=target,
                task="binary",
                threshold=self.threshold,
                average="none", # Return [tp, fp, tn, fn, support] for the positive class
            )

        elif self.task == "multilabel":
            # Auto-detect logits vs probs
            if preds.max() > 1.0 or preds.min() < 0.0:
                preds = torch.sigmoid(preds)

            preds = (preds > self.threshold).long()

            stats = stat_scores(
                preds=preds,
                target=target,
                task="multilabel",
                num_labels=self.num_classes,
                average=None,
            )
        
        elif self.task == "multiclass":
            # If input is probabilities/logits (N, C), apply argmax to get hard preds (N)
            if preds.ndim == 2 and preds.size(1) == self.num_classes:
                preds = torch.argmax(preds, dim=1)

            stats = stat_scores(
                preds=preds,
                target=target,
                task="multiclass",
                num_classes=self.num_classes,
                average=None,
            )
        
        # Stat scores returns shape (C, 5) for multi/multi-label, or (5) for binary (average=None)
        if stats.ndim == 1:
            stats = stats.unsqueeze(0)  # make it (1, 5) to unbind along dim=1

        tp, fp, tn, fn, _ = stats.unbind(dim=1)
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def compute(self) -> torch.Tensor:
        """Compute the final Balanced Accuracy score."""

        recall = self.tp / (self.tp + self.fn + 1e-8)

        # Multiclass: macro-averaged recall — matches sklearn.metrics.balanced_accuracy_score.
        # Averaging in one-vs-rest specificity inflates the score for minority classes
        # (specificity is mechanically ≈1 when the class is rare), so we drop it here.
        if self.task == "multiclass":
            return recall.mean()

        # Binary / multilabel: per-label balanced accuracy = (sensitivity + specificity) / 2.
        # For binary this is the standard definition (and equals sklearn's macro-recall on 2 classes).
        specificity = self.tn / (self.tn + self.fp + 1e-8)
        return ((recall + specificity) / 2).mean()

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
