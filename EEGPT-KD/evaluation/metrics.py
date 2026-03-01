"""Evaluation metrics for vigilance classification.

Provides balanced accuracy, F1, AUROC, and confusion matrix utilities
for binary (alert/drowsy) classification evaluation.
"""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute classification metrics.

    Args:
        y_true: (N,) ground truth labels
        y_pred: (N,) predicted labels
        y_prob: (N, C) class probabilities, optional

    Returns:
        dict with balanced_accuracy, f1, auroc (if y_prob provided)
    """
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                result["auroc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                result["auroc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
        except ValueError:
            # Only one class present in y_true
            result["auroc"] = float("nan")

    return result


def print_results_table(fold_results):
    """Print formatted results table for LOSO folds.

    Args:
        fold_results: list of dicts from LOSOTrainer.run_all_folds()
    """
    print(f"\n{'Fold':<6} {'Subject':<10} {'Bal.Acc':<10} {'F1':<10} {'AUROC':<10}")
    print("-" * 46)

    for r in fold_results:
        auroc_str = f"{r.get('auroc', float('nan')):.3f}"
        print(
            f"{r['fold'] + 1:<6} {r['subject']:<10} "
            f"{r['balanced_accuracy']:.3f}     "
            f"{r['f1']:.3f}     "
            f"{auroc_str}"
        )

    bal_accs = [r["balanced_accuracy"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    print("-" * 46)
    print(
        f"{'Mean':<6} {'':10} "
        f"{np.mean(bal_accs):.3f}     "
        f"{np.mean(f1s):.3f}"
    )
    print(
        f"{'Std':<6} {'':10} "
        f"{np.std(bal_accs):.3f}     "
        f"{np.std(f1s):.3f}"
    )


def confusion_matrix_summary(y_true, y_pred, class_names=None):
    """Print confusion matrix and classification report.

    Args:
        y_true: (N,) ground truth labels
        y_pred: (N,) predicted labels
        class_names: optional list of class name strings
    """
    if class_names is None:
        class_names = ["drowsy", "alert"]

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")

    # Header
    header = "          " + "  ".join(f"{name:>8}" for name in class_names)
    print(header)
    print("          " + "-" * (10 * len(class_names)))

    # Rows
    for i, name in enumerate(class_names):
        row = f"{name:>8}  " + "  ".join(
            f"{cm[i, j]:>8}" for j in range(len(class_names))
        )
        print(row)

    print(
        f"\n{classification_report(y_true, y_pred, target_names=class_names, zero_division=0)}"
    )
