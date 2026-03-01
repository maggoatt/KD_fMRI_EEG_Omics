"""LOSO cross-validation trainer for knowledge distillation.

Trains a GNN student model using soft labels from a teacher model,
evaluating with Leave-One-Subject-Out cross-validation across subjects.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from .kd_loss import KDLoss

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class LOSOTrainer:
    """Leave-One-Subject-Out trainer with knowledge distillation.

    Supports two modes:
    - KD mode: train student with soft labels from teacher (teacher_cache provided)
    - Baseline mode: train student with hard labels only (teacher_cache=None)

    Args:
        config: dict with training hyperparameters:
            hidden_channels (int): GNN hidden width, default 64
            input_dim (int): node feature dimension, default 1
            num_classes (int): output classes, default 2
            lr (float): learning rate, default 0.001
            weight_decay (float): L2 regularization, default 1e-4
            epochs (int): training epochs per fold, default 30
            batch_size (int): DataLoader batch size, default 16
            temperature (float): KD temperature, default 4.0
            alpha_phase1 (float): alpha for phase 1 (KD-heavy), default 0.2
            alpha_phase2 (float): alpha for phase 2 (CE-heavy), default 0.8
            phase1_epochs (int): epochs before switching to phase 2, default 20
            feature_weight (float): feature MSE weight, default 0.0
            device (str): 'cuda' or 'cpu', default auto-detect
    """

    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = {
            "hidden_channels": 64,
            "input_dim": 1,
            "num_classes": 2,
            "lr": 0.001,
            "weight_decay": 1e-4,
            "epochs": 30,
            "batch_size": 16,
            "temperature": 4.0,
            "alpha_phase1": 0.2,
            "alpha_phase2": 0.8,
            "phase1_epochs": 20,
            "feature_weight": 0.0,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        self.config.update(config)

    def _get_alpha(self, epoch):
        """Get alpha value based on training phase."""
        if epoch < self.config["phase1_epochs"]:
            return self.config["alpha_phase1"]
        return self.config["alpha_phase2"]

    def _build_student(self, student_class):
        """Create a fresh student model instance."""
        return student_class(
            input_dim=self.config["input_dim"],
            hidden_channels=self.config["hidden_channels"],
            num_classes=self.config["num_classes"],
        )

    def run_all_folds(self, dataset, student_class, teacher_cache=None):
        """Run LOSO cross-validation across all subjects.

        Args:
            dataset: VigilanceGraphDataset with subject tracking
            student_class: class (not instance) of student model
            teacher_cache: dict with 'logits' and 'features' tensors indexed
                          by sample position, or None for baseline mode

        Returns:
            dict with:
                fold_results: list of per-fold result dicts
                mean_balanced_accuracy: float
                std_balanced_accuracy: float
                mode: 'kd' or 'baseline'
        """
        from evaluation.metrics import compute_metrics

        subjects = dataset.get_subjects()
        fold_results = []
        mode = "kd" if teacher_cache is not None else "baseline"

        print(f"\nRunning LOSO CV ({mode} mode) across {len(subjects)} subjects")
        print("=" * 60)

        for fold_idx, test_subject in enumerate(subjects):
            test_indices = dataset.get_subject_indices(test_subject)
            train_indices = [
                i for i in range(len(dataset))
                if i not in set(test_indices)
            ]

            if len(test_indices) == 0 or len(train_indices) == 0:
                continue

            train_data = [dataset[i] for i in train_indices]
            test_data = [dataset[i] for i in test_indices]

            # Attach teacher cache entries directly to Data objects so
            # shuffling in the DataLoader works correctly.
            kd_mode = teacher_cache is not None
            if kd_mode:
                for i, global_idx in enumerate(train_indices):
                    train_data[i].teacher_logits = teacher_cache["logits"][global_idx]
                    train_data[i].teacher_features = teacher_cache["features"][global_idx]

            train_loader = DataLoader(
                train_data, batch_size=self.config["batch_size"], shuffle=True
            )
            test_loader = DataLoader(
                test_data, batch_size=self.config["batch_size"], shuffle=False
            )

            result = self._train_fold(
                fold_idx, test_subject,
                train_loader, test_loader,
                student_class, kd_mode,
            )
            fold_results.append(result)

            print(
                f"  Fold {fold_idx + 1}/{len(subjects)} "
                f"[{test_subject}]: "
                f"bal_acc={result['balanced_accuracy']:.3f}, "
                f"f1={result['f1']:.3f}"
            )

        # Aggregate results
        bal_accs = [r["balanced_accuracy"] for r in fold_results]
        mean_ba = sum(bal_accs) / len(bal_accs) if bal_accs else 0.0
        std_ba = float(np.std(bal_accs)) if bal_accs else 0.0

        print("=" * 60)
        print(f"Mean balanced accuracy: {mean_ba:.3f} +/- {std_ba:.3f}")

        return {
            "fold_results": fold_results,
            "mean_balanced_accuracy": mean_ba,
            "std_balanced_accuracy": std_ba,
            "mode": mode,
        }

    def _train_fold(
        self, fold_idx, test_subject,
        train_loader, test_loader,
        student_class, kd_mode,
    ):
        """Train and evaluate one LOSO fold.

        Returns dict with metrics for this fold.
        """
        from evaluation.metrics import compute_metrics

        device = self.config["device"]
        student = self._build_student(student_class).to(device)
        optimizer = torch.optim.Adam(
            student.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        ce_criterion = nn.CrossEntropyLoss()

        for epoch in range(self.config["epochs"]):
            student.train()
            alpha = self._get_alpha(epoch) if kd_mode else 1.0
            kd_criterion = KDLoss(
                alpha=alpha,
                temperature=self.config["temperature"],
                feature_weight=self.config["feature_weight"],
            ).to(device)

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                output = student(batch)

                if kd_mode:
                    loss = kd_criterion(
                        output["logits"],
                        batch.teacher_logits,
                        batch.y,
                        student_feats=output["features"],
                        teacher_feats=batch.teacher_features,
                    )
                else:
                    loss = ce_criterion(output["logits"], batch.y)

                loss.backward()
                optimizer.step()

        # Evaluate on test set
        student.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = student(batch)
                probs = torch.softmax(output["logits"], dim=1)
                preds = output["logits"].argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())
                all_probs.append(probs.cpu())

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        y_prob = torch.cat(all_probs).numpy()

        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics["subject"] = test_subject
        metrics["fold"] = fold_idx

        return metrics
