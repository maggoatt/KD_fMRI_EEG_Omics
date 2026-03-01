"""Knowledge Distillation loss combining hard and soft targets.

Implements the standard KD loss from Hinton et al. (2015):
  L = alpha * CE(student, true) + (1 - alpha) * KL(teacher_soft, student_soft) * T^2
with optional feature-level MSE alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    """Knowledge distillation loss.

    Args:
        alpha: weight for hard label CE loss (1-alpha for soft KD loss)
        temperature: softmax temperature for soft targets
        feature_weight: weight for optional feature MSE loss (0 to disable)
    """

    def __init__(self, alpha=0.5, temperature=4.0, feature_weight=0.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.feature_weight = feature_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.feature_projector = None  # lazily initialized if needed

    def forward(
        self, student_logits, teacher_logits, true_labels,
        student_feats=None, teacher_feats=None,
    ):
        """Compute combined KD loss.

        Args:
            student_logits: (B, C) raw student logits
            teacher_logits: (B, C) raw teacher logits
            true_labels: (B,) ground truth class indices
            student_feats: optional (B, D_s) student features
            teacher_feats: optional (B, D_t) teacher features

        Returns:
            total_loss: scalar tensor
        """
        # Hard label loss
        hard_loss = self.ce_loss(student_logits, true_labels)

        # Soft label KD loss
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (T * T)

        total = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss

        # Optional feature alignment
        if (
            self.feature_weight > 0
            and student_feats is not None
            and teacher_feats is not None
        ):
            # Lazy init projector to match dimensions
            if student_feats.shape[1] != teacher_feats.shape[1]:
                if (
                    self.feature_projector is None
                    or self.feature_projector.in_features != student_feats.shape[1]
                ):
                    self.feature_projector = nn.Linear(
                        student_feats.shape[1], teacher_feats.shape[1]
                    ).to(student_feats.device)
                projected = self.feature_projector(student_feats)
            else:
                projected = student_feats

            feat_loss = F.mse_loss(projected, teacher_feats.detach())
            total = total + self.feature_weight * feat_loss

        return total
