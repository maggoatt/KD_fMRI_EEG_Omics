"""EEGPT Teacher Adapter for Knowledge Distillation.

Wraps the pretrained EEGPT encoder as a frozen feature extractor
and adds a trainable linear classification head for vigilance states.

Input: 30-second EEG windows (7680 samples at 256 Hz, 58 channels).
The "4s" checkpoint was pretrained on 4s windows but the patch embeddings
are length-agnostic (each patch = 64 samples). For downstream tasks,
the reference code (FewShotKDVigilance) uses 30s windows with 120 patches
and adds sinusoidal positional encoding externally.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _add_eegpt_to_path():
    """Add EEGPT vendor directory to sys.path for imports."""
    eegpt_root = Path(__file__).parent.parent / "vendor" / "EEGPT"
    downstream_path = eegpt_root / "downstream"
    if str(downstream_path) not in sys.path:
        sys.path.insert(0, str(downstream_path))
    if str(eegpt_root) not in sys.path:
        sys.path.insert(0, str(eegpt_root))


# Standard 10-20 system channel names used by EEGPT (58 channels)
EEGPT_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
    'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2',
    'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
    'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',
    'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6',
    'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2',
]


def prepare_chan_ids(channel_names):
    """Map channel names to EEGPT indices.

    Args:
        channel_names: list of channel name strings

    Returns:
        torch.Tensor of shape (1, num_channels) with channel indices
    """
    name_to_idx = {name: i for i, name in enumerate(EEGPT_CHANNELS)}
    indices = []
    for name in channel_names:
        if name not in name_to_idx:
            raise ValueError(f"Channel '{name}' not in EEGPT channel set")
        indices.append(name_to_idx[name])
    return torch.tensor([indices], dtype=torch.long)


def temporal_interpolation(x, desired_length, mode="nearest"):
    """Resample EEG temporal dimension to a fixed length.

    Used to normalize any-length EEG windows to the model's expected
    input size (e.g. 7680 samples for 30s at 256 Hz).

    Args:
        x: (B, C, T) or (C, T) EEG tensor
        desired_length: target number of time samples
        mode: interpolation mode ('nearest', 'linear')

    Returns:
        resampled tensor with T = desired_length
    """
    if len(x.shape) == 2:
        return F.interpolate(
            x.unsqueeze(0), desired_length, mode=mode
        ).squeeze(0)
    return F.interpolate(x, desired_length, mode=mode)


def create_sincos_pos_encoding(seq_len, dim):
    """Create 1D sinusoidal positional encoding.

    Matches FewShotKDVigilance create_1d_absolute_sin_cos_embedding.

    Args:
        seq_len: number of positions (e.g. 120 patches)
        dim: embedding dimension (e.g. 2048)

    Returns:
        (seq_len, dim) positional encoding tensor
    """
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class EEGPTTeacher(nn.Module):
    """Frozen EEGPT encoder with trainable classification head.

    Uses pretrained EEGPT as a feature extractor for 30-second EEG windows.
    The encoder processes each window into 120 patches (7680 samples / 64 patch_size),
    each with 2048-dim features (embed_num=4 x embed_dim=512).

    Following the FewShotKDVigilance reference, sinusoidal positional encoding
    is added after the encoder, and features are mean-pooled across patches.

    Args:
        checkpoint_path: path to eegpt_mcae_58chs_4s_large4E.ckpt
        num_classes: number of output classes (default 2: alert/drowsy)
        num_channels: number of EEG channels in input data (default 58)
        embed_dim: EEGPT embedding dimension (default 512)
        embed_num: number of embedding components (default 4)
        window_samples: expected temporal samples per window (default 7680 = 30s at 256Hz)
        patch_size: EEGPT patch size (default 64)
    """

    # 30s at 256 Hz = 7680 samples, 7680 / 64 = 120 patches
    DEFAULT_WINDOW = 256 * 30  # 7680

    def __init__(
        self,
        checkpoint_path=None,
        num_classes=2,
        num_channels=58,
        embed_dim=512,
        embed_num=4,
        window_samples=None,
        patch_size=64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_num = embed_num
        self.feature_dim = embed_dim * embed_num  # 2048
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.window_samples = window_samples or self.DEFAULT_WINDOW
        self.patch_size = patch_size
        self.num_patches = self.window_samples // patch_size  # 120

        # Load EEGPT encoder (configured for 30s input)
        self.encoder = self._build_encoder(checkpoint_path)

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Trainable classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, num_classes),
        )

    def _build_encoder(self, checkpoint_path):
        """Build and load EEGPT encoder from checkpoint.

        The checkpoint was pretrained on 4s windows, but patch embeddings
        are length-agnostic. We instantiate with 30s input (7680 samples)
        following the FewShotKDVigilance reference code.
        """
        _add_eegpt_to_path()
        try:
            from Modules.models.EEGPT_mcae import EEGTransformer
        except ImportError:
            raise ImportError(
                "Cannot import EEGTransformer. Run setup.sh first to clone EEGPT repo."
            )

        encoder = EEGTransformer(
            img_size=[self.num_channels, self.window_samples],
            patch_size=self.patch_size,
            embed_num=self.embed_num,
            embed_dim=self.embed_dim,
            depth=8,
            num_heads=8,
        )

        if checkpoint_path is not None:
            self._load_checkpoint(encoder, checkpoint_path)

        return encoder

    def _load_checkpoint(self, encoder, checkpoint_path):
        """Load pretrained weights from PyTorch Lightning checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Filter keys starting with "target_encoder." and strip prefix
        prefix = "target_encoder."
        prefix_len = len(prefix)  # 15
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[prefix_len:]
                encoder_state[new_key] = value

        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            print(f"Warning: missing keys in encoder: {missing[:5]}...")
        if unexpected:
            print(f"Warning: unexpected keys in encoder: {unexpected[:5]}...")

    def extract_features(self, x, chan_ids=None):
        """Extract pooled features from EEGPT encoder.

        Follows FewShotKDVigilance: encode -> flatten patches to 2048-dim ->
        add sinusoidal positional encoding -> mean pool across patches.

        If input temporal length differs from window_samples, it is
        resampled via temporal_interpolation to match.

        Args:
            x: EEG input tensor (B, C, T) where C=num_channels, T=time_samples
            chan_ids: channel index tensor (B, C) or (1, C), optional

        Returns:
            features: (B, feature_dim) pooled feature vector
        """
        with torch.no_grad():
            # Resample to expected window length if needed
            if x.shape[-1] != self.window_samples:
                x = temporal_interpolation(x, self.window_samples)

            # EEGPT forward: (B, N_patches, embed_num, embed_dim)
            if chan_ids is not None:
                chan_ids = chan_ids.to(x.device)
                if chan_ids.shape[0] == 1 and x.shape[0] > 1:
                    chan_ids = chan_ids.expand(x.shape[0], -1)
                encoder_out = self.encoder(x, chan_ids)
            else:
                encoder_out = self.encoder(x)

            # Flatten embed dimensions: (B, N_patches, 2048)
            h = encoder_out.flatten(2)

            # Add sinusoidal positional encoding (same as reference code)
            pos = create_sincos_pos_encoding(h.shape[1], self.feature_dim)
            h = h + pos.unsqueeze(0).to(h)

            # Mean pool across patches: (B, 2048)
            features = h.mean(dim=1)

        return features

    def forward(self, x, chan_ids=None):
        """Forward pass returning logits and features.

        Args:
            x: EEG input tensor (B, C, T)
            chan_ids: channel index tensor (B, C) or (1, C), optional

        Returns:
            dict with:
                logits: (B, num_classes) raw classification scores
                features: (B, feature_dim) pooled encoder features
        """
        features = self.extract_features(x, chan_ids)
        logits = self.head(features)
        return {"logits": logits, "features": features}

    def get_soft_labels(self, x, chan_ids=None, temperature=4.0):
        """Get temperature-scaled soft probability distributions.

        Args:
            x: EEG input tensor (B, C, T)
            chan_ids: channel index tensor
            temperature: softmax temperature (higher = softer)

        Returns:
            soft_probs: (B, num_classes) softened probabilities
        """
        with torch.no_grad():
            output = self.forward(x, chan_ids)
            soft_probs = F.softmax(output["logits"] / temperature, dim=1)
        return soft_probs

    @torch.no_grad()
    def generate_teacher_cache(self, eeg_dataloader, chan_ids=None, device="cpu"):
        """Pre-compute teacher logits and features for all EEG samples.

        Useful for caching teacher outputs so they don't need to be
        recomputed every epoch during student training.

        Args:
            eeg_dataloader: DataLoader yielding (eeg_tensor, label) batches
            chan_ids: channel indices for EEGPT
            device: computation device

        Returns:
            dict with:
                logits: (N, num_classes) all teacher logits
                features: (N, feature_dim) all teacher features
                labels: (N,) corresponding labels
        """
        self.eval()
        self.to(device)

        all_logits = []
        all_features = []
        all_labels = []

        for batch in eeg_dataloader:
            if isinstance(batch, (list, tuple)):
                eeg, labels = batch[0], batch[1]
            else:
                eeg, labels = batch, None

            eeg = eeg.to(device)
            output = self.forward(eeg, chan_ids)
            all_logits.append(output["logits"].cpu())
            all_features.append(output["features"].cpu())
            if labels is not None:
                all_labels.append(labels.cpu())

        result = {
            "logits": torch.cat(all_logits, dim=0),
            "features": torch.cat(all_features, dim=0),
        }
        if all_labels:
            result["labels"] = torch.cat(all_labels, dim=0)

        return result
