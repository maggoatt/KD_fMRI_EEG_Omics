# EEGPT-KD: Knowledge Distillation from EEG Teacher to fMRI GNN Student

## What This Is

This folder contains the EEG teacher model (EEGPT) setup, knowledge distillation pipeline,
and the fMRI GNN student model. Together these form the core training pipeline for our
multimodal vigilance classification project.

## Why Knowledge Distillation?

The goal: classify vigilance states (alert / intermediate / drowsy) from fMRI brain graphs
alone, without needing EEG at recording time.

- EEG can classify vigilance well but requires wearing an electrode cap
- fMRI is already collected in many clinical settings
- If we can train an fMRI model to match EEG-level performance, that is useful

KD transfers "soft knowledge" from the EEG teacher to the fMRI student. Instead of just
hard labels (e.g., "drowsy"), the teacher provides probability distributions
(e.g., [0.05, 0.25, 0.70]) that encode inter-class relationships. The student learns
from these richer signals.

Our additional novelty: the fMRI brain graph uses gene expression data from the Allen Human
Brain Atlas as node features, making the GNN biologically informed.

## Architecture Overview

```
EEG signals (30s windows)
    |
    v
[EEGPT - frozen pretrained encoder, 25M params]
    |
    v
[Adaptive Spatial Filter + Linear Head] <-- only these are trained (~2K params)
    |
    v
Soft labels (3-class probability distribution)
    |
    v  (KD loss: KL-divergence + optional feature MSE)
    |
[GNN Student Model]
    ^
    |
fMRI brain graphs (nodes=~210 ROIs, edges=top-k correlations)
  with gene expression node features (294 sleep/circadian genes)
```

## Teacher: EEGPT

EEGPT is a pretrained EEG transformer from NeurIPS 2024.

- 25M parameters (the "large" variant: 512 embed dim, 8/8/8 layers, 4 summary tokens)
- Pretrained on mixed multi-task EEG data (PhysioMI, HGD, TSU, SEED, M3CV)
- Already SOTA on Sleep-EDFx 5-class sleep staging (69.2% balanced accuracy)
- Input: 58 channels, 256 Hz, configurable window length
- We use it with linear probing: freeze the encoder, train only a small classification head

We do NOT fine-tune the full model. With ~800 samples, fine-tuning 25M params would overfit.
Linear probing trains only ~2K parameters (spatial filter + linear layer), which is safe.

Links:
- Paper: https://proceedings.neurips.cc/paper_files/paper/2024/file/4540d267eeec4e5dbd9dae9448f0b739-Paper-Conference.pdf
- Code: https://github.com/BINE022/EEGPT
- Checkpoint: eegpt_mcae_58chs_4s_large4E.ckpt (download from Figshare, linked in EEGPT repo)

## Student: fMRI GNN

A Graph Neural Network (GCN or GAT) operating on brain connectivity graphs.

- Nodes: ~210 ROIs (Schaefer 200 cortical + 10 subcortical from FreeSurfer)
- Edges: top k=10 per node by absolute fMRI correlation
- Node features: fMRI connectivity profile (210-dim) concatenated with gene expression (294-dim)
- Output: 3-class (alert, intermediate, drowsy)

## KD Loss Function

Standard Hinton KD with optional feature alignment:

```python
# Core KD loss (~5 lines)
T = 2.0
student_soft = F.log_softmax(student_logits / T, dim=-1)
teacher_soft = F.softmax(teacher_logits / T, dim=-1)
kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T ** 2)

# Combined training loss
loss = alpha * CrossEntropy(student_logits, true_labels) + (1 - alpha) * kd_loss
```

Reference hyperparams (from FewShotKDVigilance):
- Temperature T = 2.0
- Two-phase schedule: KD-heavy early (alpha=0.2), CE-heavy later (alpha=0.8)
- 30 epochs, AdamW, OneCycleLR, batch_size=32

## Dataset

- NatView simultaneous EEG-fMRI, 22 subjects, resting state, TR=2.1s, 600s per scan
- 30-second intervals -> ~20 per subject -> ~800 total samples (some subjects have 2 sessions -> ~40 sessions total)
- Labels: EEG-derived vigilance states via VIGALL / alpha-theta ratio
- Gene expression: Allen Human Brain Atlas, donor 9861, ~294 sleep/circadian genes

## Evaluation Strategy

Leave-One-Subject-Out (LOSO) cross-validation:
- 22 folds, each trains on 21 subjects (~760 samples), tests on 1 (~40 samples)
- No data leakage (subject-level splits)
- This is the standard evaluation method in EEG/neuroimaging research
- Metrics: balanced accuracy, weighted F1, Cohen's kappa

## Experiments to Run

1. **Baseline (supervised, hard labels)**: Train GNN with hard vigilance labels only. No KD.
2. **KD (soft labels)**: Train GNN with KD from EEGPT teacher.
3. **Genomics ablation**: Compare GNN with vs without gene expression node features.
4. **Combined**: KD + genomics features (the full pipeline).
5. **(Optional) Fine-tune vs linear probe**: Show linear probing is better for the teacher with this sample size.

## Actionable Steps

### Phase 1: Teacher Setup

- [ ] 1.1 Clone EEGPT repo or install as dependency
- [ ] 1.2 Download the pretrained checkpoint (eegpt_mcae_58chs_4s_large4E.ckpt)
- [ ] 1.3 Write adapter to load EEGPT encoder in inference mode (frozen weights)
- [ ] 1.4 Implement adaptive spatial filter to map NatView EEG channels to EEGPT's 58-channel space
- [ ] 1.5 Implement linear classification head (512 -> 3)
- [ ] 1.6 Train linear probe on EEG data using LOSO, verify teacher accuracy
      - Target: 65%+ balanced accuracy on 3-class vigilance (comparable to Sleep-EDFx results)

### Phase 2: Student GNN

- [ ] 2.1 Define GNN architecture (GCNConv or GATConv, PyTorch Geometric)
- [ ] 2.2 Write data loader for fMRI brain graphs (from Maggie's preprocessing output)
- [ ] 2.3 Integrate gene expression node features (from Gautham's CSV)
- [ ] 2.4 Train supervised baseline (hard labels, no KD) using LOSO
      - This is the comparison point for KD

### Phase 3: Knowledge Distillation

- [ ] 3.1 Implement KD loss function (KL-divergence with temperature)
- [ ] 3.2 Generate soft labels from teacher for all training samples
- [ ] 3.3 Train student GNN with combined loss (CE + KD)
- [ ] 3.4 Implement two-phase training schedule (KD-heavy then CE-heavy)

### Phase 4: Evaluation and Ablations

- [ ] 4.1 Run all LOSO folds for each experiment variant
- [ ] 4.2 Compute metrics: balanced accuracy, weighted F1, Cohen's kappa
- [ ] 4.3 Compare: baseline vs KD vs genomics vs KD+genomics
- [ ] 4.4 Generate results tables and confusion matrices

## Dependencies

```
torch >= 2.0
torch-geometric
mne (EEG processing)
numpy, pandas, scipy, scikit-learn
```

## Folder Structure (planned)

```
EEGPT-KD/
  README.md              # This file
  teacher/
    eegpt_adapter.py     # Load and wrap EEGPT for inference
    spatial_filter.py    # Map NatView channels -> EEGPT 58-ch
    linear_probe.py      # Classification head training
  student/
    gnn_model.py         # GNN architecture definition
    graph_dataset.py     # PyG dataset for fMRI brain graphs
  distillation/
    kd_loss.py           # KD loss functions
    trainer.py           # Training loop with LOSO
  evaluation/
    metrics.py           # Balanced accuracy, F1, kappa
    ablation.py          # Run all experiment variants
  checkpoints/           # Saved model weights (gitignored)
  configs/               # Hyperparameter configs
```

## Key References

- EEGPT paper (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/file/4540d267eeec4e5dbd9dae9448f0b739-Paper-Conference.pdf
- EEGPT code: https://github.com/BINE022/EEGPT
- FewShotKDVigilance (Vanderbilt, similar approach with 3 teachers): https://github.com/neurdylab/FewShotKDVigilance
- Vanderbilt KD paper: https://direct.mit.edu/imag/article/doi/10.1162/IMAG.a.91/131628
- NatView dataset: https://fcon_1000.projects.nitrc.org/indi/retro/nat_view.html
- Allen Human Brain Atlas: https://alleninstitute.org/science-resource/allen-human-brain-atlas-tutorial/
- Hinton KD paper: https://arxiv.org/abs/1503.02531
