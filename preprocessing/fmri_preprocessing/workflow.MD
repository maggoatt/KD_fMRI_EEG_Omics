# fMRI Pre-Processing Pipeline

This document serves as a "live" context to describe the workflow of the ```fmri_preprocessing``` directory, and the significance of pre-processing in the context of the entire project.

Manually written by Maggie Zhang.

## 1. Background

The purpose of the overall project is to develop an EEG-to-fMRI knowledge distillation pipeline with an EEG transformer teacher model and fMRI graph neural network (GNN) student model. Omics-level information is used to further inform the knowledge distillation pipeline, i.e. inform the fMRI GNN on what ROIs to focus on. 

## 2. Datasets

The Simultaneous EEG-fMRI during resting state fMRI scan (TR=2.1s, 600s) from the EEG/FMRI Naturalistic Viewing (NatView) Dataset is used for this project. The NatView dataset contains 22 participants.

The Allen Human Brain Atlas (AHBA) dataset is used for the omics-level extraction.

The NatView task-rest dataset utilizes the Schaefer 2018 7-network atlas, where we select the 200-parcel/density variant. 

## 3. Current Workflow Plans

The ABHA dataset combined with Schaefer 2018 atlas will extract gene expressions for ROIs and genes. About 240 genes were extracted from literature and genome-wide association studies, but this number is subject to change depending on the final ROI count. The columns correlate to genes, and the rows are Schaefer ROI number (1-200, also subject to change; see Challenges and Questions section). Values are gene expression in each ROI for that gene. 

The EEG data will be processed as follows:

- Extract alpha, theta, and delta bands from EEG data.
- Segment EEG data into 1-minute-long epochs/intervals, where each interval will be classified as alert, intermediate, or drowsy based on their calculated vigilance score.
- Vigilance scores will be calculated as the ratio of alpha/theta OR alpha/(theta+delta) averaged across a given interval.
- Scores and epochs will be transformed via HRF delay to match fMRI timeseries, thus creating ground truth labels for the fMRI data.
- A transformer model will likely be used to represent the EEG data and predict the labels per epoch. See Challenges and Questions section for more specific inquiries.

The fMRI data will be processed as follows:

- Currently, the NatView NKI derivatives include pre-extracted Schaefer 2018 timeseries. Files look like:  
         ```sub-01_ses-01_task-rest_space-MNI152Lin_res-3mm_atlas-Schaefer2018_dens-200parcels7networks_desc-sm0_bold.tsv```
Plus GSR variants (desc-sm0gsr). These are ready-to-use TSVs of parcel-level timeseries.
- Create correlation matrices from the parcellated ROIs from the Schaefer 2018 atlas. See Challenges and Questions section for inquires about subcortical region inclusion.
- Once EEG epochs are segmented, labeled, and coupled with the fMRI timeseries, create functional connectivity (FC) graphs per epoch, with only the top 'k' edges selected/represented per ROI (see Challenges and Questions section for specific inquiries related to graph construction methods).

Once all data is processed, the knowledge distillation pipeline should be built, with the EEG transformer (or another model if better) model as the teacher, the fMRI graph neural network as the student, and omics as an attention/informant mechanism.


## Challenges and Questions
What is the "best" model to use to represent the EEG data? I am familiar with ```EEGPT```, a pretrained transformer model for representing EEG data that I stumbled across during literature review. Can this be used for the project? Are there any additional pre-processing steps to consider? Are there any widely-used/reliable models that could be used instead of ```EEGPT```?

What are GSR variants in the fMRI data, and what are the ready-to-use TSVs of parcel-level timeseries? Does this mean that correlation matrices are ready to be built? Is there any additional processing I should complete, i.e. GLM creation, low-pass filtering, confounds filtering, etc.?

How would the knowledge distillation model be constructed? 

Are the current correlation matrices implemented as an average across the timeseries? How should I save the final k=10 edge-filtered graph to send to my teammates? Would a TSV with zeroed-out edges not in k<=10 be better? Or should I store it as a PyG dataset (i.e. InMemoryDataset)? Can I now create FC graphs per ~60s interval within the timeseries now?

## Literature to Consider
Primary inspiration behind the project: [EEG-to-fMRI knowledge distillation empowers few-shot resting-state fMRI vigilance detection](https://spie.org/medical-imaging/presentation/EEG-to-fMRI-knowledge-distillation-empowers-few-shot-resting-state/13925-1) ([GitHub Repository Code Release](https://github.com/neurdylab/FewShotKDVigilance/tree/main))

Ground truth method: [Multimodal state-dependent connectivity analysis of arousal and autonomic centers in the brainstem and basal forebrain](https://direct.mit.edu/imag/article/doi/10.1162/IMAG.a.91/131628) (See Section 2.5)

Potential/intriguing inference model: [NeuroBOLT: Resting-state EEG-to-fMRI Synthesis with Multi-dimensional Feature Mapping](https://arxiv.org/abs/2410.05341) ([GitHub Repository Code Release](https://github.com/soupeeli/NeuroBOLT))

## Metadata
Last updated by Maggie on 2/15/2026.