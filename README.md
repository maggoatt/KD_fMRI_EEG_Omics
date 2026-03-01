# Multimodal Knowledge Distillation for Alertness State Classification

The purpose of the overall project is to experiment with the development of an EEG-to-fMRI knowledge distillation pipeline. This utilizes an EEG transformer teacher model and fMRI graph neural network (GNN) student model. Omics-level information is used to further inform the knowledge distillation pipeline, i.e. inform the fMRI GNN on what ROIs to focus on. 

[Results from Preprocessed NatView Dataset](https://drive.google.com/drive/folders/1RkT0beeVy4T-Ohf7qguOaulGWsSE-S8y?usp=drive_link)

### Setup
1.  ```pip install``` all packages/libraries in ```requirements.txt```
2. Run ```pip install -e .``` in the project directory to install all modules (i.e. BrainOmicsDataset)
3. Set up an ```.env``` file in accordance to ```.env.example```