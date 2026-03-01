import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from dataset.create_omics_dataset import BrainOmicsDataset
from dataset.ROI_connectivity_dataset import ROIConnectivityDataset
from downstream_analysis.expression_analysis import SleepROIDiscovery

class GeneExpressionAutoencoder(nn.Module):
    def __init__(self, n_genes=300, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_genes + 1, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 209),  # 209 = 210 ROIs - 1 (self)
            nn.Tanh()  # Correlations in [-1, 1]
        )

    def encode(self, expression, state):
        x = torch.cat([expression, state], dim=-1)  # (batch, 297)
        return self.encoder(x)
    
    def decode(self, latent):
        recon = self.decoder(latent)
        return recon
    
    def forward(self, expression, state):
        latent = self.encode(expression, state)
        recon = self.decode(latent)
        return recon, latent
    

def train(model, train_loader, test_loader, epochs=100, batch_size=64, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=15, factor=0.5, verbose=True
    )
    
    best_test_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            gene_expr = batch['gene_expr'].to(device)
            state = batch['state'].to(device)
            target = batch['connectivity'].to(device)
            
            # Forward
            reconstructed, latent = model(gene_expr, state)
            loss = F.mse_loss(reconstructed, target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        # Test
        model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                gene_expr = batch['gene_expr'].to(device)
                state = batch['state'].to(device)
                target = batch['connectivity'].to(device)
                
                reconstructed, latent = model(gene_expr, state)
                loss = F.mse_loss(reconstructed, target)
                
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        
        # Scheduler
        scheduler.step(test_loss)
        
        # Save best
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            #torch.save(model.state_dict(), 'gene_to_connectivity_autoencoder.pt')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}/{epochs}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}, "
                  f"Best: {best_test_loss:.4f}")
        
        # Early stopping
        if patience_counter > 30:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print(f"\nTraining complete!")
    print(f"  Best test loss: {best_test_loss:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('gene_to_connectivity_autoencoder.pt'))
    
    return model


def reconstruction(model, expression_data, critical_rois):
    model.eval()
    device = next(model.parameters()).device

    reconstructions = {
        'awake': np.zeros((len(critical_rois), 209)),
        'drowsy': np.zeros((len(critical_rois), 209))
    }

    with torch.no_grad():
        for i, roi in enumerate(critical_rois):
            gene_expr = torch.FloatTensor(expression_data[roi, :]).unsqueeze(0).to(device)
            
            # Awake state (state=0)
            state_awake = torch.FloatTensor([[0]]).to(device)
            recon_awake, _ = model(gene_expr, state_awake)
            reconstructions['awake'][i, :] = recon_awake.cpu().squeeze().numpy()
            
            # Drowsy state (state=1)
            state_drowsy = torch.FloatTensor([[1]]).to(device)
            recon_drowsy, _ = model(gene_expr, state_drowsy)
            reconstructions['drowsy'][i, :] = recon_drowsy.cpu().squeeze().numpy()
    
    print(f"  Awake: {reconstructions['awake'].shape}")
    print(f"  Drowsy: {reconstructions['drowsy'].shape}")
    return reconstructions
    

def state_differences(reconstructions, critical_rois):
    diff = reconstructions['drowsy'] - reconstructions['awake']
    mean_diff = np.mean(diff, axis=0)
    
    roi_mean_change = diff.mean(axis=1)
    roi_max_change = np.abs(diff).max(axis=1)

    most_dynamic_idx = np.argsort(np.abs(roi_mean_change))[-10:][::-1]
    for rank, idx in enumerate(most_dynamic_idx):
        roi = critical_rois[idx]
        print(f"  {rank+1}. ROI {roi}: Mean Change={roi_mean_change[idx]:.4f}, Max Change={roi_max_change[idx]:.4f}")

    # plot top 10 most dynamic ROIs
    top_10_rois = critical_rois[most_dynamic_idx]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_10_rois, y=roi_mean_change[most_dynamic_idx])
    plt.title("Mean Connectivity Change (Drowsy - Awake) for Top 10 Dynamic ROIs")
    plt.xlabel("ROI Index")
    plt.ylabel("Mean Connectivity Change")
    plt.show()

    
    return diff

if __name__ == "__main__":
    savepath = '/Users/gautham/deep_learning/multimodal_brain_proj/KD_fMRI_EEG_Omics/results/'
    dataset = BrainOmicsDataset(
        fmri_path='/Users/gautham/deep_learning/multimodal_brain_proj/KD_fMRI_EEG_Omics/sample_data/60s_interval_corr_matrices/',
        expression_path='/Users/gautham/deep_learning/multimodal_brain_proj/KD_fMRI_EEG_Omics/sample_data/gene_expression_schaefer210.npy',
        k=10,
        use_expression=True
    )
    batch_size = 64
    expression_data = dataset.expression_data
    roi_dataset = ROIConnectivityDataset(dataset, expression_data)

    sample_indices = list(set([s['sample_idx'] for s in roi_dataset.samples]))
    np.random.shuffle(sample_indices)
    
    train_sample_idx = set(sample_indices[:int(0.8 * len(sample_indices))])
    
    train_samples = [s for s in roi_dataset.samples if s['sample_idx'] in train_sample_idx]
    test_samples = [s for s in roi_dataset.samples if s['sample_idx'] not in train_sample_idx]

    train_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_samples, batch_size=batch_size, shuffle=False)

    model = GeneExpressionAutoencoder(n_genes=expression_data.shape[1], latent_dim=32)
    trained_model = train(model, train_loader, test_loader, epochs=100, batch_size=batch_size, lr=1e-3)

    analysis = SleepROIDiscovery(dataset, dataset.expression_data, gene_names_path=f'/Users/gautham/deep_learning/multimodal_brain_proj/KD_fMRI_EEG_Omics/sample_data/gene_names.npy')
    results = analysis.run_complete_analysis()
    critical_rois = results['critical_rois']

    reconstructions = reconstruction(trained_model, expression_data, critical_rois)
    diff = state_differences(reconstructions, critical_rois)

    np.save(savepath + 'idealized_awake_connectivity.npy', reconstructions['awake'])
    np.save(savepath + 'idealized_drowsy_connectivity.npy', reconstructions['drowsy'])
    np.save(savepath + 'connectivity_differences.npy', diff)
