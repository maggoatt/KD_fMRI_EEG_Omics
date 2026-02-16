# Gene Expression Incorporation
## Dataset information
Using Allen atlas, we extract highly relevant genes per ROI from Schaffer based on literature and a GWAS study on sleep duration. The gene expression data would give us the molecular profile of each ROI.

## Connectivity Graph
Using Schaffer correlation + sucortical regions from the harvard_oxford atlas, we create a connectivity graph where:
- Nodes = each brain ROI
- Edges = high fmri correlation (top-k) where weights are weighted by correlation.

The graph dataset is created using the BrainOmicsDataset in `create_omics_dataset.py`. The dataset outputs 3 pieces of information:
1. **Graph** from fMRI correlation
2. **Node activity** from fMRI 
3. **Gene Expression** from Allen Brain Atlas

Gene expression is transferred in **parallel** with the graphs and should be incorporated when data is fed into the GNN. 

## Data Structure Visual
Graph:
  Nodes: [0, 1, 2, ..., 209]   Just numbers, no features
  Edges: [(0->5), (0->12), (0->38), ...]   From fMRI correlations
  
Separate Data:
  node_activity: [0.53, 0.61, 0.48, ...]  <- fMRI per ROI
  expression: [[2.1, 1.8, 0.3, ...],      <- Genes per ROI
               [1.9, 2.3, 0.5, ...],
               ...]

```
        ROI 5
         /|\
        / | \
       /  |  \
    ROI 0-+-ROI 12
       \  |  /
        \ | /
         \|/
        ROI 38

Edges based on: fMRI correlation
Node features: NONE (empty nodes)
```

### **After GNN Adds Gene Expression:**
```
        ROI 5
    [fMRI: 0.53, Genes: BMAL1=2.1, CLOCK=1.8, ...]
         /|\
        / | \
       /  |  \
    ROI 0-+-ROI 12
    [fMRI: 0.61, Genes: BMAL1=1.9, HCRT=2.3, ...]
       \  |  /
        \ | /
         \|/
        ROI 38
    [fMRI: 0.48, Genes: ADORA2A=1.2, ...]

```