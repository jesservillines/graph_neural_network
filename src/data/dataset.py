import torch
from torch_geometric.data import Data, Dataset

class GraphDataset(Dataset):
    def __init__(self, features, edge_index, targets, indices):
        super().__init__()
        self.features = features
        self.edge_index = edge_index
        self.targets = targets
        self._indices = indices
    
    def len(self):
        return len(self._indices)
    
    def get(self, idx):
        # Get actual index from indices list
        actual_idx = int(self._indices[idx])
        
        # Ensure features are 2D
        features = self.features[actual_idx]
        if features.dim() == 1:
            features = features.unsqueeze(-1)
        
        # Ensure target is 1D
        target = self.targets[actual_idx]
        if target.dim() == 0:
            target = target.unsqueeze(0)
        
        return Data(
            x=features,
            edge_index=self.edge_index,
            y=target
        )
    
    def indices(self):
        return self._indices