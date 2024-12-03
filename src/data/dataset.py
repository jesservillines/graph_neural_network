from torch_geometric.data import Dataset, Data
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PatientDataset(Dataset):
    def __init__(self, df, features, target='length_of_stay', edge_threshold=0.5, transform=None):
        super(PatientDataset, self).__init__(transform)
        self.df = df
        self.features = features
        self.target = target
        self.edge_threshold = edge_threshold
        self.scaler = StandardScaler()
        
        # Prepare features and target
        self.X = self.scaler.fit_transform(df[features])
        self.y = df[target].values
        
        # Create similarity matrix once
        from sklearn.metrics.pairwise import cosine_similarity
        self.similarity_matrix = cosine_similarity(self.X)
        
    def len(self):
        return len(self.df)
    
    def get(self, idx):
        # Get node features
        x = torch.tensor(self.X[idx].reshape(1, -1), dtype=torch.float)
        y = torch.tensor([self.y[idx]], dtype=torch.float)
        
        # Create edges based on similarity
        similar_nodes = np.where(self.similarity_matrix[idx] > self.edge_threshold)[0]
        if len(similar_nodes) > 0:
            edge_index = torch.tensor([[idx, n] for n in similar_nodes if n != idx], 
                                    dtype=torch.long).t().contiguous()
        else:
            # If no similar nodes, create self-loop
            edge_index = torch.tensor([[idx, idx]], dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index, y=y)

class CrossValidationSplit:
    def __init__(self, dataset, n_splits=5, test_size=0.2, random_state=42):
        self.dataset = dataset
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.current_fold = 0
        
        # Create CV splits
        self.splits = self._create_splits()
    
    def _create_splits(self):
        n_samples = len(self.dataset)
        indices = np.arange(n_samples)
        
        # First split out test set
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Create k-fold splits from training data
        fold_size = len(train_idx) // self.n_splits
        splits = []
        
        for i in range(self.n_splits):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.n_splits - 1 else len(train_idx)
            
            val_idx = train_idx[start_idx:end_idx]
            train_fold_idx = np.concatenate([
                train_idx[:start_idx],
                train_idx[end_idx:]
            ])
            
            splits.append({
                'train': train_fold_idx,
                'val': val_idx,
                'test': test_idx
            })
        
        return splits
    
    def get_split(self, fold):
        if fold < 0 or fold >= self.n_splits:
            raise ValueError(f"Fold {fold} is out of range [0, {self.n_splits-1}]")
        return self.splits[fold]