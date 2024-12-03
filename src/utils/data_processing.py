import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

def create_patient_graph(features, edge_threshold=0.5):
    """Create a graph structure from patient data"""
    # Calculate patient similarity matrix
    similarity_matrix = cosine_similarity(features)
    
    # Create edges based on similarity threshold
    edges = np.where(similarity_matrix > edge_threshold)
    edge_index = torch.tensor(np.vstack((edges[0], edges[1])), dtype=torch.long)
    
    # Create node features
    x = torch.tensor(features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index), similarity_matrix