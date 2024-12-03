import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4):
        super(GNNModel, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=num_heads)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=1)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dim * num_heads)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dim)
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.elu(x)
        
        x = global_mean_pool(x, batch)
        x = F.elu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x