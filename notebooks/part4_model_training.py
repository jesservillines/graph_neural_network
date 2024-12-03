# Part 4: Model Training and Evaluation
# This script contains the code for the interactive Jupyter notebook

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from IPython.display import display, HTML
import ipywidgets as widgets
from tqdm.notebook import tqdm
import shap
import matplotlib.pyplot as plt

from src.config.config import ModelConfig
from src.models.gnn_model import GNNModel
from src.models.loss import WeightedMSELoss
from src.training.trainer import ModelTrainer
from torch_geometric.loader import DataLoader
from src.evaluation.evaluator import ModelEvaluator

def load_graph_data():
    """Load prepared graph data"""
    data = torch.load('data/graph_data.pt')
    print("Loaded graph data with features shape:", data['features'].shape)
    return data

def create_data_loaders(data, batch_size=32):
    """Create train/val/test data loaders"""
    # Create indices for splitting
    n_samples = len(data['features'])
    indices = torch.randperm(n_samples)
    
    # Split indices
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Create dataset splits
    from torch_geometric.data import Data, Dataset
    
    class GraphDataset(Dataset):
        def __init__(self, features, edge_index, targets, indices):
            super().__init__()
            self.features = features[indices]
            self.edge_index = edge_index
            self.targets = targets[indices]
            self.indices = indices
        
        def len(self):
            return len(self.indices)
        
        def get(self, idx):
            return Data(
                x=self.features[idx].unsqueeze(0),
                edge_index=self.edge_index,
                y=self.targets[idx].unsqueeze(0)
            )
    
    # Create datasets
    train_dataset = GraphDataset(
        data['features'],
        data['edge_index'],
        data['los_values'],
        train_indices
    )
    
    val_dataset = GraphDataset(
        data['features'],
        data['edge_index'],
        data['los_values'],
        val_indices
    )
    
    test_dataset = GraphDataset(
        data['features'],
        data['edge_index'],
        data['los_values'],
        test_indices
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def create_training_progress_plot():
    """Create interactive plot for training progress"""
    fig = go.FigureWidget()
    fig.add_scatter(name="Training Loss")
    fig.add_scatter(name="Validation Loss")
    
    fig.update_layout(
        title="Training Progress",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        showlegend=True
    )
    
    return fig

def train_model_with_visualization(model, train_loader, val_loader, num_epochs=100):
    """Train model with interactive visualization"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create progress plot
    progress_plot = create_training_progress_plot()
    display(progress_plot)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Train
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Update progress plot
        with progress_plot.batch_update():
            progress_plot.data[0].x = list(range(epoch + 1))
            progress_plot.data[0].y = train_losses
            progress_plot.data[1].x = list(range(epoch + 1))
            progress_plot.data[1].y = val_losses
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader):
    """Evaluate trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            predictions.extend(out.cpu().numpy())
            actuals.extend(batch.y.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Create scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actuals,
        y=predictions,
        mode='markers',
        name='Predictions'
    ))
    
    # Add diagonal line
    diagonal = np.linspace(
        min(actuals.min(), predictions.min()),
        max(actuals.max(), predictions.max()),
        100
    )
    fig.add_trace(go.Scatter(
        x=diagonal,
        y=diagonal,
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(
        title='Predicted vs Actual Length of Stay',
        xaxis_title='Actual Length of Stay',
        yaxis_title='Predicted Length of Stay'
    )
    
    return fig

def explain_predictions(model, test_loader, feature_names):
    """Generate SHAP explanations for model predictions"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Get background data
    background_data = next(iter(test_loader)).to(device)
    
    # Create explainer
    explainer = shap.DeepExplainer(model, background_data)
    
    # Get SHAP values
    test_data = next(iter(test_loader)).to(device)
    shap_values = explainer.shap_values(test_data)
    
    # Create summary plot
    shap.summary_plot(
        shap_values[0],
        test_data.x.cpu().numpy(),
        feature_names=feature_names,
        show=False
    )
    plt.title('Feature Importance (SHAP)')
    
    return plt.gcf()

def main():
    """Main function to run model training and evaluation"""
    # Load data
    data = load_graph_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(data)
    
    # Create model
    input_dim = data['features'].shape[1]
    model = GNNModel(input_dim=input_dim)
    
    # Train model
    trained_model, train_losses, val_losses = train_model_with_visualization(
        model, train_loader, val_loader
    )
    
    # Evaluate model
    eval_fig = evaluate_model(trained_model, test_loader)
    eval_fig.show()
    
    # Save model
    torch.save(trained_model.state_dict(), 'models/trained_gnn.pt')
    
    print("Model training and evaluation complete.")

if __name__ == "__main__":
    main()
