import torch
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import wandb
import os
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

class ModelTrainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        scheduler=None,
        patience=7,
        checkpoint_dir='checkpoints'
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.early_stopping = EarlyStopping(patience=patience)
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_model_path = None
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out, batch.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(out, batch.y)
                
                total_loss += loss.item()
                predictions.extend(out.cpu().numpy())
                actuals.extend(batch.y.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        metrics = {
            'loss': total_loss / len(loader),
            'mse': mean_squared_error(actuals, predictions),
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'mae': mean_absolute_error(actuals, predictions),
            'r2': r2_score(actuals, predictions)
        }
        
        return metrics
    
    def save_checkpoint(self, epoch, val_loss):
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'model_epoch{epoch}_loss{val_loss:.4f}.pt'
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        
        return checkpoint_path
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs,
        use_wandb=False
    ):
        """Train the model with early stopping and checkpointing"""
        
        for epoch in tqdm(range(num_epochs), desc="Training"):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Logging
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_rmse': val_metrics['rmse'],
                    'val_mae': val_metrics['mae'],
                    'val_r2': val_metrics['r2']
                })
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_path = self.save_checkpoint(epoch, val_loss)
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        return self.best_model_path

    def predict(self, loader):
        """Generate predictions for a data loader"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                predictions.extend(out.cpu().numpy())
        
        return np.array(predictions)
