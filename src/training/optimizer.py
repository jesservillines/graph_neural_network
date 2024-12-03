import optuna
import torch
from torch_geometric.loader import DataLoader
from ..models.gnn_model import GNNModel
from .trainer import ModelTrainer

class HyperparameterOptimizer:
    def __init__(
        self,
        dataset,
        cv_split,
        device,
        n_trials=100,
        timeout=None
    ):
        self.dataset = dataset
        self.cv_split = cv_split
        self.device = device
        self.n_trials = n_trials
        self.timeout = timeout
    
    def objective(self, trial):
        # Hyperparameters to optimize
        params = {
            'hidden_dim': trial.suggest_int('hidden_dim', 32, 256),
            'num_heads': trial.suggest_int('num_heads', 2, 8),
            'dropout': trial.suggest_float('dropout', 0.1, 0.7),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_int('batch_size', 16, 128),
            'edge_threshold': trial.suggest_float('edge_threshold', 0.3, 0.8)
        }
        
        # Cross-validation scores
        cv_scores = []
        
        # Perform cross-validation
        for fold in range(self.cv_split.n_splits):
            # Get fold data
            split = self.cv_split.get_split(fold)
            
            # Create data loaders
            train_loader = DataLoader(
                self.dataset[split['train']],
                batch_size=params['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                self.dataset[split['val']],
                batch_size=params['batch_size']
            )
            
            # Initialize model and training components
            model = GNNModel(
                input_dim=self.dataset.X.shape[1],
                hidden_dim=params['hidden_dim'],
                num_heads=params['num_heads'],
                dropout=params['dropout']
            ).to(self.device)
            
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            
            # Initialize trainer
            trainer = ModelTrainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device,
                patience=10
            )
            
            # Train model
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=50
            )
            
            # Get validation score
            val_metrics = trainer.evaluate(val_loader)
            cv_scores.append(val_metrics['rmse'])
            
            # Report intermediate value
            trial.report(val_metrics['rmse'], fold)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return sum(cv_scores) / len(cv_scores)
    
    def optimize(self):
        """Run hyperparameter optimization"""
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        return study.best_params, study.best_value
    
    def plot_optimization_results(self, study):
        """Plot optimization results using optuna visualization"""
        try:
            from optuna.visualization import plot_optimization_history
            from optuna.visualization import plot_param_importances
            
            # Plot optimization history
            fig1 = plot_optimization_history(study)
            fig1.show()
            
            # Plot parameter importances
            fig2 = plot_param_importances(study)
            fig2.show()
            
        except ImportError:
            print("Optuna visualization requires plotly to be installed")