from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class ModelConfig:
    # Model Architecture
    hidden_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.5
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    patience: int = 7
    min_delta: float = 0.001
    
    # Graph Construction
    edge_threshold: float = 0.5
    
    # Data
    data_path: str = 'model_df_12_02_24.csv'
    target_col: str = 'length_of_stay'
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    
    # Paths
    checkpoint_dir: str = 'checkpoints'
    results_dir: str = 'results'
    
    # Device
    device: str = 'cuda'
    
    # Feature Selection
    n_features: int = 20
    
    # Early Stopping
    early_stopping_patience: int = 7
    
    # Optimization
    n_trials: int = 100
    optimization_timeout: Optional[int] = None
    
    # Evaluation
    metrics: Tuple[str, ...] = ('mse', 'rmse', 'mae', 'r2')
    n_background_samples: int = 100