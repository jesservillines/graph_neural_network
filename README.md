# Graph Neural Network for Length of Stay Prediction

This project implements a Graph Neural Network (GNN) to predict patient length of stay using PyTorch Geometric.

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── gnn_model.py     # GNN model architecture
│   │   └── loss.py          # Custom loss functions
│   └── utils/
│       ├── data_processing.py    # Data processing utilities
│       ├── feature_engineering.py # Feature engineering functions
│       ├── feature_selection.py   # Feature selection methods
│       └── visualization.py       # Visualization utilities
├── main.py                  # Main training script
└── README.md               # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset file (`model_df_12_02_24.csv`) in the project root.
2. Run the main script:
```bash
python main.py
```

## Features

- Graph-based patient similarity modeling
- Advanced feature engineering
- SHAP-based feature selection
- 3D visualization of patient clusters
- Custom loss function for length of stay prediction
- Multi-head attention mechanisms

## TODO

- [ ] Implement data loading and cross-validation
- [ ] Add training pipeline with early stopping
- [ ] Implement model evaluation metrics
- [ ] Add hyperparameter optimization
- [ ] Enhance visualization capabilities
- [ ] Add model interpretability functions