# Graph Neural Network for Length of Stay Prediction

## Overview
This repository implements a Graph Neural Network (GNN) to predict patient length of stay using PyTorch Geometric. The implementation features advanced feature selection, interactive visualizations, and model interpretability.

## Project Structure
```
├── src/
│   ├── config/             # Configuration settings
│   ├── data/              # Dataset handling
│   ├── models/            # GNN architecture
│   ├── training/          # Training pipeline
│   ├── evaluation/        # Model evaluation
│   └── utils/             # Utility functions
├── notebooks/             # Interactive Jupyter notebooks
│   ├── part1_data_exploration.py
│   ├── part2_feature_engineering.py
│   ├── part3_graph_construction.py
│   └── part4_model_training.py
├── main.py               # Main training script
├── run_notebooks.py      # Script to run all notebooks
└── requirements.txt      # Project dependencies
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/jesservillines/graph_neural_network.git
cd graph_neural_network
```

2. Create a conda environment (recommended):
```bash
conda create -n gnn_env python=3.8
conda activate gnn_env
```

3. Install PyTorch with CUDA support (for dual RTX 4080s):
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

4. Install PyTorch Geometric:
```bash
conda install pyg -c pyg
```

5. Install remaining requirements:
```bash
pip install -r requirements.txt
```

## Running the Project

### Option 1: Interactive Notebooks (Recommended for Exploration)

1. Start JupyterLab:
```bash
jupyter lab
```

2. Navigate to the `notebooks/` directory and run the notebooks in sequence:
   - `part1_data_exploration.py`: Data loading and initial analysis
   - `part2_feature_engineering.py`: Feature engineering and selection
   - `part3_graph_construction.py`: Graph construction and visualization
   - `part4_model_training.py`: Model training and evaluation

Alternatively, run all notebooks automatically:
```bash
python run_notebooks.py
```

### Option 2: Direct Training

Run the main training script:
```bash
python main.py
```

## Data Preparation

1. Place your dataset file (`model_df_12_02_24.csv`) in the project root directory.

2. Required columns:
   - `length_of_stay` (target variable)
   - `unique_id` (patient identifier)
   - All predictor variables as listed in the documentation

## Configuration

Modify settings in `src/config/config.py`:
- Model architecture parameters
- Training parameters
- Graph construction parameters
- Evaluation settings

## Model Training

The model training process includes:
1. Data preprocessing and feature engineering
2. Graph construction based on patient similarities
3. GNN training with early stopping
4. Model evaluation and interpretability analysis

## Outputs

The project generates several outputs:
- `data/`: Processed data and intermediate results
- `models/`: Trained model checkpoints
- `results/`: Evaluation metrics and visualizations
- Notebook outputs with interactive visualizations

## Monitoring Training

1. Real-time training progress in notebooks:
   - Loss curves
   - Performance metrics
   - Interactive visualizations

2. Logs and checkpoints:
   - Training logs in `logs/`
   - Model checkpoints in `models/`
   - Evaluation results in `results/`

## Customization

1. Model Architecture:
   - Modify `src/models/gnn_model.py`
   - Adjust hyperparameters in config

2. Feature Engineering:
   - Add custom features in `src/utils/feature_engineering.py`
   - Modify feature selection in `src/utils/feature_selection.py`

3. Graph Construction:
   - Adjust similarity metrics in `src/data/dataset.py`
   - Modify edge construction rules

## Troubleshooting

Common issues and solutions:

1. CUDA/GPU Issues:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

2. Memory Issues:
   - Reduce batch size in config
   - Reduce model size
   - Adjust graph construction threshold

3. Package Dependencies:
```bash
# Verify installations
python -c "import torch_geometric; import torch; import plotly"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Include your license information here]

## Citation

If you use this code in your research, please cite:
```
@software{gnn_los_prediction,
    title={Graph Neural Network for Length of Stay Prediction},
    author={[Your Name]},
    year={2024},
    url={https://github.com/jesservillines/graph_neural_network}
}
```

## Contact

Contact the Author, Jesse Villines: https://www.linkedin.com/in/jesse-villines/
