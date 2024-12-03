import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from src.models.gnn_model import GNNModel
from src.models.loss import WeightedMSELoss
from src.utils.feature_engineering import engineer_features
from src.utils.feature_selection import select_features_shap
from src.utils.data_processing import create_patient_graph
from src.utils.visualization import create_3d_visualization

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    df = pd.read_csv('notebooks/model_df_12_02_24.csv')
    
    # Engineer features
    df = engineer_features(df)
    
    # Select features
    selected_features = select_features_shap(df, 'length_of_stay')
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[selected_features])
    
    # Create graph
    graph_data, similarity_matrix = create_patient_graph(X)
    
    # Initialize model
    model = GNNModel(
        input_dim=len(selected_features),
        hidden_dim=64,
        num_heads=4
    ).to(device)
    
    # Initialize loss and optimizer
    criterion = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # TODO: Implement data loading and training loop
    # TODO: Add model evaluation
    # TODO: Add visualization
    
if __name__ == "__main__":
    main()