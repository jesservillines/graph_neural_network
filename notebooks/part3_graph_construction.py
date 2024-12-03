# Part 3: Graph Construction and Visualization

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import ipywidgets as widgets
import networkx as nx
from sklearn.preprocessing import StandardScaler
import umap
from IPython.display import display, HTML

from src.config.config import ModelConfig
from src.data.dataset import PatientDataset
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

def load_processed_data():
    """Load engineered data and feature importance"""
    df = pd.read_pickle('data/engineered_df.pkl')
    feature_importance = pd.read_pickle('data/feature_importance.pkl')
    
    print("Loaded data shape:", df.shape)
    print("\nTop 10 important features:")
    display(feature_importance.head(10))
    
    return df, feature_importance

def prepare_features(df, feature_importance):
    """Prepare feature matrix"""
    selected_features = feature_importance['feature'].tolist()
    scaler = StandardScaler()
    features = scaler.fit_transform(df[selected_features])
    
    print("Feature matrix shape:", features.shape)
    print("\nFeature statistics:")
    feature_stats = pd.DataFrame(
        {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0)
        },
        index=selected_features
    )
    display(feature_stats)
    
    return features, selected_features

def create_similarity_matrix(features, threshold=0.5):
    """Create similarity matrix from features"""
    similarity = cosine_similarity(features)
    similarity[similarity < threshold] = 0
    return similarity

def visualize_similarity_matrix(similarity_matrix):
    """Create heatmap of similarity matrix"""
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title='Patient Similarity Matrix',
        width=800,
        height=800
    )
    
    return fig

def visualize_graph_sample(features, los_values, n_samples=100, threshold=0.5):
    """Create interactive visualization of patient graph"""
    # Sample data
    sample_idx = np.random.choice(len(features), n_samples, replace=False)
    sample_features = features[sample_idx]
    sample_los = los_values[sample_idx]
    
    # Create similarity matrix and graph
    similarity = create_similarity_matrix(sample_features, threshold)
    G = nx.from_numpy_array(similarity)
    
    # Create layout
    pos = nx.spring_layout(G, k=1/np.sqrt(n_samples))
    
    # Create edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            color=sample_los,
            size=10
        ),
        text=[f'LOS: {los:.1f}' for los in sample_los]
    ))
    
    fig.update_layout(
        title=f'Patient Similarity Graph (n={n_samples}, threshold={threshold:.2f})',
        showlegend=False,
        hovermode='closest',
        width=800,
        height=800
    )
    
    return fig

def create_3d_visualization(features, los_values, n_neighbors=15, min_dist=0.1):
    """Create 3D UMAP visualization of patient space"""
    # Create UMAP embedding
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    embedding = reducer.fit_transform(features)
    
    # Create figure
    fig = go.Figure(data=[go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=los_values,
            colorscale='Viridis',
            opacity=0.8,
            showscale=True
        ),
        text=[f'LOS: {los:.1f}' for los in los_values],
        hoverinfo='text'
    )])
    
    fig.update_layout(
        title='3D Patient Space Visualization',
        width=1000,
        height=800
    )
    
    return fig

def analyze_graph_properties(similarity_matrix, threshold=0.5):
    """Analyze graph properties"""
    G = nx.from_numpy_array(similarity_matrix >= threshold)
    
    properties = {
        'Number of nodes': G.number_of_nodes(),
        'Number of edges': G.number_of_edges(),
        'Average degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'Average clustering coefficient': nx.average_clustering(G),
        'Number of connected components': nx.number_connected_components(G),
        'Graph density': nx.density(G)
    }
    
    return pd.Series(properties)

def create_interactive_graph_widgets(features, los_values):
    """Create interactive widgets for graph visualization"""
    sample_slider = widgets.IntSlider(
        value=100,
        min=50,
        max=500,
        step=50,
        description='Sample Size:'
    )
    
    threshold_slider = widgets.FloatSlider(
        value=0.5,
        min=0.1,
        max=0.9,
        step=0.1,
        description='Similarity Threshold:'
    )
    
    return widgets.interact(
        lambda n, t: visualize_graph_sample(features, los_values, n, t).show(),
        n_samples=sample_slider,
        threshold=threshold_slider
    )

def create_interactive_umap_widgets(features, los_values):
    """Create interactive widgets for UMAP visualization"""
    neighbors_slider = widgets.IntSlider(
        value=15,
        min=5,
        max=50,
        step=5,
        description='n_neighbors:'
    )
    
    min_dist_slider = widgets.FloatSlider(
        value=0.1,
        min=0.0,
        max=0.9,
        step=0.1,
        description='min_dist:'
    )
    
    return widgets.interact(
        lambda n, d: create_3d_visualization(features, los_values, n, d).show(),
        n_neighbors=neighbors_slider,
        min_dist=min_dist_slider
    )

def main():
    """Main function to run graph construction analysis"""
    # Load data
    df, feature_importance = load_processed_data()
    
    # Prepare features
    features, selected_features = prepare_features(df, feature_importance)
    
    # Create similarity matrix
    similarity_matrix = create_similarity_matrix(features)
    
    # Visualize similarity matrix
    sim_fig = visualize_similarity_matrix(similarity_matrix)
    sim_fig.show()
    
    # Create interactive graph visualization
    create_interactive_graph_widgets(features, df['length_of_stay'].values)
    
    # Create interactive UMAP visualization
    create_interactive_umap_widgets(features, df['length_of_stay'].values)
    
    # Analyze graph properties
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    properties_df = pd.DataFrame({
        f'threshold_{t}': analyze_graph_properties(similarity_matrix, t)
        for t in thresholds
    })
    display(properties_df)
    
    # Save graph data for PyTorch Geometric
    edge_index = torch.tensor(
        np.where(similarity_matrix >= 0.5),
        dtype=torch.long
    )
    
    graph_data = {
        'features': torch.tensor(features, dtype=torch.float),
        'edge_index': edge_index,
        'los_values': torch.tensor(df['length_of_stay'].values, dtype=torch.float)
    }
    
    torch.save(graph_data, 'data/graph_data.pt')
    print("\nGraph construction complete. Graph data saved. Proceed to Part 4 for model training.")

if __name__ == "__main__":
    main()