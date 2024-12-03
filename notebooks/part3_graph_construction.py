# Part 3: Graph Construction and Visualization
# This script contains the code for the interactive Jupyter notebook

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from IPython.display import display, HTML
import ipywidgets as widgets
import networkx as nx
from sklearn.preprocessing import StandardScaler
import umap

from src.config.config import ModelConfig
from src.data.dataset import PatientDataset
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

def load_processed_data():
    """Load engineered data and feature importance"""
    df = pd.read_pickle('data/engineered_df.pkl')
    feature_importance = pd.read_pickle('data/feature_importance.pkl')
    return df, feature_importance

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

def create_graph_from_similarity(similarity_matrix, threshold=0.5):
    """Create graph structure from similarity matrix"""
    edges = np.where(similarity_matrix >= threshold)
    edge_index = torch.tensor(np.vstack((edges[0], edges[1])), dtype=torch.long)
    return edge_index

def visualize_graph_sample(df, features, n_samples=100, threshold=0.5):
    """Create interactive visualization of patient graph"""
    # Sample data
    sample_idx = np.random.choice(len(df), n_samples, replace=False)
    sample_features = features[sample_idx]
    sample_los = df.iloc[sample_idx]['length_of_stay'].values
    
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

def create_graph_widget(df, features):
    """Create interactive widget for graph visualization"""
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
    
    def update_graph(n_samples, threshold):
        fig = visualize_graph_sample(df, features, n_samples, threshold)
        fig.show()
    
    return widgets.interact(update_graph, n_samples=sample_slider, threshold=threshold_slider)

def create_3d_visualization(features, los_values, perplexity=30):
    """Create 3D UMAP visualization of patient space"""
    # Create UMAP embedding
    reducer = umap.UMAP(n_components=3, random_state=42)
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

def main():
    """Main function to run graph construction analysis"""
    # Load data
    df, feature_importance = load_processed_data()
    
    # Prepare features
    selected_features = feature_importance['feature'].tolist()
    scaler = StandardScaler()
    features = scaler.fit_transform(df[selected_features])
    
    # Create similarity matrix
    similarity_matrix = create_similarity_matrix(features)
    
    # Visualize similarity matrix
    sim_fig = visualize_similarity_matrix(similarity_matrix)
    sim_fig.show()
    
    # Create interactive graph visualization
    create_graph_widget(df, features)
    
    # Create 3D visualization
    viz_3d_fig = create_3d_visualization(features, df['length_of_stay'].values)
    viz_3d_fig.show()
    
    # Analyze graph properties
    properties = analyze_graph_properties(similarity_matrix)
    display(properties)
    
    # Save graph data for next notebook
    torch.save({
        'features': torch.tensor(features, dtype=torch.float),
        'edge_index': create_graph_from_similarity(similarity_matrix),
        'los_values': torch.tensor(df['length_of_stay'].values, dtype=torch.float)
    }, 'data/graph_data.pt')
    
    print("Graph construction complete. Proceed to Part 4 for model training.")