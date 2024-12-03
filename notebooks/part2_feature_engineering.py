# Part 2: Feature Engineering and Selection
# This script contains the code for the interactive Jupyter notebook

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML
import ipywidgets as widgets
import seaborn as sns
import matplotlib.pyplot as plt

from src.config.config import ModelConfig
from src.utils.feature_engineering import engineer_features
from src.utils.feature_selection import select_features_shap

def load_initial_data():
    """Load data from Part 1"""
    df = pd.read_pickle('data/initial_df.pkl')
    print("Loaded initial dataset with shape:", df.shape)
    return df

def apply_feature_engineering(df):
    """Apply feature engineering and display new features"""
    df_engineered = engineer_features(df)
    
    # Show new features
    new_features = set(df_engineered.columns) - set(df.columns)
    print("\nNewly created features:")
    display(df_engineered[list(new_features)].head())
    
    return df_engineered

def plot_correlation_matrix(df, min_correlation=0.0):
    """Create interactive correlation matrix plot"""
    # Calculate correlation matrix for numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr = df[numeric_cols].corr()
    
    # Apply threshold mask
    mask = np.abs(corr) >= min_correlation
    corr_filtered = corr * mask
    
    # Create heatmap
    fig = px.imshow(
        corr_filtered,
        labels=dict(color="Correlation"),
        title=f"Feature Correlation Matrix (|correlation| ≥ {min_correlation})",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    
    fig.update_layout(
        height=800,
        width=800,
        title_x=0.5
    )
    
    return fig

def create_correlation_widget(df):
    """Create interactive widget for correlation analysis"""
    correlation_slider = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=1.0,
        step=0.05,
        description='Min Correlation:'
    )
    
    def update_correlation_plot(min_correlation):
        fig = plot_correlation_matrix(df, min_correlation)
        fig.show()
    
    return widgets.interact(update_correlation_plot, min_correlation=correlation_slider)

def analyze_feature_importance(df_engineered):
    """Analyze and visualize feature importance"""
    # Get feature importance using SHAP
    feature_importance = select_features_shap(df_engineered, 'length_of_stay')
    
    # Create bar plot
    fig = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance (SHAP values)',
        labels={'importance': 'SHAP Value', 'feature': 'Feature'}
    )
    
    fig.update_layout(
        height=800,
        yaxis={'categoryorder': 'total ascending'},
        title_x=0.5
    )
    
    return fig, feature_importance

def analyze_feature_interactions(df, feature_importance):
    """Analyze interactions between top features"""
    top_features = feature_importance.head(5)['feature'].tolist()
    
    fig = go.Figure()
    
    for feature in top_features:
        if df[feature].dtype in ['int64', 'float64']:
            fig.add_trace(go.Scatter(
                x=df[feature],
                y=df['length_of_stay'],
                mode='markers',
                name=feature,
                opacity=0.6
            ))
    
    fig.update_layout(
        title='Top Features vs Length of Stay',
        xaxis_title='Feature Value',
        yaxis_title='Length of Stay',
        height=600
    )
    
    return fig

def create_feature_distribution_widget(df, feature_importance):
    """Create widget to explore feature distributions"""
    top_features = feature_importance['feature'].tolist()
    
    feature_dropdown = widgets.Dropdown(
        options=top_features,
        description='Feature:',
        style={'description_width': 'initial'}
    )
    
    def plot_feature_distribution(feature):
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=df[feature],
            name='Distribution',
            nbinsx=30
        ))
        
        # Add kernel density estimation
        if df[feature].dtype in ['int64', 'float64']:
            from scipy import stats
            kde_x = np.linspace(df[feature].min(), df[feature].max(), 100)
            kde = stats.gaussian_kde(df[feature].dropna())
            fig.add_trace(go.Scatter(
                x=kde_x,
                y=kde(kde_x) * len(df[feature]) * (df[feature].max() - df[feature].min()) / 30,
                name='KDE',
                line=dict(color='red')
            ))
        
        fig.update_layout(
            title=f'Distribution of {feature}',
            xaxis_title=feature,
            yaxis_title='Count',
            height=500
        )
        
        fig.show()
        
        # Show summary statistics
        print(f"\nSummary statistics for {feature}:")
        display(df[feature].describe())
    
    return widgets.interact(plot_feature_distribution, feature=feature_dropdown)

def main():
    """Main function to run feature engineering analysis"""
    # Load initial data
    df = load_initial_data()
    
    # Apply feature engineering
    df_engineered = apply_feature_engineering(df)
    
    # Create correlation analysis widget
    create_correlation_widget(df_engineered)
    
    # Analyze feature importance
    importance_fig, feature_importance = analyze_feature_importance(df_engineered)
    importance_fig.show()
    
    # Analyze feature interactions
    interaction_fig = analyze_feature_interactions(df_engineered, feature_importance)
    interaction_fig.show()
    
    # Create feature distribution widget
    create_feature_distribution_widget(df_engineered, feature_importance)
    
    # Save results for next notebook
    df_engineered.to_pickle('data/engineered_df.pkl')
    feature_importance.to_pickle('data/feature_importance.pkl')
    
    print("Feature engineering complete. Proceed to Part 3 for graph construction.")
