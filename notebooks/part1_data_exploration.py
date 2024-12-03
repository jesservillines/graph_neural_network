#%% md
# Part 1: Data Loading and Exploration

This notebook covers the initial data loading and exploratory data analysis for the Length of Stay prediction project.
#%%
# Import required libraries
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML
import ipywidgets as widgets

from src.config.config import ModelConfig
#%% md
## 1. Load and Display Data
#%%
# Load configuration
config = ModelConfig()

# Load data
df = pd.read_csv(config.data_path)

print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst few rows:")
display(df.head())
#%% md
## 2. Data Quality Check
#%%
# Display data info
data_info = pd.DataFrame({
    'dtype': df.dtypes,
    'non_null': df.count(),
    'null_count': df.isnull().sum(),
    'unique_values': df.nunique(),
    'memory_usage': df.memory_usage(deep=True)
})

print("Data Quality Summary:")
display(data_info)
#%% md
## 3. Interactive Data Exploration
#%%
def create_distribution_plot(df, column):
    """Create appropriate distribution plot based on data type"""
    if df[column].dtype in ['int64', 'float64']:
        # Numerical data
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[column],
            name=column,
            nbinsx=30
        ))
        fig.add_trace(go.Histogram(
            x=df[column],
            histnorm='probability density',
            name='density'
        ))
        fig.update_layout(
            title=f'Distribution of {column}',
            barmode='overlay'
        )
    else:
        # Categorical data
        value_counts = df[column].value_counts()
        fig = go.Figure(go.Bar(
            x=value_counts.index,
            y=value_counts.values,
            text=value_counts.values,
            textposition='auto'
        ))
        fig.update_layout(title=f'Value Counts for {column}')
    
    return fig

def create_summary_widget(df):
    """Create interactive widget for data exploration"""
    columns = list(df.columns)
    dropdown = widgets.Dropdown(
        options=columns,
        description='Select Column:',
        style={'description_width': 'initial'}
    )
    
    def show_column_analysis(column):
        # Basic statistics
        print(f"\nSummary Statistics for {column}:")
        display(df[column].describe())
        
        # Distribution plot
        fig = create_distribution_plot(df, column)
        fig.show()
        
        # Relationship with target if numerical
        if column != 'length_of_stay' and df[column].dtype in ['int64', 'float64']:
            correlation = df[column].corr(df['length_of_stay'])
            print(f"\nCorrelation with length_of_stay: {correlation:.3f}")
            
            fig2 = px.scatter(
                df, 
                x=column, 
                y='length_of_stay',
                trendline='ols',
                title=f'{column} vs Length of Stay'
            )
            fig2.show()
    
    return widgets.interact(show_column_analysis, column=dropdown)

# Create and display the widget
create_summary_widget(df)
#%% md
## 4. Target Variable Analysis
#%%
# Analyze length of stay distribution
fig = go.Figure()

# Add histogram
fig.add_trace(go.Histogram(
    x=df['length_of_stay'],
    name='Distribution',
    nbinsx=50
))

# Add box plot
fig.add_trace(go.Box(
    x=df['length_of_stay'],
    name='Box Plot',
    boxpoints='all',
    jitter=0.3,
    pointpos=-1.8
))

fig.update_layout(
    title='Length of Stay Distribution',
    xaxis_title='Length of Stay (days)',
    yaxis_title='Count',
    barmode='overlay'
)

fig.show()

# Print summary statistics
print("\nLength of Stay Summary Statistics:")
display(df['length_of_stay'].describe())
#%% md
## 5. Save Initial Analysis

Save the processed dataframe and analysis results for use in subsequent notebooks.
#%%
# Save processed dataframe
df.to_pickle('data/initial_df.pkl')

# Save basic statistics
data_info.to_csv('data/data_quality_summary.csv')

print("Initial data analysis complete. Proceed to Part 2 for feature engineering.")
