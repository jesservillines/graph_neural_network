# %% [markdown]
# # Part 1: Data Exploration for Length of Stay Prediction
# 
# This notebook performs comprehensive exploratory data analysis on the patient dataset
# to understand patterns and relationships that will inform our GNN model development.

# %% [markdown]
# ## Setup and Imports

# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path
import sys
from typing import Dict, List

# Add repository root to path
sys.path.append('..')

# Import project modules
from src.config.config import ModelConfig
from src.utils.exploratory_analysis import ExploratoryAnalysis
from src.utils.data_processing import create_patient_graph

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# %% [markdown]
# ## 1. Data Loading and Initial Examination

# %%
# Load configuration
config = ModelConfig()

# Create results directory
results_dir = Path(config.results_dir) / 'part1'
results_dir.mkdir(parents=True, exist_ok=True)

# Load data
logger.info("Loading data...")
df = pd.read_csv(config.data_path)
logger.info(f"Loaded dataset with {len(df)} records and {len(df.columns)} features")

# Initialize exploratory analysis
explorer = ExploratoryAnalysis(df, config.target_col)

# %% [markdown]
# ## 2. Data Overview and Quality Assessment

# %%
# Generate summary statistics
stats = explorer.generate_summary_stats()

print("=== Dataset Overview ===")
print(f"Number of records: {len(df)}")
print(f"Number of features: {len(df.columns)}")
print(f"\nFeature types:\n{df.dtypes.value_counts()}")

print("\n=== Missing Values Summary ===")
missing_stats = stats['missing_values'][stats['missing_values'] > 0]
if len(missing_stats) > 0:
    print("\nFeatures with missing values:")
    for col, count in missing_stats.items():
        print(f"{col}: {count} ({(count/len(df))*100:.2f}%)")
else:
    print("No missing values found in the dataset")

print("\n=== Basic Statistics ===")
print(stats['basic_stats'])

# %% [markdown]
# ## 3. Target Variable Analysis

# %%
# Plot target distribution
target_fig = explorer.plot_target_distribution()
target_fig.write_html(str(results_dir / 'target_distribution.html'))

# Calculate target statistics
target_stats = df[config.target_col].describe()
print("\n=== Target Variable Statistics ===")
print(target_stats)

# %% [markdown]
# ## 4. Feature Analysis

# %%
# Plot correlation heatmap
corr_fig = explorer.plot_feature_correlations(top_n=15)
corr_fig.write_html(str(results_dir / 'feature_correlations.html'))

# Plot numeric feature distributions
num_fig = explorer.plot_numeric_features()
num_fig.write_html(str(results_dir / 'numeric_distributions.html'))

# Plot categorical feature distributions
cat_fig = explorer.plot_categorical_features()
cat_fig.write_html(str(results_dir / 'categorical_distributions.html'))

# %% [markdown]
# ## 5. Feature Importance Analysis

# %%
# Calculate feature importance
importance = explorer.get_feature_importance(method='mutual_info')

print("=== Top 10 Most Important Features ===")
print(importance.head(10))

# Create feature importance plot
fig = go.Figure(go.Bar(
    x=importance.head(15).index,
    y=importance.head(15).values,
    text=importance.head(15).values.round(3),
    textposition='auto',
))

fig.update_layout(
    title='Top 15 Feature Importance Scores',
    xaxis_title='Feature',
    yaxis_title='Importance Score',
    height=500
)

fig.write_html(str(results_dir / 'feature_importance.html'))

# %% [markdown]
# ## 6. Outlier Detection and Analysis

# %%
# Detect outliers
outliers = explorer.detect_outliers(threshold=1.5)

print("=== Outlier Analysis ===")
for feature, stats in outliers.items():
    print(f"\n{feature}:")
    print(f"  Number of outliers: {stats['count']}")
    print(f"  Percentage: {stats['percentage']:.2f}%")

# %% [markdown]
# ## 7. Initial Graph Structure Analysis

# %%
# Prepare features for graph construction
numeric_features = df.select_dtypes(include=[np.number]).columns
numeric_features = numeric_features.drop(config.target_col)
features = df[numeric_features].fillna(df[numeric_features].median())

# Create initial graph structure
logger.info("Creating initial graph structure...")
graph_data, similarity_matrix = create_patient_graph(
    features.values,
    edge_threshold=config.edge_threshold
)

print("\n=== Graph Structure Summary ===")
print(f"Number of nodes: {graph_data.num_nodes}")
print(f"Number of edges: {graph_data.num_edges}")
print(f"Average node degree: {graph_data.num_edges / graph_data.num_nodes:.2f}")

# Visualize similarity distribution
fig = go.Figure(data=go.Histogram(
    x=similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)],
    nbinsx=50
))

fig.update_layout(
    title='Distribution of Patient Similarities',
    xaxis_title='Similarity Score',
    yaxis_title='Count',
    height=400
)

fig.write_html(str(results_dir / 'similarity_distribution.html'))

# %% [markdown]
# ## 8. Key Findings and Recommendations

# %%
print("=== Key Findings ===")

# Feature importance findings
top_features = importance.head(5).index.tolist()
print("\n1. Most influential features for length of stay prediction:")
for idx, feature in enumerate(top_features, 1):
    print(f"   {idx}. {feature} (importance: {importance[feature]:.3f})")

# Outlier findings
significant_outliers = {k: v for k, v in outliers.items() if v['percentage'] > 5}
if significant_outliers:
    print("\n2. Features with significant outliers (>5%):")
    for feature, stats in significant_outliers.items():
        print(f"   - {feature}: {stats['percentage']:.1f}% outliers")

# Missing data findings
missing_significant = stats['missing_values'][stats['missing_values'] > len(df)*0.01]
if len(missing_significant) > 0:
    print("\n3. Features with significant missing values (>1%):")
    for col, count in missing_significant.items():
        print(f"   - {col}: {(count/len(df))*100:.1f}% missing")

# Graph structure insights
print(f"\n4. Graph Structure:")
print(f"   - Average node connectivity: {graph_data.num_edges / graph_data.num_nodes:.2f}")
print(f"   - Edge density: {graph_data.num_edges / (graph_data.num_nodes * (graph_data.num_nodes-1)/2):.3f}")

print("\n=== Recommendations for Model Development ===")
print("""
1. Feature Engineering:
   - Focus on top important features identified
   - Create interaction terms for highly correlated features
   - Consider dimensionality reduction for less important features

2. Data Preprocessing:
   - Handle outliers in significant features
   - Implement robust scaling for numeric features
   - Consider advanced imputation methods for missing values

3. Graph Construction:
   - Fine-tune similarity threshold based on distribution
   - Consider weighted edges based on similarity scores
   - Explore different similarity metrics for edge creation

4. Model Architecture:
   - Design attention mechanisms focusing on important features
   - Implement skip connections for direct feature influence
   - Consider feature importance in node embedding generation
""")

# Save key findings to file
with open(results_dir / 'key_findings.txt', 'w') as f:
    f.write("Key Findings from Exploratory Analysis\n")
    f.write("=====================================\n\n")
    f.write("1. Feature Importance:\n")
    for idx, feature in enumerate(importance.head(10).items(), 1):
        f.write(f"   {idx}. {feature[0]}: {feature[1]:.3f}\n")
    
    f.write("\n2. Data Quality:\n")
    f.write(f"   - Total records: {len(df)}\n")
    f.write(f"   - Missing data features: {len(missing_significant)}\n")
    f.write(f"   - Features with significant outliers: {len(significant_outliers)}\n")
    
    f.write("\n3. Graph Structure:\n")
    f.write(f"   - Nodes: {graph_data.num_nodes}\n")
    f.write(f"   - Edges: {graph_data.num_edges}\n")
    f.write(f"   - Average degree: {graph_data.num_edges / graph_data.num_nodes:.2f}\n")