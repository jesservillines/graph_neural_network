import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import scipy.stats as stats
import logging

logger = logging.getLogger(__name__)

class ExploratoryAnalysis:
    def __init__(self, df: pd.DataFrame, target_col: str):
        """
        Initialize the exploratory analysis class.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
        """
        self.df = df.copy()
        self.target_col = target_col
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = df.select_dtypes(include=['object']).columns
        
    def generate_summary_stats(self) -> Dict:
        """Generate comprehensive summary statistics."""
        stats_dict = {
            'basic_stats': self.df.describe(),
            'missing_values': self.df.isnull().sum(),
            'unique_counts': self.df.nunique(),
            'data_types': self.df.dtypes,
            'skewness': self.df[self.numeric_cols].skew(),
            'kurtosis': self.df[self.numeric_cols].kurtosis()
        }
        
        # Add correlation with target for numeric features
        target_correlations = {}
        for col in self.numeric_cols:
            if col != self.target_col:
                correlation = stats.spearmanr(
                    self.df[col].fillna(self.df[col].median()),
                    self.df[self.target_col]
                )[0]
                target_correlations[col] = correlation
        
        stats_dict['target_correlations'] = target_correlations
        
        return stats_dict
    
    def plot_target_distribution(self) -> go.Figure:
        """Plot the distribution of the target variable."""
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=['Distribution', 'Box Plot'])
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=self.df[self.target_col], name='Distribution'),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=self.df[self.target_col], name='Box Plot'),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f'Distribution of {self.target_col}',
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_feature_correlations(self, top_n: int = 15) -> go.Figure:
        """Plot correlation heatmap of numeric features."""
        # Calculate correlations
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Get top correlated features with target
        target_corrs = abs(corr_matrix[self.target_col]).sort_values(ascending=False)
        top_features = target_corrs.head(top_n).index
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.loc[top_features, top_features],
            x=top_features,
            y=top_features,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title='Feature Correlations Heatmap',
            height=600,
            width=800
        )
        
        return fig
    
    def plot_numeric_features(self, features: Optional[List[str]] = None) -> go.Figure:
        """Create distribution plots for numeric features."""
        if features is None:
            features = [col for col in self.numeric_cols if col != self.target_col]
            if len(features) > 10:  # Limit to top 10 correlated features
                corr_series = abs(self.df[features].corrwith(self.df[self.target_col]))
                features = corr_series.nlargest(10).index.tolist()
        
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + 1) // 2
        
        fig = make_subplots(rows=n_rows, cols=n_cols,
                           subplot_titles=features)
        
        for idx, feature in enumerate(features, 1):
            row = (idx - 1) // n_cols + 1
            col = (idx - 1) % n_cols + 1
            
            # Add histogram
            fig.add_trace(
                go.Histogram(x=self.df[feature], name=feature),
                row=row, col=col
            )
            
            # Update layout for each subplot
            fig.update_xaxes(title_text=feature, row=row, col=col)
            fig.update_yaxes(title_text='Count', row=row, col=col)
        
        fig.update_layout(
            height=300 * n_rows,
            width=1000,
            showlegend=False,
            title_text='Numeric Feature Distributions'
        )
        
        return fig
    
    def plot_categorical_features(self, features: Optional[List[str]] = None) -> go.Figure:
        """Create bar plots for categorical features."""
        if features is None:
            features = self.categorical_cols
            if len(features) > 10:  # Limit to 10 features
                features = features[:10]
        
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + 1) // 2
        
        fig = make_subplots(rows=n_rows, cols=n_cols,
                           subplot_titles=features)
        
        for idx, feature in enumerate(features, 1):
            row = (idx - 1) // n_cols + 1
            col = (idx - 1) % n_cols + 1
            
            # Calculate value counts
            value_counts = self.df[feature].value_counts()
            
            # Add bar chart
            fig.add_trace(
                go.Bar(x=value_counts.index, y=value_counts.values, name=feature),
                row=row, col=col
            )
            
            # Update layout for each subplot
            fig.update_xaxes(title_text=feature, row=row, col=col)
            fig.update_yaxes(title_text='Count', row=row, col=col)
        
        fig.update_layout(
            height=300 * n_rows,
            width=1000,
            showlegend=False,
            title_text='Categorical Feature Distributions'
        )
        
        return fig
    
    def plot_target_vs_feature(self, feature: str) -> go.Figure:
        """Create a plot of target vs a specific feature."""
        if feature in self.numeric_cols:
            fig = px.scatter(self.df, x=feature, y=self.target_col,
                           trendline="ols")
            fig.update_layout(title=f'{self.target_col} vs {feature}')
        else:
            # For categorical features, create box plots
            fig = go.Figure()
            for category in self.df[feature].unique():
                fig.add_trace(go.Box(
                    y=self.df[self.df[feature] == category][self.target_col],
                    name=str(category)
                ))
            fig.update_layout(
                title=f'{self.target_col} Distribution by {feature}',
                xaxis_title=feature,
                yaxis_title=self.target_col
            )
        
        return fig
    
    def detect_outliers(self, threshold: float = 1.5) -> Dict[str, pd.Series]:
        """
        Detect outliers in numeric features using IQR method.
        
        Args:
            threshold: IQR multiplier for outlier detection
        
        Returns:
            Dictionary with outlier counts and indices for each feature
        """
        outliers_dict = {}
        
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = self.df[
                (self.df[col] < lower_bound) | 
                (self.df[col] > upper_bound)
            ]
            
            if len(outliers) > 0:
                outliers_dict[col] = {
                    'count': len(outliers),
                    'indices': outliers.index.tolist(),
                    'percentage': (len(outliers) / len(self.df)) * 100
                }
        
        return outliers_dict
    
    def get_feature_importance(self, method: str = 'correlation') -> pd.Series:
        """
        Calculate feature importance scores.
        
        Args:
            method: Method to use for importance calculation ('correlation' or 'mutual_info')
        
        Returns:
            Series with feature importance scores
        """
        if method == 'correlation':
            importance = abs(self.df[self.numeric_cols].corrwith(self.df[self.target_col]))
        else:
            # For categorical features, use mutual information
            from sklearn.feature_selection import mutual_info_regression
            importance = pd.Series(
                mutual_info_regression(
                    self.df.drop(columns=[self.target_col]),
                    self.df[self.target_col]
                ),
                index=self.df.drop(columns=[self.target_col]).columns
            )
        
        return importance.sort_values(ascending=False)