import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import torch
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data_path = Path(config.data_path)
        self.random_state = config.random_state
        self.scaler = StandardScaler()
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV file."""
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded {len(df)} records from {self.data_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform initial feature preprocessing."""
        df = df.copy()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
            
        # Fill categorical missing values with mode
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        # Convert dates to datetime features
        date_columns = df.columns[df.columns.str.contains('date', case=False)]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            except Exception as e:
                logger.warning(f"Could not process date column {col}: {str(e)}")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using one-hot encoding."""
        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # One-hot encode categorical variables
        for col in categorical_cols:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
            
        return df
    
    def scale_numerical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """Scale numerical features using StandardScaler."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop(self.config.target_col)
        
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df, self.scaler
    
    def generate_feature_stats(self, df: pd.DataFrame) -> Dict:
        """Generate statistical summaries of features."""
        stats = {
            'basic_stats': df.describe(),
            'missing_values': df.isnull().sum(),
            'unique_values': df.nunique(),
            'feature_types': df.dtypes,
            'correlations': df.corr()
        }
        
        return stats
    
    def get_processed_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Load and process data, returning both processed data and statistics."""
        # Load raw data
        df = self.load_raw_data()
        
        # Generate initial statistics
        initial_stats = self.generate_feature_stats(df)
        
        # Preprocess features
        df = self.preprocess_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Scale numerical features
        df, _ = self.scale_numerical_features(df)
        
        # Generate final statistics
        final_stats = self.generate_feature_stats(df)
        
        stats = {
            'initial': initial_stats,
            'final': final_stats,
            'feature_names': list(df.columns),
            'n_features': len(df.columns) - 1  # excluding target
        }
        
        return df, stats
