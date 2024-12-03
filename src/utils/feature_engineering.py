import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for length of stay prediction."""
    
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df.copy()
        self.target_col = target_col
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = df.select_dtypes(include=['object']).columns
        
    def create_medical_complexity_features(self) -> None:
        """Create features related to medical complexity and comorbidity interactions."""
        # Base comorbidity score
        comorbidity_cols = [
            'diabetes', 'hypertension', 'heart_disease',
            'neurological_disorder', 'psychiatric_disorder',
            'respiratory_disease', 'renal_disease', 'liver_disease'
        ]
        
        # Create weighted comorbidity score
        weights = {
            'diabetes': 1.2,
            'heart_disease': 1.5,
            'neurological_disorder': 1.8,
            'respiratory_disease': 1.3,
            'renal_disease': 1.4,
            'liver_disease': 1.6,
            'psychiatric_disorder': 1.1,
            'hypertension': 1.1
        }
        
        # Calculate weighted score
        self.df['weighted_comorbidity_score'] = sum(
            self.df[col] * weights.get(col, 1.0)
            for col in comorbidity_cols
        )
        
        # Create interaction terms for high-impact conditions
        high_impact_pairs = [
            ('heart_disease', 'respiratory_disease'),
            ('diabetes', 'renal_disease'),
            ('neurological_disorder', 'psychiatric_disorder')
        ]
        
        for cond1, cond2 in high_impact_pairs:
            self.df[f'{cond1}_{cond2}_interaction'] = self.df[cond1] * self.df[cond2]
        
        # Age-based risk factors
        self.df['age_risk_factor'] = self.df['age'].apply(
            lambda x: 1 + 0.1 * max(0, (x - 65) / 10)
        )
        
        # Complex medical index
        self.df['complex_medical_index'] = (
            self.df['weighted_comorbidity_score'] * 
            self.df['age_risk_factor']
        )
        
        logger.info("Created medical complexity features")
    
    def create_temporal_features(self) -> None:
        """Create features related to temporal patterns and admission timing."""
        # Extract admission timing features
        admission_date_col = [col for col in self.df.columns if 'admission' in col.lower() and 'date' in col.lower()][0]
        self.df['admission_datetime'] = pd.to_datetime(self.df[admission_date_col])
        
        self.df['admission_hour'] = self.df['admission_datetime'].dt.hour
        self.df['admission_day'] = self.df['admission_datetime'].dt.day
        self.df['admission_month'] = self.df['admission_datetime'].dt.month
        self.df['admission_dayofweek'] = self.df['admission_datetime'].dt.dayofweek
        
        # Create cyclical features for temporal variables
        self.df['admission_hour_sin'] = np.sin(2 * np.pi * self.df['admission_hour'] / 24)
        self.df['admission_hour_cos'] = np.cos(2 * np.pi * self.df['admission_hour'] / 24)
        self.df['admission_month_sin'] = np.sin(2 * np.pi * self.df['admission_month'] / 12)
        self.df['admission_month_cos'] = np.cos(2 * np.pi * self.df['admission_month'] / 12)
        
        # Weekend/holiday indicator
        self.df['is_weekend'] = self.df['admission_dayofweek'].isin([5, 6]).astype(int)
        
        # Time-based capacity features
        admissions_by_day = self.df.groupby('admission_datetime').size()
        self.df['daily_admission_count'] = self.df['admission_datetime'].map(admissions_by_day)
        
        logger.info("Created temporal features")
    
    def create_clinical_severity_features(self) -> None:
        """Create features related to clinical severity and patient condition."""
        # Vital signs risk score
        vital_cols = ['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
                     'respiratory_rate', 'temperature', 'oxygen_saturation']
        
        # Define normal ranges for vital signs
        normal_ranges = {
            'heart_rate': (60, 100),
            'blood_pressure_systolic': (90, 140),
            'blood_pressure_diastolic': (60, 90),
            'respiratory_rate': (12, 20),
            'temperature': (36.5, 37.5),
            'oxygen_saturation': (95, 100)
        }
        
        # Calculate deviation from normal range
        for col in vital_cols:
            if col in self.df.columns:
                low, high = normal_ranges.get(col, (0, 0))
                self.df[f'{col}_deviation'] = self.df[col].apply(
                    lambda x: min(x - high, low - x, 0) if pd.notnull(x) else 0
                )
        
        # Create composite severity score
        self.df['vital_signs_risk_score'] = self.df[[f'{col}_deviation' for col in vital_cols 
                                                    if col in self.df.columns]].sum(axis=1)
        
        # Lab results severity score (if available)
        lab_cols = [col for col in self.df.columns if 'lab_' in col.lower()]
        if lab_cols:
            # Z-score based severity
            for col in lab_cols:
                z_scores = stats.zscore(self.df[col].fillna(self.df[col].median()))
                self.df[f'{col}_zscore'] = np.abs(z_scores)
            
            self.df['lab_severity_score'] = self.df[[f'{col}_zscore' for col in lab_cols]].mean(axis=1)
        
        logger.info("Created clinical severity features")
    
    def create_social_determinant_features(self) -> None:
        """Create features related to social determinants of health."""
        # Insurance type impact
        if 'insurance_type' in self.df.columns:
            insurance_los_mean = self.df.groupby('insurance_type')[self.target_col].transform('mean')
            self.df['insurance_los_factor'] = insurance_los_mean / insurance_los_mean.mean()
        
        # Support system indicator
        support_cols = ['marital_status', 'lives_alone', 'caregiver_availability']
        if all(col in self.df.columns for col in support_cols):
            self.df['support_system_score'] = (
                (self.df['marital_status'] == 'Married').astype(int) +
                (~self.df['lives_alone']).astype(int) +
                self.df['caregiver_availability'].astype(int)
            )
        
        logger.info("Created social determinant features")
    
    def create_interaction_features(self, top_n: int = 5) -> None:
        """Create interaction features between most important numeric variables."""
        # Select top correlated features with target
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop(self.target_col)
        
        correlations = self.df[numeric_cols].corrwith(self.df[self.target_col]).abs()
        top_features = correlations.nlargest(top_n).index
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        interactions = poly.fit_transform(self.df[top_features])
        
        # Add interaction features to dataframe
        feature_names = [f"{a}_{b}_interaction" 
                        for i, a in enumerate(top_features) 
                        for b in top_features[i+1:]]
        
        for idx, name in enumerate(feature_names):
            self.df[name] = interactions[:, idx + len(top_features)]
        
        logger.info(f"Created {len(feature_names)} interaction features")
    
    def engineer_features(self) -> pd.DataFrame:
        """Execute complete feature engineering pipeline."""
        logger.info("Starting feature engineering process...")
        
        # Execute each feature engineering step
        self.create_medical_complexity_features()
        self.create_temporal_features()
        self.create_clinical_severity_features()
        self.create_social_determinant_features()
        self.create_interaction_features()
        
        # Remove temporary columns used in feature creation
        cols_to_drop = [col for col in self.df.columns if '_temp' in col]
        self.df = self.df.drop(columns=cols_to_drop)
        
        logger.info(f"Feature engineering complete. Total features: {len(self.df.columns)}")
        
        return self.df
