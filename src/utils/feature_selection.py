import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

def select_features_shap(df, target_col, n_features=20):
    """Perform feature selection using SHAP values"""
    X = df.drop([target_col, 'unique_id'], axis=1)
    y = df[target_col]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    feature_importance = np.abs(shap_values).mean(0)
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance_df.head(n_features)['feature'].tolist()