import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor

def select_features_shap(df, target_col, n_features=20):
    """Perform feature selection using SHAP values with HistGradientBoostingRegressor"""
    X = df.drop([target_col, 'unique_id'], axis=1)
    y = df[target_col]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X, y)

    # Using SHAP's KernelExplainer for HistGradientBoostingRegressor
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    feature_importance = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    return feature_importance_df