import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Tuple
import plotly.graph_objects as go


class ModelEvaluator:
    def __init__(self, model, device, feature_names: List[str]):
        self.model = model
        self.device = device
        self.feature_names = feature_names

    def compute_metrics(self, loader) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics"""
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                predictions.extend(out.cpu().numpy())
                actuals.extend(batch.y.cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        return {
            'mse': mean_squared_error(actuals, predictions),
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'mae': mean_absolute_error(actuals, predictions),
            'r2': r2_score(actuals, predictions),
            'mean_error': np.mean(predictions - actuals),
            'std_error': np.std(predictions - actuals)
        }

    def analyze_errors(self, loader) -> pd.DataFrame:
        """Analyze prediction errors in detail"""
        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                predictions = self.model(batch.x, batch.edge_index, batch.batch).cpu().numpy()
                actuals = batch.y.cpu().numpy()

                for pred, actual in zip(predictions, actuals):
                    error = pred - actual
                    results.append({
                        'predicted': pred,
                        'actual': actual,
                        'error': error,
                        'abs_error': abs(error)
                    })

        return pd.DataFrame(results)

    def plot_prediction_analysis(self, error_df: pd.DataFrame) -> None:
        """Create visualization of prediction analysis"""
        fig = go.Figure()

        # Scatter plot
        fig.add_trace(go.Scatter(
            x=error_df['actual'],
            y=error_df['predicted'],
            mode='markers',
            name='Predictions'
        ))

        # Identity line
        max_val = max(error_df['actual'].max(), error_df['predicted'].max())
        min_val = min(error_df['actual'].min(), error_df['predicted'].min())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash')
        ))

        fig.update_layout(
            title='Predicted vs Actual Length of Stay',
            xaxis_title='Actual Length of Stay',
            yaxis_title='Predicted Length of Stay',
            height=600
        )

        return fig

    def compute_feature_importance(self, loader) -> pd.DataFrame:
        """Compute feature importance using integrated gradients"""
        self.model.eval()
        importance_scores = np.zeros(len(self.feature_names))
        n_samples = 0

        for batch in loader:
            batch = batch.to(self.device)
            batch.x.requires_grad = True

            # Forward pass
            out = self.model(batch.x, batch.edge_index, batch.batch)

            # Compute gradients
            grad_outputs = torch.ones_like(out)
            gradients = torch.autograd.grad(
                outputs=out,
                inputs=batch.x,
                grad_outputs=grad_outputs,
                create_graph=True
            )[0]

            # Compute importance scores
            importance = torch.abs(gradients * batch.x).mean(dim=0)
            importance_scores += importance.cpu().detach().numpy()
            n_samples += 1

        # Average importance scores
        importance_scores /= n_samples

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

    def explain_predictions(self, loader, num_background=100) -> Dict:
        """Generate SHAP explanations for predictions"""
        self.model.eval()

        # Get background data
        background_data = []
        for batch in loader:
            background_data.append(batch.x.cpu().numpy())
            if len(background_data) * batch.x.shape[0] >= num_background:
                break
        background_data = np.concatenate(background_data)[:num_background]

        # Create explainer
        explainer = shap.DeepExplainer(
            model=self.model,
            data=torch.tensor(background_data).to(self.device)
        )

        # Get SHAP values for test data
        shap_values = []
        test_data = []

        for batch in loader:
            batch_shap_values = explainer.shap_values(batch.x.to(self.device))
            shap_values.append(batch_shap_values[0])
            test_data.append(batch.x.cpu().numpy())

        shap_values = np.concatenate(shap_values)
        test_data = np.concatenate(test_data)

        return {
            'shap_values': shap_values,
            'test_data': test_data,
            'feature_names': self.feature_names
        }

    def plot_learning_curves(self, train_losses: List[float], val_losses: List[float]) -> None:
        """Plot training and validation learning curves"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=train_losses,
            name='Training Loss',
            mode='lines'
        ))

        fig.add_trace(go.Scatter(
            y=val_losses,
            name='Validation Loss',
            mode='lines'
        ))

        fig.update_layout(
            title='Learning Curves',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            height=500
        )

        return fig