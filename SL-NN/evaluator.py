"""
Evaluation and Visualization
Generates all required plots and metrics for the assignment.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, auc, confusion_matrix,
    mean_absolute_error, mean_squared_error, median_absolute_error,
    calibration_curve
)
from typing import Dict, List, Tuple, Optional
import pandas as pd


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


class NNEvaluator:
    """
    Comprehensive evaluation for neural networks.
    
    Generates all required plots:
    1. Learning curves (train/val metric vs. training size)
    2. Model complexity curves (metric vs. hyperparameter)
    3. Epoch curves (loss/metric vs. epoch with early stopping marker)
    4. Classification-specific: ROC, PR curves, confusion matrix, calibration
    5. Regression-specific: residual plots
    """
    
    def __init__(self, task_type: str, output_dir: str = './results/figures'):
        self.task_type = task_type
        self.output_dir = output_dir
    
    # ========================================================================
    # EPOCH CURVES (Required: iteration-based learning curve)
    # ========================================================================
    
    def plot_epoch_curve(
        self,
        history: Dict,
        best_epoch: int,
        save_path: Optional[str] = None
    ):
        """
        Plot training and validation metrics vs. epoch.
        
        Shows early stopping marker at best epoch.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = history['epochs']
        
        # Loss curve
        ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 's-', label='Val Loss', linewidth=2)
        ax1.axvline(best_epoch, color='red', linestyle='--', 
                    label=f'Best Epoch ({best_epoch})', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss vs. Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Metric curve
        metric_name = 'ROC-AUC' if self.task_type == 'classification' else 'MAE'
        train_metric = [abs(m) for m in history['train_metric']]
        val_metric = [abs(m) for m in history['val_metric']]
        
        ax2.plot(epochs, train_metric, 'o-', label=f'Train {metric_name}', linewidth=2)
        ax2.plot(epochs, val_metric, 's-', label=f'Val {metric_name}', linewidth=2)
        ax2.axvline(best_epoch, color='red', linestyle='--', 
                    label=f'Best Epoch ({best_epoch})', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name)
        ax2.set_title(f'{metric_name} vs. Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved epoch curve: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # LEARNING CURVES (Required: metric vs. training size)
    # ========================================================================
    
    def plot_learning_curve(
        self,
        train_sizes: List[int],
        train_scores: List[float],
        val_scores: List[float],
        metric_name: str = 'ROC-AUC',
        save_path: Optional[str] = None
    ):
        """
        Plot learning curve: performance vs. training set size.
        
        Diagnoses bias/variance tradeoff.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, train_scores, 'o-', label='Train Score', 
                linewidth=2, markersize=8)
        ax.plot(train_sizes, val_scores, 's-', label='Val Score', 
                linewidth=2, markersize=8)
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel(metric_name)
        ax.set_title(f'Learning Curve: {metric_name} vs. Training Size')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add diagnostic annotations
        gap = abs(train_scores[-1] - val_scores[-1])
        if gap > 0.1:
            ax.text(0.5, 0.05, 'High variance (overfitting)', 
                   transform=ax.transAxes, fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        elif val_scores[-1] < 0.7:
            ax.text(0.5, 0.05, 'High bias (underfitting)', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved learning curve: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # MODEL COMPLEXITY CURVES (Required: metric vs. hyperparameter)
    # ========================================================================
    
    def plot_complexity_curve(
        self,
        param_values: List,
        train_scores: List[float],
        val_scores: List[float],
        param_name: str = 'Width',
        metric_name: str = 'ROC-AUC',
        save_path: Optional[str] = None
    ):
        """
        Plot model complexity curve: performance vs. one hyperparameter.
        
        Examples:
        - Width variations at fixed depth
        - L2 regularization strength
        - Learning rate
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(param_values, train_scores, 'o-', label='Train Score', 
                linewidth=2, markersize=8)
        ax.plot(param_values, val_scores, 's-', label='Val Score', 
                linewidth=2, markersize=8)
        
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric_name)
        ax.set_title(f'Model Complexity: {metric_name} vs. {param_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Mark best parameter
        best_idx = np.argmax(val_scores)
        best_param = param_values[best_idx]
        ax.axvline(best_param, color='red', linestyle='--', alpha=0.7,
                  label=f'Best {param_name}: {best_param}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved complexity curve: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # CLASSIFICATION-SPECIFIC PLOTS
    # ========================================================================
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved ROC curve: {save_path}")
        
        plt.show()
        
        return roc_auc
    
    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot Precision-Recall curve.
        
        CRITICAL for imbalanced datasets.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        # Prevalence baseline
        prevalence = y_true.mean()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
        ax.axhline(prevalence, color='red', linestyle='--', linewidth=1,
                  label=f'Prevalence Baseline ({prevalence:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved PR curve: {save_path}")
        
        plt.show()
        
        return pr_auc, prevalence
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=labels, yticklabels=labels)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved confusion matrix: {save_path}")
        
        plt.show()
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Plot reliability diagram (calibration curve).
        
        Required if reporting probabilities for classification.
        """
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(prob_pred, prob_true, 's-', linewidth=2, markersize=8,
               label='Model')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve (Reliability Diagram)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved calibration curve: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # REGRESSION-SPECIFIC PLOTS
    # ========================================================================
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot residuals vs. predictions.
        
        Diagnoses:
        - Homoscedasticity
        - Systematic bias
        - Outliers
        """
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs. predictions
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Value')
        ax1.set_ylabel('Residual')
        ax1.set_title('Residuals vs. Predictions')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residual Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved residual plots: {save_path}")
        
        plt.show()
    
    # ========================================================================
    # COMPREHENSIVE EVALUATION REPORT
    # ========================================================================
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute all relevant metrics for the task.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if self.task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            
            if y_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                
                # PR-AUC
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                metrics['pr_auc'] = auc(recall, precision)
                metrics['prevalence'] = y_true.mean()
        
        else:  # Regression
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['median_ae'] = median_absolute_error(y_true, y_pred)
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Pretty print metrics."""
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        
        for key, value in metrics.items():
            print(f"{key.upper():20s}: {value:.4f}")
        
        print("=" * 60)


if __name__ == "__main__":
    print("NNEvaluator module loaded successfully")
    print("\nTo use:")
    print("  evaluator = NNEvaluator(task_type='classification')")
    print("  evaluator.plot_epoch_curve(history, best_epoch)")
    print("  evaluator.plot_learning_curve(...)")
    print("  metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)")
