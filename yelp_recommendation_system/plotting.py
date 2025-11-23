import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Any

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_training_history(
        history: Dict[str, List],
        save_path: Optional[str] = None,
        show: bool = True):
    """
    Args:
        history: Training history dictionary containing 'epoch', 'train_rmse', 'train_mae'
        save_path: Optional path to save the plot
        show: Whether to display the plot

    """
    # check if history contains required keys, if there is no history you cant plot
    if not history or 'epoch' not in history:
        raise ValueError("History must contain 'epoch' key")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = history['epoch']

    # plot RMSE
    if 'train_rmse' in history:
        ax1.plot(epochs, history['train_rmse'], 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('RMSE', fontsize=12)
        ax1.set_title('Training RMSE Over Epochs', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

    # plot MAE
    if 'train_mae' in history:
        ax2.plot(epochs, history['train_mae'], 'g-', linewidth=2, marker='o')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MAE', fontsize=12)
        ax2.set_title('Training MAE Over Epochs', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_metrics_summary(
        metrics: Dict[str, float],
        save_path: Optional[str] = None,
        show: bool = True):
    """
    Create a comprehensive visualization of model evaluation metrics.

    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Optional path to save the plot
        show: Whether to display the plot

    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # error metrics (RMSE, MAE)
    if 'RMSE' in metrics and 'MAE' in metrics:
        error_metrics = ['RMSE', 'MAE']
        error_values = [metrics[m] for m in error_metrics]

        axes[0, 0].bar(error_metrics, error_values, color=['#3498db', '#2ecc71'], alpha=0.8)
        axes[0, 0].set_title('Error Metrics', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Value', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(error_values):
            axes[0, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

    # precision@k
    precision_metrics = [k for k in metrics.keys() if k.startswith('Precision@')]
    if precision_metrics:
        k_values = [int(m.split('@')[1]) for m in precision_metrics]
        precision_values = [metrics[m] for m in precision_metrics]

        axes[0, 1].plot(k_values, precision_values, 'o-', linewidth=2, markersize=8, color='#e74c3c')
        axes[0, 1].set_title('Precision@K', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('K (Number of Recommendations)', fontsize=12)
        axes[0, 1].set_ylabel('Precision', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)

        for k, p in zip(k_values, precision_values):
            axes[0, 1].text(k, p, f'{p:.3f}', ha='center', va='bottom')

    # recall@k
    recall_metrics = [k for k in metrics.keys() if k.startswith('Recall@')]
    if recall_metrics:
        k_values = [int(m.split('@')[1]) for m in recall_metrics]
        recall_values = [metrics[m] for m in recall_metrics]

        axes[1, 0].plot(k_values, recall_values, 's-', linewidth=2, markersize=8, color='#9b59b6')
        axes[1, 0].set_title('Recall@K', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('K (Number of Recommendations)', fontsize=12)
        axes[1, 0].set_ylabel('Recall', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)

        for k, r in zip(k_values, recall_values):
            axes[1, 0].text(k, r, f'{r:.3f}', ha='center', va='bottom')

    # precision-recal curve
    if precision_metrics and recall_metrics:
        axes[1, 1].plot(recall_values, precision_values, 'D-', linewidth=2, markersize=8, color='#f39c12')
        axes[1, 1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Recall', fontsize=12)
        axes[1, 1].set_ylabel('Precision', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)

        for k, p, r in zip(k_values, precision_values, recall_values):
            axes[1, 1].text(r, p, f'K={k}', ha='left', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_prediction_distribution(
        predictions: pd.DataFrame,
        rating_column: str = 'predicted_rating',
        bins: int = 20,
        save_path: Optional[str] = None,
        show: bool = True):
    """
    Args:
        predictions: DataFrame containing predictions
        rating_column: Name of the column with ratings
        bins: Number of histogram bins
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ratings = predictions[rating_column].dropna()

    ax.hist(ratings, bins=bins, color='#3498db', alpha=0.7, edgecolor='black')
    ax.axvline(ratings.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ratings.mean():.2f}')
    ax.axvline(ratings.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {ratings.median():.2f}')

    ax.set_xlabel('Predicted Rating', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Predicted Ratings', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()

    return fig


