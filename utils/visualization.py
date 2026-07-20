
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import os


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5)
):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    train_loss_key = 'train_loss' if 'train_loss' in history else 'loss'
    axes[0].plot(history[train_loss_key], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    if 'lr' in history:
        axes[1].plot(history['lr'], label='Learning Rate', linewidth=2, color='green')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    else:
        if 'mae' in history:
            axes[1].plot(history['mae'], label='Train MAE', linewidth=2)
            if 'val_mae' in history:
                axes[1].plot(history['val_mae'], label='Val MAE', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('MAE', fontsize=12)
            axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")

    plt.show()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str = "Test Set",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10),
    show_metrics: bool = True
):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    axes[0, 0].plot(y_true, label='Observed', linewidth=1.5, alpha=0.8)
    axes[0, 0].plot(y_pred, label='Predicted', linewidth=1.5, alpha=0.8)
    axes[0, 0].set_xlabel('Time Step', fontsize=11)
    axes[0, 0].set_ylabel('Runoff', fontsize=11)
    axes[0, 0].set_title(f'{dataset_name}: Observed vs Predicted', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(y_true, y_pred, alpha=0.5, s=10)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')

    axes[0, 1].set_xlabel('Observed Runoff', fontsize=11)
    axes[0, 1].set_ylabel('Predicted Runoff', fontsize=11)
    axes[0, 1].set_title('Scatter Plot', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    residuals = y_true - y_pred
    axes[1, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted Runoff', fontsize=11)
    axes[1, 0].set_ylabel('Residuals', fontsize=11)
    axes[1, 0].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Residuals', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    if show_metrics:
        from .metrics import calculate_all_metrics
        metrics = calculate_all_metrics(y_true, y_pred)

        metrics_text = (
            f"R² = {metrics['R2']:.4f}\n"
            f"NSE = {metrics['NSE']:.4f}\n"
            f"RMSE = {metrics['RMSE']:.2f}\n"
            f"MAE = {metrics['MAE']:.2f}\n"
            f"PBIAS = {metrics['PBIAS']:.2f}%"
        )

        axes[0, 1].text(
            0.05, 0.95, metrics_text,
            transform=axes[0, 1].transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to: {save_path}")

    plt.show()


def plot_multi_predictions(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    n_datasets = len(y_true_dict)
    fig, axes = plt.subplots(1, n_datasets, figsize=figsize)

    if n_datasets == 1:
        axes = [axes]

    for idx, (dataset_name, y_true) in enumerate(y_true_dict.items()):
        y_pred = y_pred_dict[dataset_name]

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        axes[idx].scatter(y_true, y_pred, alpha=0.5, s=10)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')

        axes[idx].set_xlabel('Observed', fontsize=11)
        axes[idx].set_ylabel('Predicted', fontsize=11)
        axes[idx].set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)

        from .metrics import calculate_r2, calculate_nse
        r2 = calculate_r2(y_true, y_pred)
        nse = calculate_nse(y_true, y_pred)

        axes[idx].text(
            0.05, 0.95,
            f'R² = {r2:.3f}\nNSE = {nse:.3f}',
            transform=axes[idx].transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-dataset comparison plot saved to: {save_path}")

    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    sorted_indices = np.argsort(importance_scores)
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_scores = importance_scores[sorted_indices]

    plt.figure(figsize=figsize)
    plt.barh(sorted_features, sorted_scores, color='steelblue', alpha=0.8)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")

    plt.show()


if __name__ == "__main__":

    print("=" * 60)
    print("Testing visualization utility module")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 1000

    print("\nTest 1: Plot training history")
    history = {
        'train_loss': np.exp(-np.linspace(0, 3, 50)) + np.random.randn(50) * 0.01,
        'val_loss': np.exp(-np.linspace(0, 2.5, 50)) + np.random.randn(50) * 0.02,
        'lr': np.linspace(1e-4, 1e-6, 50)
    }

    print("\nTest 2: Plot predictions")
    y_true = np.random.randn(n_samples) * 10 + 50
    y_pred = y_true + np.random.randn(n_samples) * 2

    print("\nTest 3: Plot multi-dataset comparison")
    y_true_dict = {
        'Train': np.random.randn(500) * 10 + 50,
        'Val': np.random.randn(300) * 10 + 50,
        'Test': np.random.randn(200) * 10 + 50
    }
    y_pred_dict = {
        'Train': y_true_dict['Train'] + np.random.randn(500) * 2,
        'Val': y_true_dict['Val'] + np.random.randn(300) * 2.5,
        'Test': y_true_dict['Test'] + np.random.randn(200) * 3
    }

    print("\n" + "=" * 60)
    print("Visualization utility tests complete!")
    print("(Plot display is commented out, uncomment for actual use)")
    print("=" * 60)
