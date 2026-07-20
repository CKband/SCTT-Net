
import numpy as np
from typing import Dict


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(calculate_mse(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    r2 = 1 - (ss_res / ss_tot)
    return r2


def calculate_nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    if denominator == 0:
        return 0.0

    nse = 1 - (numerator / denominator)
    return nse


def calculate_kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kling-Gupta efficiency using correlation, variability, and bias ratios."""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    true_std = np.std(y_true)
    true_mean = np.mean(y_true)
    if true_std == 0 or true_mean == 0:
        return 0.0

    pred_std = np.std(y_pred)
    pred_mean = np.mean(y_pred)
    if len(y_true) < 2:
        correlation = 0.0
    else:
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        if not np.isfinite(correlation):
            correlation = 0.0

    alpha = pred_std / true_std
    beta = pred_mean / true_mean
    return float(1.0 - np.sqrt(
        (correlation - 1.0) ** 2
        + (alpha - 1.0) ** 2
        + (beta - 1.0) ** 2
    ))


def calculate_ioa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Willmott index of agreement."""
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    observed_mean = np.mean(y_true)
    numerator = np.sum((y_pred - y_true) ** 2)
    denominator = np.sum(
        (np.abs(y_pred - observed_mean) + np.abs(y_true - observed_mean)) ** 2
    )
    if denominator == 0:
        return 0.0
    return float(1.0 - numerator / denominator)


def calculate_pbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    numerator = np.sum(y_true - y_pred)
    denominator = np.sum(y_true)

    if denominator == 0:
        return 0.0

    pbias = 100 * (numerator / denominator)
    return pbias


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    metrics = {
        'MSE': calculate_mse(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred),
        'NSE': calculate_nse(y_true, y_pred),
        'KGE': calculate_kge(y_true, y_pred),
        'IOA': calculate_ioa(y_true, y_pred),
        'PBIAS': calculate_pbias(y_true, y_pred)
    }

    return metrics


def print_metrics(metrics: Dict[str, float], dataset_name: str = "Dataset"):
    print(f"\n{dataset_name} Evaluation Metrics:")
    print("-" * 50)
    print(f"MSE:    {metrics['MSE']:.4f}")
    print(f"RMSE:   {metrics['RMSE']:.4f}")
    print(f"MAE:    {metrics['MAE']:.4f}")
    print(f"R2:     {metrics['R2']:.4f}")
    print(f"NSE:    {metrics['NSE']:.4f}")
    print(f"KGE:    {metrics['KGE']:.4f}")
    print(f"IOA:    {metrics['IOA']:.4f}")
    print(f"PBIAS:  {metrics['PBIAS']:.2f}%")
    print("-" * 50)


if __name__ == "__main__":

    print("=" * 60)
    print("Testing evaluation metrics module")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 1000

    print("\nTest 1: Perfect prediction")
    y_true = np.random.randn(n_samples)
    y_pred = y_true.copy()

    metrics = calculate_all_metrics(y_true, y_pred)
    print_metrics(metrics, "Perfect prediction")

    print("\nTest 2: Noisy prediction")
    y_true = np.random.randn(n_samples) * 10 + 50
    y_pred = y_true + np.random.randn(n_samples) * 2

    metrics = calculate_all_metrics(y_true, y_pred)
    print_metrics(metrics, "Noisy prediction")

    print("\nTest 3: Systematic bias (overestimate)")
    y_true = np.random.randn(n_samples) * 10 + 50
    y_pred = y_true + 5

    metrics = calculate_all_metrics(y_true, y_pred)
    print_metrics(metrics, "Systematic overestimate")

    print("\nTest 4: Systematic bias (underestimate)")
    y_true = np.random.randn(n_samples) * 10 + 50
    y_pred = y_true - 5

    metrics = calculate_all_metrics(y_true, y_pred)
    print_metrics(metrics, "Systematic underestimate")

    print("\nTest 5: Poor prediction")
    y_true = np.random.randn(n_samples) * 10 + 50
    y_pred = np.random.randn(n_samples) * 10 + 50

    metrics = calculate_all_metrics(y_true, y_pred)
    print_metrics(metrics, "Poor prediction")

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)
