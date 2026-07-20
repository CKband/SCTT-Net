from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .configs import SCTTNetConfig
    from .data_processing import (
        RunoffDataPreprocessor,
        create_sequences,
        prepare_dataloaders,
    )
    from .model import build_sctt_net_from_config
    from .utils import calculate_all_metrics, print_metrics
except ImportError:
    package_parent = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(package_parent))
    from SCTT_Net_manuscript0712_modified.configs import SCTTNetConfig
    from SCTT_Net_manuscript0712_modified.data_processing import (
        RunoffDataPreprocessor,
        create_sequences,
        prepare_dataloaders,
    )
    from SCTT_Net_manuscript0712_modified.model import build_sctt_net_from_config
    from SCTT_Net_manuscript0712_modified.utils import (
        calculate_all_metrics,
        print_metrics,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device(use_gpu: bool) -> torch.device:
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def move_meteo_input(meteo_input: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in meteo_input.items()}


def train_epoch(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    scaler=None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for meteo_input, gate_input, target in data_loader:
        meteo_input = move_meteo_input(meteo_input, device)
        gate_input = gate_input.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            output = model(meteo_input, gate_input)
            loss = criterion(output, target)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for meteo_input, gate_input, target in data_loader:
        meteo_input = move_meteo_input(meteo_input, device)
        gate_input = gate_input.to(device)
        target = target.to(device)
        output = model(meteo_input, gate_input)
        total_loss += criterion(output, target).item() * target.size(0)
        total_samples += target.size(0)
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def predict(model: nn.Module, data_loader, device: torch.device):
    model.eval()
    predictions = []
    targets = []
    for meteo_input, gate_input, target in data_loader:
        meteo_input = move_meteo_input(meteo_input, device)
        gate_input = gate_input.to(device)
        predictions.append(model(meteo_input, gate_input).cpu().numpy())
        targets.append(target.numpy())
    return np.concatenate(targets, axis=0), np.concatenate(predictions, axis=0)


def inverse_sequences(preprocessor, values: np.ndarray) -> np.ndarray:
    original_shape = values.shape
    restored = preprocessor.inverse_transform_target(values.reshape(-1, 1))
    return restored.reshape(original_shape)


def calculate_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    metrics = {"overall": calculate_all_metrics(y_true, y_pred)}
    for horizon in range(y_true.shape[1]):
        metrics[f"step_{horizon + 1}"] = calculate_all_metrics(
            y_true[:, horizon, :], y_pred[:, horizon, :]
        )
    return metrics


def config_to_dict(config) -> dict[str, Any]:
    values = {}
    for name in dir(config):
        if name.startswith("_"):
            continue
        value = getattr(config, name)
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            values[name] = value
    return values


def prepare_experiment_data(config):
    preprocessor = RunoffDataPreprocessor(
        data_path=config.DATA_PATH,
        meteo_features=config.METEO_FEATURES,
        gate_features=config.GATE_FEATURES,
        target_col=config.TARGET_COL,
    )
    preprocessor.load_data()
    preprocessor.preprocess(time_col=config.TIME_COL)

    if config.USE_DATE_SPLIT:
        train_df, val_df, test_df = preprocessor.split_by_date(
            config.TRAIN_START,
            config.TRAIN_END,
            config.VAL_START,
            config.VAL_END,
            config.TEST_START,
            config.TEST_END,
        )
    else:
        train_df, val_df, test_df = preprocessor.split_train_val_test(
            config.TRAIN_RATIO,
            config.VAL_RATIO,
            config.TEST_RATIO,
        )

    preprocessor.fit_scalers(train_df)

    transformed = [preprocessor.transform(frame) for frame in (train_df, val_df, test_df)]
    sequence_sets = [
        create_sequences(
            meteo,
            gate,
            target,
            seq_len=config.SEQ_LEN,
            label_len=config.LABEL_LEN,
            pred_len=config.PRED_LEN,
            include_target_in_input=config.INCLUDE_TARGET_IN_INPUT,
        )
        for meteo, gate, target in transformed
    ]

    loaders = prepare_dataloaders(
        sequence_sets[0],
        sequence_sets[1],
        sequence_sets[2],
        batch_size=config.BATCH_SIZE,
        shuffle_train=True,
        num_workers=0,
    )
    frames = {"train": train_df, "validation": val_df, "test": test_df}
    return preprocessor, sequence_sets, loaders, frames


def save_prediction_workbook(
    output_path: Path,
    split_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    rows = []
    for split, (true_values, predicted_values) in split_predictions.items():
        for sample in range(true_values.shape[0]):
            for horizon in range(true_values.shape[1]):
                rows.append(
                    {
                        "split": split,
                        "sample": sample,
                        "forecast_step": horizon + 1,
                        "observed_runoff": float(true_values[sample, horizon, 0]),
                        "predicted_runoff": float(predicted_values[sample, horizon, 0]),
                    }
                )
    pd.DataFrame(rows).to_excel(output_path, index=False)


@torch.no_grad()
def save_gate_diagnostics(model, data_loader, output_path: Path, device: torch.device):
    model.eval()
    rows = []
    sample_offset = 0
    for meteo_input, gate_input, _ in data_loader:
        meteo_input = move_meteo_input(meteo_input, device)
        gate_input = gate_input.to(device)
        _, diagnostics = model(meteo_input, gate_input, return_diagnostics=True)

        batch_size = gate_input.size(0)
        gate = diagnostics["gate"][:, 0, 0].cpu().numpy()
        g_mlp = diagnostics["g_mlp"][:, -1, 0].cpu().numpy()
        g_phy = diagnostics["g_phy"][:, -1, 0].cpu().numpy()
        current = diagnostics["score_current"][:, -1].cpu().numpy()
        memory = diagnostics["score_memory"][:, -1].cpu().numpy()
        lambda_value = float(diagnostics["lambda"].cpu())

        for i in range(batch_size):
            rows.append(
                {
                    "sample": sample_offset + i,
                    "g": float(gate[i]),
                    "g_mlp": float(g_mlp[i]),
                    "g_phy": float(g_phy[i]),
                    "lambda": lambda_value,
                    "score_current": float(current[i]),
                    "score_memory": float(memory[i]),
                }
            )
        sample_offset += batch_size

    pd.DataFrame(rows).to_excel(output_path, index=False)


def train_model(config: SCTTNetConfig):
    set_seed(config.RANDOM_SEED)
    device = setup_device(config.USE_GPU)
    output_dir = Path(config.SAVE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor, sequence_sets, loaders, frames = prepare_experiment_data(config)
    train_loader, val_loader, test_loader = loaders

    if config.RUN_BAYESIAN_OPTIMIZATION:
        try:
            from .bayesian_optimize import optimize_gate_weights
        except ImportError:
            from SCTT_Net_manuscript0712_modified.bayesian_optimize import (
                optimize_gate_weights,
            )
        best_weights = optimize_gate_weights(
            config,
            train_loader,
            val_loader,
            device,
        )
        for index in range(1, 8):
            setattr(config, f"GATE_W{index}", best_weights[f"w{index}"])

    model = build_sctt_net_from_config(config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    if config.LR_SCHEDULE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=config.MIN_LR
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=config.MIN_LR
        )

    use_amp = bool(config.MIXED_PRECISION and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_model_path = output_dir / "best_model.pth"
    history = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp,
            scaler,
        )
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        if config.LR_SCHEDULE == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        learning_rate = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
            }
        )
        print(
            f"Epoch {epoch:03d}/{config.EPOCHS} "
            f"train={train_loss:.6f} val={val_loss:.6f} lr={learning_rate:.3e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": config_to_dict(config),
                },
                best_model_path,
            )
        else:
            patience_counter += 1
            if config.EARLY_STOPPING and patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    split_predictions = {}
    split_metrics = {}
    for split, loader in zip(("train", "validation", "test"), loaders):
        true_scaled, pred_scaled = predict(model, loader, device)
        true_values = inverse_sequences(preprocessor, true_scaled)
        predicted_values = inverse_sequences(preprocessor, pred_scaled)
        split_predictions[split] = (true_values, predicted_values)
        split_metrics[split] = calculate_horizon_metrics(true_values, predicted_values)
        print_metrics(split_metrics[split]["overall"], split.capitalize())

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(split_metrics, file, indent=2, ensure_ascii=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config_to_dict(config), file, indent=2, ensure_ascii=False)
    with (output_dir / "preprocessing.pkl").open("wb") as file:
        pickle.dump(
            {
                "meteo_scaler": preprocessor.meteo_scaler,
                "gate_scaler": preprocessor.gate_scaler,
                "target_scaler": preprocessor.target_scaler,
                "meteo_features": config.METEO_FEATURES,
                "gate_features": config.GATE_FEATURES,
                "target_col": config.TARGET_COL,
            },
            file,
        )

    save_prediction_workbook(output_dir / "predictions.xlsx", split_predictions)
    save_gate_diagnostics(model, test_loader, output_dir / "gate_diagnostics.xlsx", device)

    results = {
        "metrics": split_metrics,
        "predictions": split_predictions,
        "frames": frames,
        "sequence_sets": sequence_sets,
        "checkpoint": str(best_model_path),
    }
    return model, history, results, preprocessor, loaders


def main():
    config = SCTTNetConfig()
    print(config)
    train_model(config)


if __name__ == "__main__":
    main()
