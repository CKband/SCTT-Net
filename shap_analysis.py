from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "SHAP analysis requires the shap package. Install the project requirements."
    ) from exc

try:
    from .configs import SCTTNetConfig
    from .model import build_sctt_net_from_config
    from .train import prepare_experiment_data, setup_device
except ImportError:
    package_parent = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(package_parent))
    from SCTT_Net_manuscript0712_modified.configs import SCTTNetConfig
    from SCTT_Net_manuscript0712_modified.model import build_sctt_net_from_config
    from SCTT_Net_manuscript0712_modified.train import (
        prepare_experiment_data,
        setup_device,
    )


class ForecastWrapper(nn.Module):
    """Expose a scalar forecast to SHAP while retaining both model inputs."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, encoder_input: torch.Tensor, gate_input: torch.Tensor):
        prediction = self.model(encoder_input, gate_input)
        return prediction.mean(dim=(1, 2)).unsqueeze(-1)


def collect_inputs(data_loader, limit: int):
    encoder_inputs = []
    gate_inputs = []
    count = 0
    for meteo_input, gate_input, _ in data_loader:
        encoder_inputs.append(meteo_input["enc_x"])
        gate_inputs.append(gate_input)
        count += gate_input.size(0)
        if count >= limit:
            break
    return (
        torch.cat(encoder_inputs, dim=0)[:limit],
        torch.cat(gate_inputs, dim=0)[:limit],
    )


def save_importance_plot(table: pd.DataFrame, output_path: Path, title: str):
    ordered = table.sort_values("mean_absolute_shap", ascending=True)
    height = max(4.0, 0.35 * len(ordered))
    fig, axis = plt.subplots(figsize=(8, height))
    axis.barh(ordered["feature"], ordered["mean_absolute_shap"])
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_shap_analysis(config: SCTTNetConfig):
    device = setup_device(config.USE_GPU)
    _, _, loaders, _ = prepare_experiment_data(config)
    train_loader, _, test_loader = loaders

    checkpoint_path = Path(config.SAVE_DIR) / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_sctt_net_from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    background = collect_inputs(train_loader, config.SHAP_BACKGROUND_SIZE)
    explained = collect_inputs(test_loader, config.SHAP_EXPLAIN_SIZE)
    background = [tensor.to(device) for tensor in background]
    explained = [tensor.to(device) for tensor in explained]

    wrapper = ForecastWrapper(model).to(device).eval()
    explainer = shap.GradientExplainer(wrapper, background)
    shap_values = explainer.shap_values(explained)

    if not isinstance(shap_values, list) or len(shap_values) != 2:
        raise RuntimeError("Unexpected SHAP output for the two SCTT-Net inputs")

    encoder_values = np.asarray(shap_values[0])
    gate_values = np.asarray(shap_values[1])
    encoder_importance = np.mean(np.abs(encoder_values), axis=(0, 1)).reshape(-1)
    gate_importance = np.mean(np.abs(gate_values), axis=(0, 1)).reshape(-1)

    display = config.FEATURE_DISPLAY_NAMES
    encoder_names = [display.get(name, name) for name in config.input_feature_names]
    gate_names = [f"Gate {display.get(name, name)}" for name in config.GATE_FEATURES]

    encoder_table = pd.DataFrame(
        {"feature": encoder_names, "mean_absolute_shap": encoder_importance}
    ).sort_values("mean_absolute_shap", ascending=False)
    gate_table = pd.DataFrame(
        {"feature": gate_names, "mean_absolute_shap": gate_importance}
    ).sort_values("mean_absolute_shap", ascending=False)

    combined = {}
    for name, value in zip(encoder_names, encoder_importance):
        combined[name] = combined.get(name, 0.0) + float(value)
    for raw_name, value in zip(config.GATE_FEATURES, gate_importance):
        name = display.get(raw_name, raw_name)
        combined[name] = combined.get(name, 0.0) + float(value)
    combined_table = pd.DataFrame(
        {
            "feature": list(combined.keys()),
            "mean_absolute_shap": list(combined.values()),
        }
    ).sort_values("mean_absolute_shap", ascending=False)

    output_dir = Path(config.SAVE_DIR) / "shap"
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_table.to_csv(output_dir / "encoder_feature_importance.csv", index=False)
    gate_table.to_csv(output_dir / "gate_feature_importance.csv", index=False)
    combined_table.to_csv(output_dir / "combined_feature_importance.csv", index=False)
    save_importance_plot(
        encoder_table,
        output_dir / "encoder_feature_importance.png",
        "SCTT-Net global feature importance",
    )
    save_importance_plot(
        gate_table,
        output_dir / "gate_feature_importance.png",
        "SCTT-Net physical gate importance",
    )
    save_importance_plot(
        combined_table,
        output_dir / "combined_feature_importance.png",
        "SCTT-Net combined feature importance",
    )
    return encoder_table, gate_table, combined_table


def main():
    run_shap_analysis(SCTTNetConfig())


if __name__ == "__main__":
    main()
