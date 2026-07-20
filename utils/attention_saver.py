"""Compatibility utilities for saving SCTT-Net gate and branch diagnostics."""

from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def save_attention_data(
    model,
    data_loader,
    device,
    save_path,
    num_samples=None,
    feature_names=None,
):
    """Save gate diagnostics under the historical utility function name."""
    model.eval()
    saved = {
        "gate_values": [],
        "g_mlp": [],
        "g_phy": [],
        "score_current": [],
        "score_memory": [],
        "predictions": [],
        "targets": [],
        "meteo_inputs": [],
        "gate_inputs": [],
    }

    count = 0
    with torch.no_grad():
        for meteo_input, gate_input, target in data_loader:
            meteo_input = {key: value.to(device) for key, value in meteo_input.items()}
            gate_input = gate_input.to(device)
            output, diagnostics = model(
                meteo_input, gate_input, return_diagnostics=True
            )

            saved["gate_values"].append(diagnostics["gate"].cpu().numpy())
            saved["g_mlp"].append(diagnostics["g_mlp"].cpu().numpy())
            saved["g_phy"].append(diagnostics["g_phy"].cpu().numpy())
            saved["score_current"].append(
                diagnostics["score_current"].cpu().numpy()
            )
            saved["score_memory"].append(
                diagnostics["score_memory"].cpu().numpy()
            )
            saved["predictions"].append(output.cpu().numpy())
            saved["targets"].append(target.numpy())
            saved["meteo_inputs"].append(meteo_input["enc_x"].cpu().numpy())
            saved["gate_inputs"].append(gate_input.cpu().numpy())
            count += target.size(0)
            if num_samples is not None and count >= num_samples:
                break

    arrays = {key: np.concatenate(value, axis=0) for key, value in saved.items()}
    if num_samples is not None:
        arrays = {key: value[:num_samples] for key, value in arrays.items()}

    arrays["lambda"] = np.array(float(torch.sigmoid(model.gate_controller.theta)))
    arrays["timestamp"] = np.array(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if feature_names is not None:
        arrays["feature_names"] = np.array(feature_names, dtype=object)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **arrays)
    return str(save_path)


def load_attention_data(file_path):
    data = np.load(Path(file_path), allow_pickle=True)
    return {key: data[key] for key in data.files}


def extract_sample_for_visualization(
    data_dict,
    sample_idx=0,
    meteo_feature_names=None,
):
    feature_names = meteo_feature_names
    if feature_names is None and "feature_names" in data_dict:
        feature_names = data_dict["feature_names"].tolist()

    return {
        "gate_values": data_dict["gate_values"][sample_idx],
        "g_mlp": data_dict["g_mlp"][sample_idx],
        "g_phy": data_dict["g_phy"][sample_idx],
        "score_current": data_dict["score_current"][sample_idx],
        "score_memory": data_dict["score_memory"][sample_idx],
        "gate_inputs": data_dict["gate_inputs"][sample_idx],
        "meteo_inputs": data_dict["meteo_inputs"][sample_idx],
        "feature_names": feature_names,
        "prediction": data_dict["predictions"][sample_idx],
        "target": data_dict["targets"][sample_idx],
    }
