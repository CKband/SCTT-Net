from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "Bayesian optimization requires optuna. Install the project requirements."
    ) from exc

try:
    from .configs import SCTTNetConfig
    from .model import build_sctt_net_from_config
except ImportError:
    package_parent = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(package_parent))
    from SCTT_Net_manuscript0712_modified.configs import SCTTNetConfig
    from SCTT_Net_manuscript0712_modified.model import build_sctt_net_from_config


def _move_meteo(meteo_input, device):
    return {key: value.to(device) for key, value in meteo_input.items()}


def _validation_loss(model, data_loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    samples = 0
    with torch.no_grad():
        for meteo_input, gate_input, target in data_loader:
            output = model(
                _move_meteo(meteo_input, device),
                gate_input.to(device),
            )
            target = target.to(device)
            total += criterion(output, target).item() * target.size(0)
            samples += target.size(0)
    return total / max(samples, 1)


def optimize_gate_weights(config, train_loader, val_loader, device):
    """Estimate w1 to w7 in the manuscript physical gate with Optuna."""

    def objective(trial: optuna.Trial) -> float:
        trial_config = copy.deepcopy(config)
        for index in range(1, 8):
            setattr(
                trial_config,
                f"GATE_W{index}",
                trial.suggest_float(
                    f"w{index}",
                    config.BAYESIAN_WEIGHT_LOW,
                    config.BAYESIAN_WEIGHT_HIGH,
                ),
            )

        model = build_sctt_net_from_config(trial_config).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        best_loss = float("inf")
        for epoch in range(config.BAYESIAN_EPOCHS):
            model.train()
            for meteo_input, gate_input, target in train_loader:
                meteo_input = _move_meteo(meteo_input, device)
                gate_input = gate_input.to(device)
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(meteo_input, gate_input), target)
                loss.backward()
                optimizer.step()

            val_loss = _validation_loss(model, val_loader, criterion, device)
            best_loss = min(best_loss, val_loss)
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        study_name="sctt_net_physical_gate",
    )
    study.optimize(objective, n_trials=config.BAYESIAN_TRIALS)

    output_dir = Path(config.SAVE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(
        output_dir / "bayesian_trials.csv", index=False
    )
    with (output_dir / "best_gate_weights.json").open("w", encoding="utf-8") as file:
        json.dump(study.best_params, file, indent=2)

    return study.best_params


def main():
    try:
        from .train import prepare_experiment_data, setup_device
    except ImportError:
        from SCTT_Net_manuscript0712_modified.train import (
            prepare_experiment_data,
            setup_device,
        )

    config = SCTTNetConfig()
    device = setup_device(config.USE_GPU)
    _, _, loaders, _ = prepare_experiment_data(config)
    best = optimize_gate_weights(config, loaders[0], loaders[1], device)
    print(best)


if __name__ == "__main__":
    main()
