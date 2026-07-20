# SCTT-Net manuscript implementation

This folder is a complete copy of the original code with the core model revised to match `manuscript0712.docx`.

The former Autoformer and auxiliary attention source files are retained for traceability. The revised SCTT-Net model does not import or execute them.


- The global branch is an encoder-decoder Transformer.
- The local branch is a causal dilated TCN.
- The Basic Gate uses a two-hidden-layer MLP with PCP, SW, and PET.
- The Physical Gate uses the current state and 3-day and 7-day memories.
- The learnable coefficient is `lambda = sigmoid(theta)`.
- Fusion follows `g * F_global + (1 - g) * F_local`.
- Data are split into 2002 to 2016, 2017 to 2019, and 2020 to 2022.
- Metrics include NSE, KGE, RMSE, and IOA.
- Prediction lengths of 1, 3, and 7 are supported through `PRED_LEN`.


## Run

Edit `configs/config.py` when changing the station file or forecast horizon.

```powershell
python -m SCTT_Net_manuscript0712_modified.train
```

Set `RUN_BAYESIAN_OPTIMIZATION = True` to optimize `w1` to `w7` before final training.

For architectural ablation, set `MODEL_VARIANT` to `MG`, `Transformer`, or `TCN`. The NV and NS comparisons require the corresponding uncalibrated-SWAT or meteorology-only input files and are controlled through `DATA_PATH` and `METEO_FEATURES`.

After training and installing SHAP, run:

```powershell
python -m SCTT_Net_manuscript0712_modified.shap_analysis
```

Outputs are written to the package `outputs` folder.
