from pathlib import Path


class SCTTNetConfig:
    """Configuration for the manuscript-aligned SCTT-Net implementation."""

    MODEL_NAME = "SCTT-Net"
    MODEL_VARIANT = "SCTT-Net"
    DATA_PATH = r"E:\graduatestudent\SWAT-GHCformer\model\Autoformer-main\data\头道拐.xlsx"
    TIME_COL = "time"
    TARGET_COL = "runoff"

    # Meteorological variables, SWAT state variables, and SWAT-simulated runoff.
    # GW_RCHGmm is the available workbook field closest to the manuscript PERC term.
    METEO_FEATURES = [
        "prec",
        "tmax",
        "tmin",
        "pres",
        "wind",
        "rhu",
        "SOLARmj_m2",
        "WYLD_Qmm",
        "LATQ_mm",
        "GW_RCHGmm",
        "PETmm",
        "GW_Qmm",
        "SW_INITmm",
        "SURQ_CNTmm",
    ]

    FEATURE_DISPLAY_NAMES = {
        "prec": "PCP",
        "tmax": "HTEM",
        "tmin": "LTEM",
        "pres": "PRES",
        "wind": "WS",
        "rhu": "RH",
        "SOLARmj_m2": "SR",
        "WYLD_Qmm": "SWAT runoff",
        "LATQ_mm": "LATQ",
        "GW_RCHGmm": "PERC",
        "PETmm": "PET",
        "GW_Qmm": "GWQ",
        "SW_INITmm": "SW",
        "SURQ_CNTmm": "SURQ",
        "runoff": "Observed runoff history",
    }

    # Equation (1) to Equation (6) use precipitation, soil water, and PET.
    GATE_FEATURES = ["prec", "SW_INITmm", "PETmm"]

    INCLUDE_TARGET_IN_INPUT = True

    SEQ_LEN = 30
    LABEL_LEN = 15
    PRED_LEN = 1

    # Manuscript periods. Scaling statistics are fitted on 2002 to 2016 only.
    USE_DATE_SPLIT = True
    TRAIN_START = "2002-01-01"
    TRAIN_END = "2016-12-31"
    VAL_START = "2017-01-01"
    VAL_END = "2019-12-31"
    TEST_START = "2020-01-01"
    TEST_END = "2022-12-31"

    # Ratio split is retained as an explicit fallback for datasets without dates.
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    D_MODEL = 128
    N_HEADS = 8
    E_LAYERS = 2
    D_LAYERS = 1
    D_FF = 512
    DROPOUT = 0.1
    ACTIVATION = "gelu"

    TCN_CHANNELS = [128, 128, 128]
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.2

    GATE_HIDDEN_DIMS = [32, 16]
    GATE_DROPOUT = 0.1
    GATE_CURRENT_RATIO = 0.7
    GATE_MEMORY_RATIO = 0.3
    GATE_THETA_INIT = 0.0
    GATE_SHORT_MEMORY = 3
    GATE_LONG_MEMORY = 7

    # Physical-gate weights in Equations (3) and (4).
    GATE_W1 = 1.5
    GATE_W2 = 1.0
    GATE_W3 = 1.0
    GATE_W4 = 1.0
    GATE_W5 = 0.8
    GATE_W6 = 0.5
    GATE_W7 = 0.6

    RUN_BAYESIAN_OPTIMIZATION = False
    BAYESIAN_TRIALS = 20
    BAYESIAN_EPOCHS = 5
    BAYESIAN_WEIGHT_LOW = -2.0
    BAYESIAN_WEIGHT_HIGH = 2.0

    EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-5
    LR_SCHEDULE = "cosine"
    MIN_LR = 1e-6
    EARLY_STOPPING = True
    PATIENCE = 10

    SAVE_DIR = str(Path(__file__).resolve().parents[1] / "outputs")
    SAVE_BEST_ONLY = True

    USE_GPU = True
    MIXED_PRECISION = False
    RANDOM_SEED = 42
    VERBOSE = 1

    SHAP_BACKGROUND_SIZE = 64
    SHAP_EXPLAIN_SIZE = 128

    @property
    def num_meteo_features(self) -> int:
        base = len(self.METEO_FEATURES)
        return base + int(self.INCLUDE_TARGET_IN_INPUT)

    @property
    def num_gate_features(self) -> int:
        return len(self.GATE_FEATURES)

    @property
    def input_feature_names(self) -> list[str]:
        names = list(self.METEO_FEATURES)
        if self.INCLUDE_TARGET_IN_INPUT:
            names.append(self.TARGET_COL)
        return names

    def __repr__(self) -> str:
        return (
            f"{self.MODEL_NAME}(seq_len={self.SEQ_LEN}, pred_len={self.PRED_LEN}, "
            f"meteo_features={self.num_meteo_features}, "
            f"gate_features={self.GATE_FEATURES})"
        )


default_config = SCTTNetConfig()


if __name__ == "__main__":
    print(SCTTNetConfig())
