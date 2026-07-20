
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class RunoffDataPreprocessor:

    def __init__(
        self,
        data_path: str,
        meteo_features: List[str],
        gate_features: List[str],
        target_col: str = 'runoff'
    ):
        self.data_path = data_path
        self.meteo_features = meteo_features
        self.gate_features = gate_features
        self.target_col = target_col

        self.meteo_scaler = None
        self.gate_scaler = None
        self.target_scaler = None

        self.df = None

    def load_data(self) -> pd.DataFrame:
        print(f"Loading data: {self.data_path}")

        df = pd.read_excel(self.data_path)

        print(f"Data loaded successfully, shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        self.df = df
        return df

    def preprocess(
        self,
        handle_missing: str = 'interpolate',
        time_col: str = 'time'
    ) -> pd.DataFrame:
        print("\nStarting data preprocessing...")

        df = self.df.copy()

        if time_col in df.columns:
            if df[time_col].dtype != 'datetime64[ns]':
                try:
                    df[time_col] = pd.to_datetime(df[time_col], format='%Y%m%d')
                except:
                    df[time_col] = pd.to_datetime(df[time_col])

            df = df.set_index(time_col)
            print(f"Time range: {df.index.min()} to {df.index.max()}")

        missing_count = df.isnull().sum()
        if missing_count.sum() > 0:
            print(f"\nMissing values found:")
            print(missing_count[missing_count > 0])

            if handle_missing == 'drop':
                df = df.dropna()
                print(f"Samples remaining after dropping missing values: {len(df)}")
            elif handle_missing == 'interpolate':
                df = df.interpolate(method='linear', limit_direction='both')
                print("Filling missing values with linear interpolation")
            elif handle_missing == 'fill_zero':
                df = df.fillna(0)
                print("Filling missing values with 0")

        all_features = self.meteo_features + self.gate_features + [self.target_col]
        missing_cols = [col for col in all_features if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Data is missing the following columns: {missing_cols}")

        print(f"\nPreprocessing complete, final data shape: {df.shape}")
        print(f"Meteo features({len(self.meteo_features)}): {self.meteo_features}")
        print(f"Gate features({len(self.gate_features)}): {self.gate_features}")
        print(f"Target variable: {self.target_col}")

        self.df = df
        return df

    def split_train_val_test(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1"

        df = self.df

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        print(f"\nDataset split:")
        print(f"Train: {len(train_df)} samples ({train_ratio*100:.1f}%)")
        print(f"Validation: {len(val_df)} samples ({val_ratio*100:.1f}%)")
        print(f"Test: {len(test_df)} samples ({test_ratio*100:.1f}%)")

        return train_df, val_df, test_df

    def split_by_date(
        self,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        test_start: str,
        test_end: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data using the manuscript periods instead of sample ratios."""
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("Date splitting requires a DatetimeIndex")

        df = self.df.sort_index()
        train_df = df.loc[pd.Timestamp(train_start):pd.Timestamp(train_end)]
        val_df = df.loc[pd.Timestamp(val_start):pd.Timestamp(val_end)]
        test_df = df.loc[pd.Timestamp(test_start):pd.Timestamp(test_end)]

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(
                "One or more date-based splits are empty. Check the configured periods."
            )

        print("\nDataset split by manuscript periods:")
        print(f"Train: {train_df.index.min().date()} to {train_df.index.max().date()} ({len(train_df)} samples)")
        print(f"Validation: {val_df.index.min().date()} to {val_df.index.max().date()} ({len(val_df)} samples)")
        print(f"Test: {test_df.index.min().date()} to {test_df.index.max().date()} ({len(test_df)} samples)")
        return train_df, val_df, test_df

    def fit_scalers(self, train_df: pd.DataFrame):
        print("\nFitting scalers...")

        self.meteo_scaler = StandardScaler()
        self.meteo_scaler.fit(train_df[self.meteo_features])

        self.gate_scaler = StandardScaler()
        self.gate_scaler.fit(train_df[self.gate_features])

        self.target_scaler = StandardScaler()
        self.target_scaler.fit(train_df[[self.target_col]])

        print("Scalers fitted successfully")

    def transform(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.meteo_scaler is None:
            raise ValueError("Scalers not fitted, please call fit_scalers() first")

        meteo_data = self.meteo_scaler.transform(df[self.meteo_features])
        gate_data = self.gate_scaler.transform(df[self.gate_features])
        target_data = self.target_scaler.transform(df[[self.target_col]])

        return meteo_data, gate_data, target_data

    def inverse_transform_target(self, target_scaled: np.ndarray) -> np.ndarray:
        if self.target_scaler is None:
            raise ValueError("Target scaler not fitted")

        if target_scaled.ndim == 1:
            target_scaled = target_scaled.reshape(-1, 1)

        return self.target_scaler.inverse_transform(target_scaled)


def create_sequences(
    meteo_data: np.ndarray,
    gate_data: np.ndarray,
    target_data: np.ndarray,
    seq_len: int,
    label_len: int,
    pred_len: int,
    stride: int = 1,
    include_target_in_input: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if seq_len <= 0 or pred_len <= 0:
        raise ValueError("seq_len and pred_len must be positive")
    if label_len <= 0 or label_len > seq_len:
        raise ValueError("label_len must be in the range [1, seq_len]")

    print(f"\nCreating temporal sequences...")
    print(f"seq_len={seq_len}, label_len={label_len}, pred_len={pred_len}, stride={stride}")
    print(f"include_target_in_input={include_target_in_input} (whether to include historical runoff as input)")

    X_meteo_list = []
    X_gate_list = []
    y_list = []

    n_samples = len(meteo_data)
    total_len = seq_len + pred_len

    for i in range(0, n_samples - total_len + 1, stride):
        if include_target_in_input:
            enc_meteo = meteo_data[i:i + seq_len]
            enc_target = target_data[i:i + seq_len]
            enc_x = np.concatenate([enc_meteo, enc_target], axis=1)
        else:
            enc_x = meteo_data[i:i + seq_len]

        dec_meteo_historical = meteo_data[i + seq_len - label_len:i + seq_len]
        dec_meteo_future = np.zeros((pred_len, meteo_data.shape[1]))

        if include_target_in_input:
            dec_target_historical = target_data[i + seq_len - label_len:i + seq_len]
            dec_target_future = np.zeros((pred_len, 1))

            dec_x_historical = np.concatenate([dec_meteo_historical, dec_target_historical], axis=1)
            dec_x_future = np.concatenate([dec_meteo_future, dec_target_future], axis=1)
        else:
            dec_x_historical = dec_meteo_historical
            dec_x_future = dec_meteo_future

        dec_x = np.vstack([dec_x_historical, dec_x_future])

        # The physical gate uses only information available at the forecast origin.
        # The full encoder history is retained for the 3-day and 7-day memories.
        gate_x = gate_data[i:i + seq_len]

        y = target_data[i + seq_len:i + seq_len + pred_len]

        meteo_input = {
            'enc_x': enc_x,
            'dec_x': dec_x
        }

        X_meteo_list.append(meteo_input)
        X_gate_list.append(gate_x)
        y_list.append(y)

    print(f"Created {len(X_meteo_list)} sequence samples")
    if include_target_in_input:
        print(f"Input feature count: meteo features({meteo_data.shape[1]}) + historical runoff(1) = {meteo_data.shape[1] + 1}")

    return X_meteo_list, X_gate_list, y_list


if __name__ == "__main__":

    print("=" * 60)
    print("Testing data preprocessing module")
    print("=" * 60)

    DATA_PATH = 'data/huanyuankou.xlsx'
    METEO_FEATURES = ['petH', 'prec', 'pres', 'rhu', 'tmax', 'tmin', 'wind', 'SOLARmj_m2']
    GATE_FEATURES = ['prec', 'SW_INITmm', 'PETmm']
    TARGET_COL = 'runoff'

    print("\nTest 1: Load data")
    preprocessor = RunoffDataPreprocessor(
        data_path=DATA_PATH,
        meteo_features=METEO_FEATURES,
        gate_features=GATE_FEATURES,
        target_col=TARGET_COL
    )

    df = preprocessor.load_data()

    print("\nTest 2: Preprocessing")
    df = preprocessor.preprocess(handle_missing='interpolate')

    print("\nTest 3: Split dataset")
    train_df, val_df, test_df = preprocessor.split_train_val_test(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    print("\nTest 4: Standardization")
    preprocessor.fit_scalers(train_df)

    train_meteo, train_gate, train_target = preprocessor.transform(train_df)
    print(f"Train meteo data shape: {train_meteo.shape}")
    print(f"Train gate data shape: {train_gate.shape}")
    print(f"Train target data shape: {train_target.shape}")

    print("\nTest 5: Create sequences")
    X_meteo, X_gate, y = create_sequences(
        train_meteo, train_gate, train_target,
        seq_len=96,
        label_len=48,
        pred_len=1,
        stride=1
    )

    print(f"Number of sequences created: {len(X_meteo)}")
    print(f"Encoder input shape: {X_meteo[0]['enc_x'].shape}")
    print(f"Decoder input shape: {X_meteo[0]['dec_x'].shape}")
    print(f"Gate input shape: {X_gate[0].shape}")
    print(f"Target output shape: {y[0].shape}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
