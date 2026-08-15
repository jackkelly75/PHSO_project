import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 7, 14, 21, 28, 35, 364, 371]
ROLL_WINDOWS = [7, 14, 28]
FOURIER_HARMONICS = 3
ANNUAL_PERIOD = 365.25

def add_calendar_features(df: pd.DataFrame, origin_date: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["trend"] = (df["date"] - origin_date).dt.days  # integer day index
    return df

def add_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t = df["day_of_year"].values.astype(float)
    for k in range(1, FOURIER_HARMONICS + 1):
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * t / ANNUAL_PERIOD)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * t / ANNUAL_PERIOD)
    return df

def add_lag_features(df: pd.DataFrame, target_col: str = "complaints") -> pd.DataFrame:
    df = df.copy()
    for lag in LAGS:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    for w in ROLL_WINDOWS:
        # shift(1) first so the window never includes the current day
        df[f"roll_mean_{w}"] = df[target_col].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df[target_col].shift(1).rolling(w).std()
    return df

FEATURE_COLUMNS = (
    ["dow", "month", "is_weekend", "bank_holiday_flag",
     "is_month_start", "is_month_end", "trend"]
    + [f"fourier_sin_{k}" for k in range(1, FOURIER_HARMONICS + 1)]
    + [f"fourier_cos_{k}" for k in range(1, FOURIER_HARMONICS + 1)]
    + [f"lag_{lag}" for lag in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
)

CATEGORICAL_FEATURES = ["dow", "month"]

def build_feature_frame(df: pd.DataFrame, origin_date: pd.Timestamp | None = None) -> pd.DataFrame:
    if origin_date is None:
        origin_date = df["date"].min()
    df = add_calendar_features(df, origin_date)
    df = add_fourier_features(df)
    df = add_lag_features(df)
    return df
