import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from . import features as feat
import statsmodels.api as sm

def seasonal_naive_forecast(history: pd.Series, horizon: int, season: int = 7) -> np.ndarray:
    """history: complaints indexed by consecutive days, most recent last."""
    last_season = history.values[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, reps)[:horizon]

def holt_winters_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    model = ExponentialSmoothing(
        history.values,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=7,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    fc = fit.forecast(horizon)
    return np.clip(fc, a_min=0, a_max=None)

def _fit_annual_seasonal_profile(dates: pd.Series, values: pd.Series, harmonics: int = 3):
    doy = dates.dt.dayofyear.values.astype(float)
    trend_idx = (dates - dates.min()).dt.days.values.astype(float)
    dow = dates.dt.dayofweek.values

    X = {"const": np.ones(len(dates)), "trend": trend_idx}
    for k in range(1, harmonics + 1):
        X[f"sin_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        X[f"cos_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    dow_dummies = pd.get_dummies(dow, prefix="dow", drop_first=True).astype(float)
    Xdf = pd.DataFrame(X)
    Xdf = pd.concat([Xdf, dow_dummies.reset_index(drop=True)], axis=1)

    y = np.log1p(values.values)
    ols = sm.OLS(y, Xdf.astype(float)).fit()

    fourier_cols = [c for c in Xdf.columns if c.startswith(("sin_", "cos_"))]
    coefs = ols.params[fourier_cols]

    def seasonal_log_effect(day_of_year: np.ndarray) -> np.ndarray:
        out = np.zeros(len(day_of_year), dtype=float)
        for k in range(1, harmonics + 1):
            out += coefs[f"sin_{k}"] * np.sin(2 * np.pi * k * day_of_year / 365.25)
            out += coefs[f"cos_{k}"] * np.cos(2 * np.pi * k * day_of_year / 365.25)
        return out

    return seasonal_log_effect


def holt_winters_seasonal_corrected_forecast(history_df: pd.DataFrame, horizon: int) -> np.ndarray:
    hw_fc = holt_winters_forecast(history_df.set_index("date")["complaints"], horizon)

    seasonal_fn = _fit_annual_seasonal_profile(history_df["date"], history_df["complaints"])

    last_date = history_df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    s_future = seasonal_fn(future_dates.dayofyear.values.astype(float))
    s_last = seasonal_fn(np.array([float(last_date.dayofyear)]))[0]

    # multiplicative correction relative to the seasonal position of the
    # cutoff date, since HW's own trend/level already reflects "now"
    correction = np.exp(s_future - s_last)
    corrected = hw_fc * correction
    return np.clip(corrected, a_min=0, a_max=None)


LGB_PARAMS = dict(
    #keep it shallow for now, not much data
    #from experience these params work for low data with high seasonlity
    objective="poisson",
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=7,
    max_depth=3,
    min_child_samples=30,
    subsample=0.7,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=3.0,
    reg_alpha=1.0,
    random_state=42,
    verbosity=-1,
)

def _prep_for_lgb(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in feat.CATEGORICAL_FEATURES:
        df[c] = df[c].astype("category")
    return df

def fit_lightgbm(train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> lgb.LGBMRegressor:
    X = _prep_for_lgb(train_df[feat.FEATURE_COLUMNS])
    y = train_df["complaints"]
    model = lgb.LGBMRegressor(**LGB_PARAMS)

    fit_kwargs = {}
    if val_df is not None and len(val_df) > 0:
        X_val = _prep_for_lgb(val_df[feat.FEATURE_COLUMNS])
        y_val = val_df["complaints"]
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(30, verbose=False)]

    model.fit(X, y, **fit_kwargs)
    return model


def recursive_lightgbm_forecast(
    model: lgb.LGBMRegressor,
    history_df: pd.DataFrame,
    horizon: int,
    origin_date: pd.Timestamp,
) -> pd.DataFrame:
    max_lookback = max(feat.LAGS + feat.ROLL_WINDOWS)
    working = history_df[["date", "complaints"]].copy()

    last_date = working["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # deterministic calendar attributes for the future dates
    future_calendar = pd.DataFrame({"date": future_dates})
    future_calendar["is_weekend"] = future_calendar["date"].dt.dayofweek.isin([5, 6]).astype(int)
    import holidays as _holidays
    yrs = sorted(set(future_calendar["date"].dt.year))
    uk = _holidays.UnitedKingdom(subdiv="Wales", years=yrs)
    hol_dates = set(uk.keys())
    future_calendar["bank_holiday_flag"] = future_calendar["date"].dt.date.isin(hol_dates).astype(int)

    preds = []
    for i, row in future_calendar.iterrows():
        cur_date = row["date"]
        # tail of known+predicted series, used to compute lag/rolling features
        tail = working.tail(max_lookback + 1).copy()
        tail = pd.concat(
            [tail, pd.DataFrame({"date": [cur_date], "complaints": [np.nan]})],
            ignore_index=True,
        )
        tail["is_weekend"] = row["is_weekend"]
        tail["bank_holiday_flag"] = row["bank_holiday_flag"]
        tail = feat.add_calendar_features(tail, origin_date)
        tail = feat.add_fourier_features(tail)
        tail = feat.add_lag_features(tail)

        feat_row = _prep_for_lgb(tail.iloc[[-1]][feat.FEATURE_COLUMNS])
        yhat = float(model.predict(feat_row)[0])
        yhat = max(yhat, 0.0)
        preds.append(yhat)

        working = pd.concat(
            [working, pd.DataFrame({"date": [cur_date], "complaints": [yhat]})],
            ignore_index=True,
        )

    return pd.DataFrame({"date": future_dates, "forecast": preds})
