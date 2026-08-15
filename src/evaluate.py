import numpy as np
import pandas as pd
from . import features as feat
from . import models as mdl

def _metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    err = forecast - actual
    abs_err = np.abs(err)
    mae = abs_err.mean()
    rmse = np.sqrt((err ** 2).mean())
    smape = (200 * abs_err / (np.abs(actual) + np.abs(forecast) + 1e-8)).mean()
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape}

def make_cutoffs(df: pd.DataFrame, horizon: int, n_cutoffs: int, min_train_days: int) -> list[pd.Timestamp]:
    last_possible = df["date"].max() - pd.Timedelta(days=horizon)
    earliest = df["date"].min() + pd.Timedelta(days=min_train_days)
    if earliest > last_possible:
        raise ValueError("Not enough history for the requested min_train_days/horizon.")
    cutoffs = pd.date_range(earliest, last_possible, periods=n_cutoffs)
    all_dates = df["date"].values
    return [pd.Timestamp(all_dates[np.argmin(np.abs(all_dates - c.to_datetime64()))]) for c in cutoffs]

def backtest(df: pd.DataFrame, horizon: int = 90, n_cutoffs: int = 5,
             min_train_days: int = 500, n_val: int = 60) -> pd.DataFrame:
    origin = df["date"].min()
    maxlag = max(feat.LAGS + feat.ROLL_WINDOWS)
    cutoffs = make_cutoffs(df, horizon, n_cutoffs, max(min_train_days, maxlag + n_val + 30))

    records = []
    for cutoff in cutoffs:
        hist = df[df["date"] <= cutoff].reset_index(drop=True)
        future = df[(df["date"] > cutoff) & (df["date"] <= cutoff + pd.Timedelta(days=horizon))].reset_index(drop=True)
        if len(future) < horizon:
            continue

        actual = future["complaints"].values
        dates = future["date"].values

        sn = mdl.seasonal_naive_forecast(hist.set_index("date")["complaints"], horizon)
        hw = mdl.holt_winters_forecast(hist.set_index("date")["complaints"], horizon)
        hw_sc = mdl.holt_winters_seasonal_corrected_forecast(hist[["date", "complaints"]], horizon)

        fdf_hist = feat.build_feature_frame(hist, origin)
        train_hist = fdf_hist.iloc[maxlag:].reset_index(drop=True)
        tr, val = train_hist.iloc[:-n_val], train_hist.iloc[-n_val:]
        model = mdl.fit_lightgbm(tr, val)
        lgb_fc = mdl.recursive_lightgbm_forecast(model, hist, horizon, origin)["forecast"].values

        for name, fc in [("seasonal_naive", sn), ("holt_winters", hw),
                         ("holt_winters_seasonal_corrected", hw_sc), ("lightgbm", lgb_fc)]:
            for h, (d, a, f) in enumerate(zip(dates, actual, fc), start=1):
                records.append({
                    "cutoff": cutoff, "model": name, "horizon_day": h,
                    "date": d, "actual": a, "forecast": f,
                })

    return pd.DataFrame.from_records(records)


def summarise(backtest_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, g in backtest_df.groupby("model"):
        m = _metrics(g["actual"].values, g["forecast"].values)
        m["model"] = model_name
        m["bucket"] = "overall (1-90d)"
        rows.append(m)
        buckets = [(1, 7, "1-7d"), (8, 30, "8-30d"), (31, 90, "31-90d")]
        for lo, hi, label in buckets:
            sub = g[(g["horizon_day"] >= lo) & (g["horizon_day"] <= hi)]
            m2 = _metrics(sub["actual"].values, sub["forecast"].values)
            m2["model"] = model_name
            m2["bucket"] = label
            rows.append(m2)
    out = pd.DataFrame(rows)[["model", "bucket", "MAE", "RMSE", "sMAPE"]]
    return out.sort_values(["bucket", "model"]).reset_index(drop=True)


def residual_quantiles_by_bucket(backtest_df: pd.DataFrame, model_name: str = "lightgbm") -> pd.DataFrame:
    g = backtest_df[backtest_df["model"] == model_name].copy()
    g["resid"] = g["forecast"] - g["actual"]
    buckets = [(1, 7, "1-7d"), (8, 30, "8-30d"), (31, 60, "31-60d"), (61, 90, "61-90d")]
    rows = []
    for lo, hi, label in buckets:
        sub = g[(g["horizon_day"] >= lo) & (g["horizon_day"] <= hi)]
        rows.append({
            "bucket": label, "horizon_lo": lo, "horizon_hi": hi,
            "q10": sub["resid"].quantile(0.10),
            "q90": sub["resid"].quantile(0.90),
        })
    return pd.DataFrame(rows)
