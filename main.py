"""
single entry point for the complaints forecasting pipeline

  1. Load & clean the raw daily extract (src/data_loader.py).
  2. Rolling-origin backtest three candidate models at a 90-day horizon
     (src/evaluate.py)
  3. Select the primary model = whichever has the lowest overall backtest
     MAE
  4. Refit the selected model on the complete data and forecast the 90 days
     after the last date in the dataset
  5. Save the forecast to outputs/forecasts/forecast_90d.csv and produce
     two charts under outputs/plots/

"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())
from src.data_loader import load_clean
from src import features as feat
from src import models as mdl
from src import evaluate as ev

DATA_PATH = "data/Principle_Data_Scientist_Tech_Assessment.xlsx"
SHEET = 'daily records'
HORIZON = 90
N_CUTOFFS = 5
MIN_TRAIN_DAYS = 500
N_VAL = 60
primary_model = "holt_winters_seasonal_corrected"

def fit_and_forecast(model_name: str, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    origin = df["date"].min()
    if model_name == "holt_winters":
        fc = mdl.holt_winters_forecast(df.set_index("date")["complaints"], horizon)
    elif model_name == "holt_winters_seasonal_corrected":
        fc = mdl.holt_winters_seasonal_corrected_forecast(df[["date", "complaints"]], horizon)
    elif model_name == "lightgbm":
        maxlag = max(feat.LAGS + feat.ROLL_WINDOWS)
        fdf = feat.build_feature_frame(df, origin)
        train = fdf.iloc[maxlag:].reset_index(drop=True)
        tr, val = train.iloc[:-N_VAL], train.iloc[-N_VAL:]
        model = mdl.fit_lightgbm(tr, val)
        fc = mdl.recursive_lightgbm_forecast(model, df, horizon, origin)["forecast"].values
    elif model_name == "seasonal_naive":
        fc = mdl.seasonal_naive_forecast(df.set_index("date")["complaints"], horizon)
    else:
        raise ValueError(model_name)
    last_date = df["date"].max()
    dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame({"date": dates, "forecast": np.clip(fc, 0, None)})


def add_prediction_interval(forecast_df: pd.DataFrame, backtest_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rq = ev.residual_quantiles_by_bucket(backtest_df, model_name=model_name)
    forecast_df = forecast_df.copy()
    forecast_df["horizon_day"] = np.arange(1, len(forecast_df) + 1)

    def bucket_for(h):
        row = rq[(rq["horizon_lo"] <= h) & (h <= rq["horizon_hi"])]
        return row.iloc[0] if len(row) else rq.iloc[-1]

    lowers, uppers = [], []
    for h, point in zip(forecast_df["horizon_day"], forecast_df["forecast"]):
        b = bucket_for(h)
        lowers.append(max(0, point - b["q90"]))
        uppers.append(point - b["q10"])
    forecast_df["lower_80"] = lowers
    forecast_df["upper_80"] = uppers
    return forecast_df.drop(columns=["horizon_day"])

def plot_backtest_comparison(summary: pd.DataFrame):
    piv = summary[summary["bucket"] != "overall (1-90d)"].pivot(
        index="bucket", columns="model", values="MAE"
    ).reindex(["1-7d", "8-30d", "31-90d"])
    ax = piv.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("MAE (complaints/day)")
    ax.set_title("Rolling-origin backtest: MAE by horizon bucket")
    plt.tight_layout()
    plt.savefig("outputs/plots/backtest_comparison.png", dpi=110)
    plt.close()

def plot_final_forecast(df: pd.DataFrame, forecast: pd.DataFrame, model_name: str):
    fig, ax = plt.subplots(figsize=(13, 5))
    recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=180)]
    ax.plot(recent["date"], recent["complaints"], label="Actual (last 180 days)", color="tab:blue")
    ax.plot(forecast["date"], forecast["forecast"], label=f"Forecast ({model_name})", color="tab:red")
    ax.fill_between(forecast["date"], forecast["lower_80"], forecast["upper_80"],
                     color="tab:red", alpha=0.15, label="80% prediction interval")
    ax.axvline(df["date"].max(), color="grey", ls="--", lw=1)
    ax.set_title("Complaints: 90-day forecast")
    ax.set_ylabel("Complaints/day")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/final_forecast.png", dpi=110)
    plt.close()

def main():
    print("1. Loading & cleaning data...")
    df = load_clean(DATA_PATH)
    print(f"   {len(df)} daily rows, {df['complaints_imputed'].sum()} imputed, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")

    print("\n2. Rolling-origin backtest (90-day horizon, "
          f"{N_CUTOFFS} cutoffs)...")
    bt = ev.backtest(df, horizon=HORIZON, n_cutoffs=N_CUTOFFS,
                      min_train_days=MIN_TRAIN_DAYS, n_val=N_VAL)
    bt.to_csv("outputs/forecasts/backtest_detail.csv", index=False)
    summary = ev.summarise(bt)
    summary.to_csv("outputs/forecasts/backtest_summary.csv", index=False)
    print(summary.to_string(index=False))

    overall = summary[summary["bucket"] == "overall (1-90d)"].set_index("model")
    best_by_mae = overall["MAE"].idxmin()#

    print(f"Selected for production forecast: '{primary_model}'")
    print(f"\n 3. Refitting '{primary_model}' on full history and forecasting "
          f"{HORIZON} days ({(df['date'].max()+pd.Timedelta(days=1)).date()} -> "
          f"{(df['date'].max()+pd.Timedelta(days=HORIZON)).date()})...")
    forecast = fit_and_forecast(primary_model, df, HORIZON)
    forecast = add_prediction_interval(forecast, bt, primary_model)
    forecast["model"] = primary_model
    forecast.to_csv("outputs/forecasts/forecast_90d.csv", index=False)
    print(forecast.head())
    print("...")
    print(forecast.tail())

    print("\n4. Plotting")
    plot_backtest_comparison(summary)
    plot_final_forecast(df, forecast, primary_model)
    print("\nDone. See outputs/forecasts/forecast_90d.csv and outputs/plots/")

if __name__ == "__main__":
    main()
