import pandas as pd
import numpy as np
import holidays

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="daily records", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def _uk_bank_holiday_flags(dates: pd.DatetimeIndex) -> np.ndarray:
    years = sorted(set(dates.year))
    uk = holidays.UnitedKingdom(subdiv='England', years=years)
    holiday_dates = set(uk.keys())
    return np.array([d.date() in holiday_dates for d in dates], dtype=int)

def reindex_to_full_calendar(df: pd.DataFrame) -> pd.DataFrame:
    full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_idx)
    df.index.name = "date"
    df = df.reset_index()
    df["is_gap_day"] = df["complaints"].isna() & df["row_id"].isna()
    return df

def clean(df: pd.DataFrame, validate_holidays: bool = True) -> pd.DataFrame:
    df = reindex_to_full_calendar(df)

    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    computed_holiday = _uk_bank_holiday_flags(pd.DatetimeIndex(df["date"]))
    if validate_holidays:
        overlap = df["bank_holiday_flag"].notna()
        agree = (df.loc[overlap, "bank_holiday_flag"].astype(int).values
                  == computed_holiday[overlap.values]).mean()
        if agree < 0.9:
            raise ValueError(
                f"UK '{UK_HOLIDAY_SUBDIVISION}' holiday calendar only agrees "
                f"with the supplied bank_holiday_flag {agree:.0%} of the time "
                "- check the holiday subdivision/country before trusting it "
                "for the forecast horizon."
            )
    df["bank_holiday_flag"] = computed_holiday

    log_complaints = np.log1p(df["complaints"])
    log_complaints = log_complaints.interpolate(method="linear", limit_direction="both")
    df["complaints_imputed"] = df["complaints"].isna()
    df["complaints"] = np.expm1(log_complaints).round().clip(lower=0).astype(int)

    for col in ["staffing_level_fte", "backlog_days", "channel_mix_index"]:
        df[col] = df[col].interpolate(method="linear", limit_direction="both")

    df["media_mentions"] = df["media_mentions"].fillna(0)

    df = df.drop(columns=["row_id", "is_gap_day"])
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(1, len(df) + 1)

    return df


def load_clean(path: str) -> pd.DataFrame:
    return clean(load_raw(path))
