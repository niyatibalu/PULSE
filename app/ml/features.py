from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


LAG_HOURS = (1, 3, 6)


def cyclical_encode(x: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
    rad = 2.0 * np.pi * x / period
    return np.sin(rad), np.cos(rad)


@dataclass
class FeatureSpec:
    feature_cols: List[str]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = out["timestamp"]

    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek

    out["hour_sin"], out["hour_cos"] = cyclical_encode(out["hour"], 24)
    out["dow_sin"], out["dow_cos"] = cyclical_encode(out["dow"], 7)
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("timestamp")

    for lag in LAG_HOURS:
        out[f"accident_lag_{lag}"] = (
            out.groupby("grid_id")["accident"].shift(lag).fillna(0).astype(float)
        )
    return out


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, FeatureSpec]:
    out = df.copy()
    out = add_time_features(out)
    out = add_lag_features(out)

    feature_cols = [
        "temperature",
        "precipitation",
        "visibility",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "accident_lag_1",
        "accident_lag_3",
        "accident_lag_6",
    ]

    X = out[feature_cols].astype(float)
    y = out["accident"].astype(int)

    return X, y, FeatureSpec(feature_cols=feature_cols)