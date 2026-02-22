from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class GridSpec:
    lat_edges: np.ndarray
    lon_edges: np.ndarray


def fit_grid(df: pd.DataFrame, lat_bins: int, lon_bins: int) -> GridSpec:
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()

    # Slight padding so edge points are included
    lat_pad = (lat_max - lat_min) * 1e-6 + 1e-9
    lon_pad = (lon_max - lon_min) * 1e-6 + 1e-9

    lat_edges = np.linspace(lat_min - lat_pad, lat_max + lat_pad, lat_bins + 1)
    lon_edges = np.linspace(lon_min - lon_pad, lon_max + lon_pad, lon_bins + 1)

    return GridSpec(lat_edges=lat_edges, lon_edges=lon_edges)


def assign_grid(df: pd.DataFrame, grid: GridSpec) -> pd.DataFrame:
    out = df.copy()
    out["lat_bin"] = pd.cut(out["latitude"], bins=grid.lat_edges, labels=False, include_lowest=True)
    out["lon_bin"] = pd.cut(out["longitude"], bins=grid.lon_edges, labels=False, include_lowest=True)

    out["lat_bin"] = out["lat_bin"].astype("Int64")
    out["lon_bin"] = out["lon_bin"].astype("Int64")

    # If any point falls out of bounds, it becomes NA
    if out["lat_bin"].isna().any() or out["lon_bin"].isna().any():
        raise ValueError("Some points fell outside the fitted grid. Check grid edges.")

    out["grid_id"] = out["lat_bin"].astype(str) + "_" + out["lon_bin"].astype(str)
    return out