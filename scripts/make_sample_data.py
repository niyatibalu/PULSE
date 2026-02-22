import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    rng = np.random.default_rng(42)

    # Generate a "Madison-ish" area bounding box
    lat_min, lat_max = 43.00, 43.15
    lon_min, lon_max = -89.55, -89.25

    # Create discrete spatial "cells" (centers) so lag features have meaning
    n_cells = 40
    cell_lats = rng.uniform(lat_min, lat_max, n_cells)
    cell_lons = rng.uniform(lon_min, lon_max, n_cells)

    # Time range: hourly for 28 days
    start = datetime(2024, 1, 1, 0, 0, 0)
    hours = 28 * 24
    timestamps = [start + timedelta(hours=h) for h in range(hours)]

    rows = []
    for ts in timestamps:
        hour = ts.hour
        dow = ts.weekday()

        # Simple weather process
        # Daily temp cycle + noise
        temperature = 25 + 10 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 2)
        # Precip: mostly 0, sometimes >0
        precip_flag = rng.random() < 0.18
        precipitation = (rng.exponential(0.2) + 0.05) if precip_flag else 0.0
        # Visibility drops with precip + noise
        visibility = max(0.5, 10 - (5.5 if precipitation > 0 else 0) + rng.normal(0, 0.8))

        # Commute indicator
        commute = 1 if (7 <= hour <= 9 or 16 <= hour <= 18) else 0
        weekendish = 1 if dow >= 4 else 0

        # Accident probability driver (logit)
        # tuned to yield a few % positives across the dataset
        logit = (
            -3.6
            + 0.9 * (1 if precipitation > 0 else 0)
            + 0.7 * commute
            + 0.25 * weekendish
            + 0.25 * (10 - visibility) / 10
        )
        p_acc = sigmoid(logit)

        # For each hour, create one row per cell (so grid_id repeats over time)
        for lat_c, lon_c in zip(cell_lats, cell_lons):
            # small jitter within a cell
            lat = lat_c + rng.normal(0, 0.002)
            lon = lon_c + rng.normal(0, 0.002)

            accident = 1 if rng.random() < p_acc else 0

            rows.append(
                {
                    "timestamp": ts.isoformat(sep=" "),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "temperature": float(temperature),
                    "precipitation": float(precipitation),
                    "visibility": float(visibility),
                    "accident": int(accident),
                }
            )

    df = pd.DataFrame(rows)

    Path("data").mkdir(exist_ok=True)
    out_path = Path("data") / "accidents.csv"
    df.to_csv(out_path, index=False)

    pos_rate = df["accident"].mean()
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Accident positive rate: {pos_rate:.4f}")

if __name__ == "__main__":
    main()