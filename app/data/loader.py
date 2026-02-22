import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "latitude",
    "longitude",
    "temperature",
    "precipitation",
    "visibility",
    "accident",
}


def load_accidents_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Some timestamps could not be parsed. Fix your CSV.")

    # Normalize types
    df["accident"] = df["accident"].astype(int)

    # Sort for time-series features
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df