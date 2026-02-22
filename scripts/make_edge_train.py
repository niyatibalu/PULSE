import pandas as pd

EDGE_IN = "data/edge_hourly_panel.csv"
WEATHER_IN = "data/weather_hourly.csv"
OUT = "data/edge_train.csv"

LAGS = [1, 3, 6]

def main():
    df = pd.read_csv(EDGE_IN)
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df["accidents"] = pd.to_numeric(df["accidents"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["hour"]).sort_values(["edge_id","hour"])

    w = pd.read_csv(WEATHER_IN)
    w["hour"] = pd.to_datetime(w["hour"], errors="coerce")
    w = w.dropna(subset=["hour"]).sort_values("hour")

    df = df.merge(w, on="hour", how="left")

    # Fill numeric weather gaps safely
    for col in ["temperature","precipitation","visibility","wind_speed","dewpoint"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Wind is missing in your file: fill with 0
    if "wind_speed" in df.columns:
        df["wind_speed"] = df["wind_speed"].fillna(0)

    # Other columns: forward/back fill then remaining -> 0
    for col in ["temperature","precipitation","visibility","dewpoint"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(0)

    # Label
    df["y"] = (df["accidents"] > 0).astype(int)

    # Lags
    for lag in LAGS:
        df[f"accidents_lag_{lag}"] = df.groupby("edge_id")["accidents"].shift(lag).fillna(0)

    # Time features
    df["hour_of_day"] = df["hour"].dt.hour
    df["dow"] = df["hour"].dt.dayofweek
    df["month"] = df["hour"].dt.month

    df.to_csv(OUT, index=False)
    print("Saved ->", OUT)
    print("y distribution:", df["y"].value_counts().to_dict())
    print("accidents min/max:", df["accidents"].min(), df["accidents"].max())

if __name__ == "__main__":
    main()