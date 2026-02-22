import pandas as pd
import numpy as np

POS_EDGE_HOURLY = "data/edge_hourly.csv"       # positives-only
WEATHER = "data/weather_hourly.csv"
OUT = "data/edge_train_compact.csv"

NEG_PER_POS = 25          # 30 negatives per positive (tune 10-50)
SEED = 42
LAGS = [1, 3, 6]

def main():
    rng = np.random.default_rng(SEED)

    pos = pd.read_csv(POS_EDGE_HOURLY)
    pos["hour"] = pd.to_datetime(pos["hour"], errors="coerce")
    pos = pos.dropna(subset=["hour"])
    pos["accidents"] = pd.to_numeric(pos["accidents"], errors="coerce").fillna(0).astype(int)

    # Keep only true positives
    pos = pos[pos["accidents"] > 0].copy()
    pos["y"] = 1

    w = pd.read_csv(WEATHER)
    w["hour"] = pd.to_datetime(w["hour"], errors="coerce")
    w = w.dropna(subset=["hour"]).sort_values("hour")

    hours = w["hour"].unique()
    edges = pos["edge_id"].astype(str).unique()

    pos_pairs = set(zip(pos["edge_id"].astype(str), pos["hour"].astype("datetime64[ns]")))

    n_neg_target = NEG_PER_POS * len(pos)
    neg_edges = rng.choice(edges, size=n_neg_target, replace=True)
    neg_hours = rng.choice(hours, size=n_neg_target, replace=True)

    neg = pd.DataFrame({"edge_id": neg_edges, "hour": pd.to_datetime(neg_hours)})
    neg["accidents"] = 0
    neg["y"] = 0

    # Remove collisions with positives + duplicates
    keep = []
    for e, h in zip(neg["edge_id"], neg["hour"]):
        keep.append((e, h.to_datetime64()) not in pos_pairs)
    neg = neg.loc[keep].drop_duplicates(subset=["edge_id","hour"])

    # Merge weather
    pos2 = pos.merge(w, on="hour", how="left")
    neg2 = neg.merge(w, on="hour", how="left")

    df = pd.concat([pos2, neg2], ignore_index=True).sort_values(["edge_id","hour"])

    # Fill weather safely
    for col in ["temperature","precipitation","visibility","wind_speed","dewpoint"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].ffill().bfill().fillna(0)

    # Lag features (computed within sampled rows)
    for lag in LAGS:
        df[f"accidents_lag_{lag}"] = df.groupby("edge_id")["accidents"].shift(lag).fillna(0)

    df["hour_of_day"] = df["hour"].dt.hour
    df["dow"] = df["hour"].dt.dayofweek
    df["month"] = df["hour"].dt.month

    df.to_csv(OUT, index=False)
    print("Saved ->", OUT)
    print("Rows:", len(df))
    print("y distribution:", df["y"].value_counts().to_dict())

if __name__ == "__main__":
    main()
