import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

INFILE = "data/edge_train_compact.csv"
WEATHER_FILE = "data/weather_hourly.csv"

MODEL_OUT = "models/edge_model.joblib"
PRED_OUT = "data/edge_predictions.csv"

FEATURES = [
    "temperature","precipitation","visibility","wind_speed","dewpoint",
    "hour_of_day","dow","month",
    "accidents_lag_1","accidents_lag_3","accidents_lag_6",
    "speed_kph","travel_time","length",
    "edge_rate_all",
]

# Hold out last N days (time-based). Will expand if too few positives.
HOLDOUT_DAYS_PRIMARY = 30
HOLDOUT_DAYS_FALLBACK = 120
MIN_HOLDOUT_POS = 50


def main():
    df = pd.read_csv(INFILE)
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"]).sort_values("hour").reset_index(drop=True)
    base = pd.read_csv("data/edge_baselines.csv")
    df = df.merge(base, on="edge_id", how="left")
    df["edge_rate_all"] = df["edge_rate_all"].fillna(0.0)
    # Restrict to hours that exist in weather file (prevents 2022 contamination)
    w = pd.read_csv(WEATHER_FILE, usecols=["hour"])
    w["hour"] = pd.to_datetime(w["hour"], errors="coerce")
    w = w.dropna(subset=["hour"])
    min_h, max_h = w["hour"].min(), w["hour"].max()
    df = df[(df["hour"] >= min_h) & (df["hour"] <= max_h)].copy()
    df = df.sort_values("hour").reset_index(drop=True)
    print(f"Training hour window: {min_h} -> {max_h} | rows={len(df):,} | positives={int(df['y'].sum()):,}")

    X = df[[c for c in FEATURES if c in df.columns]].fillna(0)
    y = df["y"].astype(int)

    # ---- time-based holdout that guarantees positives ----
    def split_by_days(days: int):
        cutoff = df["hour"].max() - pd.Timedelta(days=days)
        test_mask = df["hour"] >= cutoff
        return cutoff, test_mask

    cutoff, test_mask = split_by_days(HOLDOUT_DAYS_PRIMARY)
    pos_holdout = int(y[test_mask].sum())

    if pos_holdout < MIN_HOLDOUT_POS:
        cutoff, test_mask = split_by_days(HOLDOUT_DAYS_FALLBACK)
        pos_holdout = int(y[test_mask].sum())

    X_train, y_train = X[~test_mask], y[~test_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Holdout cutoff: {cutoff} | holdout rows={len(X_test):,} | holdout positives={pos_holdout:,} | holdout y unique={y_test.nunique()}")

    model = LogisticRegression(max_iter=3000, class_weight="balanced", n_jobs=-1)
    model.fit(X_train, y_train)

    # AUC only defined if both classes appear
    if y_test.nunique() < 2:
        auc = float("nan")
        print("Holdout AUC: nan (holdout has one class)")
    else:
        p = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, p)
        print(f"Holdout AUC: {auc:.4f}")

    Path("models").mkdir(exist_ok=True)
    dump({"model": model, "feature_cols": X.columns.tolist()}, MODEL_OUT)

    # Score all rows in the filtered window
    out = df[["edge_id","hour","accidents","y"]].copy()
    out["risk"] = model.predict_proba(X)[:, 1]
    out.to_csv(PRED_OUT, index=False)

    print("Saved ->", MODEL_OUT)
    print("Saved ->", PRED_OUT)
    print("Top risks:")
    print(out.sort_values("risk", ascending=False).head(10))


if __name__ == "__main__":
    main()