import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

raise SystemExit("DEPRECATED: Use make_edge_train_compact.py + train_edge_model_compact.py")


INFILE = "data/edge_train.csv"
MODEL_OUT = "models/edge_model.joblib"
PRED_OUT = "data/edge_predictions.csv"

MAX_ROWS = 300_000     # cap total rows for speed
TEST_FRAC = 0.20       # last 20% of rows as test (time-ordered)
RANDOM_SEED = 42

FEATURE_CANDIDATES = [
    "temperature","precipitation","visibility","wind_speed","dewpoint",
    "hour_of_day","dow","month",
    "accidents_lag_1","accidents_lag_3","accidents_lag_6",
    "mean_speed_kph","mean_travel_time_s","mean_length_m"
]

def main():
    df = pd.read_csv(INFILE)
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"]).sort_values("hour")

    if len(df) > MAX_ROWS:
        df = df.iloc[-MAX_ROWS:].copy()

    if "y" not in df.columns:
        raise ValueError("edge_train.csv missing 'y'")

    y = df["y"].astype(int)
    vc = y.value_counts()
    print("Rows:", len(df), "y distribution:", vc.to_dict())
    if len(vc) < 2:
        raise ValueError("Only one class in y. Cannot train.")

    feature_cols = [c for c in FEATURE_CANDIDATES if c in df.columns]
    if not feature_cols:
        raise ValueError("No feature columns found.")

    X = df[feature_cols].fillna(0)

    # Time-ordered row split (always non-empty)
    n = len(df)
    n_test = max(1, int(TEST_FRAC * n))
    n_train = n - n_test
    if n_train < 1:
        raise ValueError("Train split ended up empty. Reduce TEST_FRAC or increase MAX_ROWS.")

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test, y_test = X.iloc[n_train:], y.iloc[n_train:]
    print("Train rows:", len(X_train), "Test rows:", len(X_test))

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=RANDOM_SEED
    )

    print("Fitting model...")
    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_test) if y_test.nunique() == 2 else float("nan")
    print("Holdout ROC-AUC:", auc)

    Path("models").mkdir(exist_ok=True)
    dump({"model": model, "feature_cols": feature_cols}, MODEL_OUT)

    df_out = df[["edge_id","hour","accidents","y"]].copy()
    df_out["risk"] = model.predict_proba(X)[:, 1]
    df_out.to_csv(PRED_OUT, index=False)

    print("Saved model ->", MODEL_OUT)
    print("Saved preds  ->", PRED_OUT)
    print(df_out.sort_values("risk", ascending=False).head(10))

if __name__ == "__main__":
    main()