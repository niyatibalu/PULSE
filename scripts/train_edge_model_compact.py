import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

INFILE = "data/edge_train_compact.csv"
MODEL_OUT = "models/edge_model.joblib"
PRED_OUT = "data/edge_predictions.csv"

FEATURES = [
    "temperature","precipitation","visibility","wind_speed","dewpoint",
    "hour_of_day","dow","month",
    "accidents_lag_1","accidents_lag_3","accidents_lag_6",
]

TEST_FRAC = 0.2

def main():
    df = pd.read_csv(INFILE)
    df["hour"] = pd.to_datetime(df["hour"])
    df = df.sort_values("hour")

    X = df[[c for c in FEATURES if c in df.columns]].fillna(0)
    y = df["y"].astype(int)

    n = len(df)
    n_test = int(TEST_FRAC * n)
    n_train = n - n_test

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test, y_test = X.iloc[n_train:], y.iloc[n_train:]

    model = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p) if y_test.nunique() == 2 else float("nan")
    print("Holdout AUC:", auc)

    Path("models").mkdir(exist_ok=True)
    dump({"model": model, "feature_cols": X.columns.tolist()}, MODEL_OUT)

    out = df[["edge_id","hour","accidents","y"]].copy()
    out["risk"] = model.predict_proba(X)[:, 1]
    out.to_csv(PRED_OUT, index=False)

    print("Saved ->", MODEL_OUT)
    print("Saved ->", PRED_OUT)
    print(out.sort_values("risk", ascending=False).head(10))

if __name__ == "__main__":
    main()
