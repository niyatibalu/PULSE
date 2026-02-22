import pandas as pd
from joblib import dump, load
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

MODEL_IN = "models/edge_model.joblib"
TRAIN_IN = "data/edge_train_balanced.csv"
FULL_IN = "data/edge_train.csv"
OUT_MODEL = "models/edge_model_calibrated.joblib"
OUT_PRED = "data/edge_predictions.csv"

TEST_FRAC = 0.2  # time-ordered

def main():
    bundle = load(MODEL_IN)
    base_model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_csv(TRAIN_IN)
    df["hour"] = pd.to_datetime(df["hour"])
    df = df.sort_values("hour")

    X = df[feature_cols].fillna(0)
    y = df["y"].astype(int)

    n = len(df)
    n_test = int(TEST_FRAC * n)
    n_train = n - n_test

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_cal, y_cal = X.iloc[n_train:], y.iloc[n_train:]

    # Calibrate on the calibration slice
    calib = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
    calib.fit(X_cal, y_cal)

    p_cal = calib.predict_proba(X_cal)[:, 1]
    auc = roc_auc_score(y_cal, p_cal) if y_cal.nunique() == 2 else float("nan")
    brier = brier_score_loss(y_cal, p_cal)

    print("Calibration holdout AUC:", auc)
    print("Calibration holdout Brier:", brier)

    Path("models").mkdir(exist_ok=True)
    dump({"model": calib, "feature_cols": feature_cols}, OUT_MODEL)

    # Score full dataset for mapping
    full = pd.read_csv(FULL_IN)
    full["hour"] = pd.to_datetime(full["hour"])
    X_full = full[feature_cols].fillna(0)
    out = full[["edge_id","hour","accidents","y"]].copy()
    out["risk"] = calib.predict_proba(X_full)[:, 1]
    out.to_csv(OUT_PRED, index=False)

    print("Saved calibrated model ->", OUT_MODEL)
    print("Saved preds ->", OUT_PRED)
    print(out.sort_values("risk", ascending=False).head(10))

if __name__ == "__main__":
    main()