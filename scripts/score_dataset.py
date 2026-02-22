# scripts/score_dataset.py
import pandas as pd
from pathlib import Path
from joblib import load

from app.config import settings
from app.data.loader import load_accidents_csv
from app.ml.spatial import GridSpec, assign_grid
from app.ml.features import build_feature_matrix
from app.ml.ensemble import ensemble_mean_std


def main() -> None:
    # Load trained ensemble bundle
    bundle = load(settings.model_path)

    # Load raw data
    df_raw = load_accidents_csv(settings.data_path)

    # Assign the SAME grid used in training
    grid = GridSpec(lat_edges=bundle.lat_edges, lon_edges=bundle.lon_edges)
    df = assign_grid(df_raw, grid)

    # Build feature matrix (adds time + lag features)
    X, y, spec = build_feature_matrix(df)

    # Predict mean/std from ensemble
    risk_mean, risk_uncertainty = ensemble_mean_std(bundle, X)

    # Save scored dataset for dashboard
    df_out = df.copy()
    df_out["risk_mean"] = risk_mean
    df_out["risk_uncertainty"] = risk_uncertainty

    out_path = Path("data") / "scored.csv"
    df_out.to_csv(out_path, index=False)

    print(f"Saved: {out_path.resolve()}")
    print(df_out[["risk_mean", "risk_uncertainty"]].describe())


if __name__ == "__main__":
    main()