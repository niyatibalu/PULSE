from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from app.ml.ensemble import EnsembleBundle
from app.ml.spatial import GridSpec, fit_grid, assign_grid
from app.ml.features import build_feature_matrix


def _base_model(random_state: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state,
    )


def time_series_auc(X: pd.DataFrame, y: pd.Series, seed: int) -> float:
    tscv = TimeSeriesSplit(n_splits=5)
    aucs: List[float] = []
    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        m = _base_model(seed + fold)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
    return float(np.mean(aucs))


def train_ensemble(
    df_raw: pd.DataFrame,
    lat_bins: int,
    lon_bins: int,
    n_models: int,
    seed: int,
) -> Tuple[EnsembleBundle, float]:
    # Fit grid on training data
    grid = fit_grid(df_raw, lat_bins=lat_bins, lon_bins=lon_bins)
    df = assign_grid(df_raw, grid)

    X, y, spec = build_feature_matrix(df)

    # Evaluate with time-series CV on a single representative model
    auc = time_series_auc(X, y, seed)

    # Train bootstrapped ensemble
    rng = np.random.default_rng(seed)
    models = []
    n = len(X)

    for i in range(n_models):
        idx = rng.integers(0, n, size=n, endpoint=False)
        m = _base_model(seed + 1000 + i)
        m.fit(X.iloc[idx], y.iloc[idx])
        models.append(m)

    bundle = EnsembleBundle(
        models=models,
        feature_cols=spec.feature_cols,
        lat_edges=grid.lat_edges,
        lon_edges=grid.lon_edges,
    )
    return bundle, auc


def save_bundle(bundle: EnsembleBundle, path: str) -> None:
    dump(bundle, path)