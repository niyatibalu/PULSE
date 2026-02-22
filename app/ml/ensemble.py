from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


@dataclass
class EnsembleBundle:
    models: List[BaseEstimator]
    feature_cols: List[str]
    lat_edges: np.ndarray
    lon_edges: np.ndarray


def ensemble_predict_proba(bundle: EnsembleBundle, X: pd.DataFrame) -> np.ndarray:
    X_use = X[bundle.feature_cols]
    probs = []
    for m in bundle.models:
        probs.append(m.predict_proba(X_use)[:, 1])
    return np.vstack(probs)  # shape: (n_models, n_rows)


def ensemble_mean_std(bundle: EnsembleBundle, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probs = ensemble_predict_proba(bundle, X)
    return probs.mean(axis=0), probs.std(axis=0)