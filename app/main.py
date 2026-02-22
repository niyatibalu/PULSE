import logging
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load

from app.config import settings
from app.logging_config import configure_logging
from app.schemas import PredictRow, PredictResponse, ExplainResponse
from app.ml.ensemble import EnsembleBundle, ensemble_mean_std
from app.ml.spatial import GridSpec, assign_grid
from app.ml.features import add_time_features
from app.ml.explain import make_explainer, explain_row, ExplainerBundle

app = FastAPI(title="Urban Safety & Resource AI")


@lru_cache(maxsize=1)
def get_bundle() -> EnsembleBundle:
    bundle: EnsembleBundle = load(settings.model_path)
    return bundle


@lru_cache(maxsize=1)
def get_explainer() -> ExplainerBundle:
    # Small background for SHAP, based on a synthetic neutral sample.
    # For best results: replace with a real background sample from training data.
    bundle = get_bundle()
    bg = pd.DataFrame([{c: 0.0 for c in bundle.feature_cols} for _ in range(50)])
    return make_explainer(bundle, bg)


def row_to_features(row: PredictRow) -> pd.DataFrame:
    bundle = get_bundle()
    grid = GridSpec(lat_edges=bundle.lat_edges, lon_edges=bundle.lon_edges)

    df = pd.DataFrame([{
        "timestamp": row.timestamp,
        "latitude": row.latitude,
        "longitude": row.longitude,
        "temperature": row.temperature,
        "precipitation": row.precipitation,
        "visibility": row.visibility,
        "accident": 0,  # placeholder for lag pipeline expectations
    }])

    df = assign_grid(df, grid)
    df = add_time_features(df)

    # Lags come from request for now (or from a real feature store)
    df["accident_lag_1"] = float(row.accident_lag_1 or 0.0)
    df["accident_lag_3"] = float(row.accident_lag_3 or 0.0)
    df["accident_lag_6"] = float(row.accident_lag_6 or 0.0)

    # Construct X with exact feature columns
    X = df[get_bundle().feature_cols].astype(float)
    return X


@app.on_event("startup")
def startup() -> None:
    configure_logging()
    log = logging.getLogger("api")
    _ = get_bundle()
    _ = get_explainer()
    log.info("API started. Model loaded from %s", settings.model_path)


@app.post("/predict", response_model=PredictResponse)
def predict(row: PredictRow) -> PredictResponse:
    try:
        X = row_to_features(row)
        mean, std = ensemble_mean_std(get_bundle(), X)
        return PredictResponse(risk_mean=float(mean[0]), risk_uncertainty=float(std[0]))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain", response_model=ExplainResponse)
def explain(row: PredictRow) -> ExplainResponse:
    try:
        X = row_to_features(row)
        top = explain_row(get_explainer(), X)
        return ExplainResponse(top_features=top)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))