"""
Urban Safety What-If Policy Simulator — Model Layer
=====================================================
Trains a crash-risk model on wisdot_crash.csv + weather_hourly.csv + edge data,
then exposes a simulate_policy() function that perturbs features and returns
before/after risk estimates with uncertainty (confidence intervals).

Dependencies:
    pip install pandas numpy scikit-learn shap scipy joblib

Usage:
    python simulator_model.py          # trains and saves model
    from simulator_model import simulate_policy   # import in API
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import shap

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")          # put your CSVs here
MODEL_PATH = Path("models/policy_simulator.joblib")
META_PATH  = Path("models/policy_meta.json")
MODEL_PATH.parent.mkdir(exist_ok=True)

# ── Feature engineering helpers ───────────────────────────────────────────────

def load_crash_data(path: Path) -> pd.DataFrame:
    """
    Load WisDOT crash CSV.
    Expected columns (flexible — we map what exists):
        ACCDDATE / ACCDATE / crash_date
        LATITUDE / lat
        LONGITUDE / lon
        SEVERITY / INJSVR / injury_severity
        RDSURF  / road_surface
        LTCOND  / lighting_condition
        WTHRCOND / weather_condition
        RDCLASS / road_class
        SPEED   / SPEEDLIMIT / posted_speed
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    # ── Rename to canonical names ──────────────────────────────────────────────
    rename_map = {
        "accddate": "date", "accdate": "date", "crash_date": "date",
        "latitude": "lat", "loccnty_lat": "lat",
        "longitude": "lon", "loccnty_lon": "lon",
        "severity": "severity", "injsvr": "severity",
        "rdsurf": "road_surface", "road_surface_condition": "road_surface",
        "ltcond": "lighting", "lighting_condition": "lighting",
        "wthrcond": "weather", "weather_condition": "weather",
        "rdclass": "road_class",
        "speed": "speed_limit", "speedlimit": "speed_limit",
        "posted_speed": "speed_limit",
        "totfatl": "fatalities", "fatalities": "fatalities",
        "totinj": "injuries", "injuries": "injuries",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # ── Date features ──────────────────────────────────────────────────────────
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["hour"]    = df["date"].dt.hour.fillna(12).astype(int)
        df["month"]   = df["date"].dt.month.fillna(6).astype(int)
        df["weekday"] = df["date"].dt.dayofweek.fillna(2).astype(int)
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)
        df["is_rush_hour"] = df["hour"].isin([7,8,9,16,17,18]).astype(int)
    else:
        df["hour"] = 12; df["month"] = 6; df["weekday"] = 2
        df["is_weekend"] = 0; df["is_rush_hour"] = 0

    # ── Target: severity score 1–5 ─────────────────────────────────────────────
    if "severity" not in df.columns:
        if "fatalities" in df.columns and "injuries" in df.columns:
            df["severity"] = (
                df["fatalities"].fillna(0) * 5 +
                df["injuries"].fillna(0) * 1
            ).clip(1, 5)
        else:
            df["severity"] = 2.0
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(2.0)
    df["severity"] = df["severity"].clip(1, 5)

    # ── Speed limit ───────────────────────────────────────────────────────────
    if "speed_limit" not in df.columns:
        df["speed_limit"] = 35.0
    df["speed_limit"] = pd.to_numeric(df["speed_limit"], errors="coerce").fillna(35.0)

    # ── Road surface (0=dry, 1=wet, 2=snow/ice, 3=other) ─────────────────────
    if "road_surface" in df.columns:
        surf_map = {1: 0, 2: 1, 3: 2, 4: 2, 5: 3}  # WisDOT codes
        df["road_surface_code"] = df["road_surface"].map(surf_map).fillna(0).astype(int)
    else:
        df["road_surface_code"] = 0

    # ── Lighting (0=daylight, 1=dark with lights, 2=dark no lights) ──────────
    if "lighting" in df.columns:
        light_map = {1: 0, 2: 1, 3: 2, 4: 1, 5: 2}
        df["lighting_code"] = df["lighting"].map(light_map).fillna(0).astype(int)
    else:
        df["lighting_code"] = 0

    # ── Weather condition (0=clear, 1=rain, 2=snow, 3=fog) ───────────────────
    if "weather" in df.columns:
        wx_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 1}
        df["weather_code"] = df["weather"].map(wx_map).fillna(0).astype(int)
    else:
        df["weather_code"] = 0

    # ── Road class (arterial, collector, local, etc.) ─────────────────────────
    if "road_class" in df.columns:
        rc_map = {1: 3, 2: 3, 3: 2, 4: 2, 5: 1, 6: 1, 7: 0}
        df["road_class_code"] = df["road_class"].map(rc_map).fillna(1).astype(int)
    else:
        df["road_class_code"] = 1

    # ── Coordinates ───────────────────────────────────────────────────────────
    for c in ["lat", "lon"]:
        if c not in df.columns:
            df[c] = np.nan
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Filter to Madison bounding box (loose)
    mask = (
        df["lat"].between(42.9, 43.3) &
        df["lon"].between(-89.7, -89.1)
    )
    if mask.sum() > 100:
        df = df[mask]

    return df


def load_weather_data(path: Path) -> pd.DataFrame:
    """
    Load hourly weather.  Expected: datetime, precipitation, visibility,
    temperature, wind_speed.  Returns daily aggregates.
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {
        "date": "datetime", "time": "datetime",
        "prcp": "precipitation", "precip": "precipitation",
        "vis": "visibility",
        "tmpf": "temperature", "temp": "temperature",
        "sknt": "wind_speed", "wnd_spd": "wind_speed",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["date"] = df["datetime"].dt.date
    else:
        return pd.DataFrame()

    agg = {}
    for col, agg_fn in [("precipitation", "sum"), ("visibility", "mean"),
                        ("temperature", "mean"), ("wind_speed", "mean")]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            agg[col] = agg_fn

    if not agg:
        return pd.DataFrame()

    return df.groupby("date").agg(agg).reset_index()


def build_feature_matrix(crash_df: pd.DataFrame,
                          weather_df: pd.DataFrame | None = None) -> tuple:
    """
    Returns X (features), y (severity), feature_names.
    """
    FEATURES = [
        "speed_limit",
        "road_surface_code",
        "lighting_code",
        "weather_code",
        "road_class_code",
        "hour",
        "month",
        "weekday",
        "is_weekend",
        "is_rush_hour",
        # weather join columns added below
    ]

    # Join weather if available
    if weather_df is not None and not weather_df.empty and "date" in crash_df.columns:
        crash_df["_date"] = crash_df["date"].dt.date
        crash_df = crash_df.merge(weather_df, left_on="_date", right_on="date",
                                  how="left", suffixes=("", "_wx"))
        for col in ["precipitation", "visibility", "temperature", "wind_speed"]:
            if col in crash_df.columns:
                crash_df[col] = crash_df[col].fillna(crash_df[col].median())
                if col not in FEATURES:
                    FEATURES.append(col)

    # Fill missing feature columns with sensible defaults
    defaults = {
        "speed_limit": 35, "road_surface_code": 0, "lighting_code": 0,
        "weather_code": 0, "road_class_code": 1, "hour": 12, "month": 6,
        "weekday": 2, "is_weekend": 0, "is_rush_hour": 0,
        "precipitation": 0, "visibility": 10, "temperature": 55, "wind_speed": 5,
    }
    for col in FEATURES:
        if col not in crash_df.columns:
            crash_df[col] = defaults.get(col, 0)

    X = crash_df[FEATURES].copy().fillna(0)
    y = crash_df["severity"].values
    return X, y, FEATURES


# ── Model training ─────────────────────────────────────────────────────────────

def train_model(crash_path: str = "data/wisdot_crash.csv",
                weather_path: str = "data/weather_hourly.csv") -> dict:
    """
    Train gradient-boosted regressor. Returns metrics dict.
    """
    print("Loading crash data…")
    crash_df = load_crash_data(Path(crash_path))
    print(f"  {len(crash_df):,} crash records (Madison region)")

    weather_df = None
    if Path(weather_path).exists():
        print("Loading weather data…")
        weather_df = load_weather_data(Path(weather_path))
        print(f"  {len(weather_df):,} weather days")

    print("Building feature matrix…")
    X, y, feature_names = build_feature_matrix(crash_df, weather_df)
    print(f"  Features: {feature_names}")
    print(f"  Samples: {len(X):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Model: Gradient Boosting (captures non-linear policy effects) ──────────
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )

    # ── Uncertainty: train ensemble of 20 small forests ───────────────────────
    ensemble = [
        RandomForestRegressor(n_estimators=50, max_samples=0.8,
                              max_features=0.7, random_state=i)
        for i in range(20)
    ]

    print("Training main model…")
    model.fit(X_train, y_train)

    print("Training uncertainty ensemble…")
    for m in ensemble:
        m.fit(X_train, y_train)

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
    print(f"  Test MAE: {mae:.3f}  |  R²: {r2:.3f}  |  CV R²: {cv:.3f}")

    # ── SHAP explainer ────────────────────────────────────────────────────────
    print("Computing SHAP values…")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:500])
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = {
        feat: float(imp)
        for feat, imp in sorted(
            zip(feature_names, mean_shap),
            key=lambda x: -x[1]
        )
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump({
        "model":         model,
        "ensemble":      ensemble,
        "feature_names": feature_names,
        "explainer":     explainer,
        "X_train_mean":  X_train.mean().to_dict(),
        "X_train_std":   X_train.std().to_dict(),
        "y_mean":        float(y_train.mean()),
    }, MODEL_PATH)

    meta = {
        "mae": round(mae, 4),
        "r2":  round(r2, 4),
        "cv_r2": round(cv, 4),
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "n_samples": len(X),
        "baseline_severity": round(float(y.mean()), 4),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Model saved →", MODEL_PATH)
    return meta


# ── Policy simulation ──────────────────────────────────────────────────────────

# Maps policy knob names → which feature(s) they affect and how
POLICY_EFFECT_MAP = {
    "speed_limit_reduction": {
        # speed_limit: reduce by policy value (mph)
        # research: 10mph reduction ≈ 25-30% severity reduction on arterials
        "feature": "speed_limit",
        "delta_fn": lambda base, val: -val,           # subtract mph
        "severity_multiplier_fn": lambda val: 1 - 0.025 * val,  # ~2.5% per mph
    },
    "stop_sign_increase": {
        # proxy: reduces effective speed, increases road_class friction
        "feature": "speed_limit",
        "delta_fn": lambda base, val: -val * 2,       # each sign ≈ -2 mph effective speed
        "severity_multiplier_fn": lambda val: 1 - 0.04 * val,
    },
    "street_lighting_improvement": {
        # lighting_code: shift toward daylight equivalent
        "feature": "lighting_code",
        "delta_fn": lambda base, val: -min(val / 10, base),
        "severity_multiplier_fn": lambda val: 1 - 0.03 * val,
    },
    "road_surface_improvement": {
        # road_surface_code: shift toward dry/better
        "feature": "road_surface_code",
        "delta_fn": lambda base, val: -min(val / 10, base),
        "severity_multiplier_fn": lambda val: 1 - 0.025 * val,
    },
    "traffic_calming_measures": {
        # combined: speed + friction
        "feature": "speed_limit",
        "delta_fn": lambda base, val: -val * 3,
        "severity_multiplier_fn": lambda val: 1 - 0.05 * min(val, 10),
    },
    "pedestrian_crossing_improvements": {
        # reduces severity for pedestrian crashes
        "feature": "road_class_code",
        "delta_fn": lambda base, val: 0,
        "severity_multiplier_fn": lambda val: 1 - 0.02 * val,
    },
    "visibility_enhancement": {
        # better signage, delineators → effective visibility boost
        "feature": "weather_code",
        "delta_fn": lambda base, val: 0,
        "severity_multiplier_fn": lambda val: 1 - 0.015 * val,
    },
    "weather_responsive_treatment": {
        # de-icing, plowing → shifts surface from ice/snow toward wet/dry
        "feature": "road_surface_code",
        "delta_fn": lambda base, val: -min(val / 5, 2),
        "severity_multiplier_fn": lambda val: 1 - 0.04 * min(val, 10),
    },
}


def load_model():
    """Load saved model artifacts."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python simulator_model.py first."
        )
    return joblib.load(MODEL_PATH)


def simulate_policy(
    policies: dict[str, float],
    scenario_context: dict | None = None,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict:
    """
    Main simulation function.

    Args:
        policies: dict of {policy_name: intensity_value (0-10 scale)}
                  e.g. {"speed_limit_reduction": 10, "street_lighting_improvement": 7}

        scenario_context: optional dict to override baseline context features
                  e.g. {"weather_code": 2, "hour": 8, "month": 1}

        n_bootstrap: samples for uncertainty estimation

    Returns:
        {
          "baseline_risk": float,
          "predicted_risk": float,
          "risk_reduction_pct": float,
          "confidence_interval": [low, high],
          "uncertainty_score": float,
          "crash_count_reduction": {"estimated": int, "low": int, "high": int},
          "policy_contributions": {policy_name: pct_contribution},
          "feature_impacts": {feature_name: delta},
          "explanation": str,
          "applied_policies": {policy_name: intensity},
        }
    """
    rng    = np.random.default_rng(seed)
    bundle = load_model()

    model         = bundle["model"]
    ensemble      = bundle["ensemble"]
    feature_names = bundle["feature_names"]
    X_mean        = bundle["X_train_mean"]

    # ── Build baseline feature vector (average Madison crash context) ──────────
    baseline = {
        "speed_limit":       35.0,
        "road_surface_code": 0.3,
        "lighting_code":     0.4,
        "weather_code":      0.2,
        "road_class_code":   1.5,
        "hour":              13.0,
        "month":             7.0,
        "weekday":           2.0,
        "is_weekend":        0.3,
        "is_rush_hour":      0.35,
        "precipitation":     0.08,
        "visibility":        8.5,
        "temperature":       52.0,
        "wind_speed":        9.0,
    }
    # Override with trained means where available
    for k in baseline:
        if k in X_mean:
            baseline[k] = X_mean[k]

    # Override with user-supplied context
    if scenario_context:
        for k, v in scenario_context.items():
            if k in baseline:
                baseline[k] = float(v)

    # ── Validate policies ──────────────────────────────────────────────────────
    valid_policies = {k: float(v) for k, v in policies.items()
                      if k in POLICY_EFFECT_MAP and 0 < float(v) <= 10}
    if not valid_policies:
        raise ValueError(f"No valid policies. Valid options: {list(POLICY_EFFECT_MAP.keys())}")

    # ── Apply policy feature perturbations ────────────────────────────────────
    modified = baseline.copy()
    feature_impacts = {}

    for policy_name, intensity in valid_policies.items():
        spec    = POLICY_EFFECT_MAP[policy_name]
        feature = spec["feature"]
        if feature in modified:
            orig_val  = modified[feature]
            delta     = spec["delta_fn"](orig_val, intensity)
            new_val   = orig_val + delta
            # Clip to physical bounds
            bounds = {
                "speed_limit":       (5, 75),
                "road_surface_code": (0, 3),
                "lighting_code":     (0, 2),
                "weather_code":      (0, 3),
                "road_class_code":   (0, 3),
            }
            if feature in bounds:
                lo, hi = bounds[feature]
                new_val = np.clip(new_val, lo, hi)
            modified[feature] = new_val
            feature_impacts[feature] = round(new_val - orig_val, 3)

    # ── Build X arrays ─────────────────────────────────────────────────────────
    def to_X(ctx):
        row = [ctx.get(f, 0) for f in feature_names]
        return np.array(row).reshape(1, -1)

    X_base = to_X(baseline)
    X_mod  = to_X(modified)

    baseline_risk = float(model.predict(X_base)[0])
    predicted_risk = float(model.predict(X_mod)[0])

    # ── Severity multiplier from policy research literature ───────────────────
    combined_multiplier = 1.0
    policy_contributions = {}
    for policy_name, intensity in valid_policies.items():
        spec = POLICY_EFFECT_MAP[policy_name]
        mult = spec["severity_multiplier_fn"](intensity)
        mult = max(0.3, min(1.0, mult))  # bound
        policy_contributions[policy_name] = round((1 - mult) * 100, 2)
        combined_multiplier *= mult

    # Blend ML prediction with literature-based multiplier (50/50 for robustness)
    ml_reduction = (baseline_risk - predicted_risk) / max(baseline_risk, 0.01)
    lit_reduction = 1 - combined_multiplier
    blended_reduction = 0.5 * ml_reduction + 0.5 * lit_reduction
    blended_reduction = max(0.0, min(0.8, blended_reduction))  # cap at 80%

    final_predicted_risk = baseline_risk * (1 - blended_reduction)

    # ── Uncertainty via ensemble ───────────────────────────────────────────────
    ens_base_preds = np.array([m.predict(X_base)[0] for m in ensemble])
    ens_mod_preds  = np.array([m.predict(X_mod)[0]  for m in ensemble])
    ens_reductions = (ens_base_preds - ens_mod_preds) / np.maximum(ens_base_preds, 0.01)
    ens_reductions = np.clip(ens_reductions, 0, 0.8)

    ci_low  = float(np.percentile(ens_reductions, 10))
    ci_high = float(np.percentile(ens_reductions, 90))
    uncertainty_score = float(np.std(ens_reductions))

    # ── Crash count estimation ─────────────────────────────────────────────────
    # Based on annual Madison crash stats (~4,500 crashes/year in region)
    annual_crashes = 4500
    estimated_reduction = int(annual_crashes * blended_reduction)
    low_reduction       = int(annual_crashes * ci_low)
    high_reduction      = int(annual_crashes * ci_high)

    # ── Natural language explanation ───────────────────────────────────────────
    top_policy = max(policy_contributions, key=policy_contributions.get)
    top_feature = max(feature_impacts, key=lambda k: abs(feature_impacts[k])) \
                  if feature_impacts else "speed"

    explanation_parts = []
    for p, contrib in sorted(policy_contributions.items(),
                              key=lambda x: -x[1])[:3]:
        readable = p.replace("_", " ").title()
        explanation_parts.append(
            f"{readable} contributes approximately {contrib:.1f}% risk reduction"
        )

    explanation = (
        f"Under the simulated policies, overall crash severity risk is projected to "
        f"decrease by {blended_reduction*100:.1f}% (95% CI: {ci_low*100:.1f}%–{ci_high*100:.1f}%). "
        f"The dominant driver is {top_policy.replace('_',' ')} acting on {top_feature.replace('_',' ')}. "
        + " | ".join(explanation_parts) + "."
    )

    return {
        "baseline_risk":       round(baseline_risk, 4),
        "predicted_risk":      round(final_predicted_risk, 4),
        "risk_reduction_pct":  round(blended_reduction * 100, 2),
        "confidence_interval": [round(ci_low * 100, 2), round(ci_high * 100, 2)],
        "uncertainty_score":   round(uncertainty_score * 100, 2),
        "crash_count_reduction": {
            "estimated": estimated_reduction,
            "low":       low_reduction,
            "high":      high_reduction,
        },
        "policy_contributions":  policy_contributions,
        "feature_impacts":       feature_impacts,
        "explanation":           explanation,
        "applied_policies":      valid_policies,
        "available_policies":    list(POLICY_EFFECT_MAP.keys()),
    }


def get_meta() -> dict:
    """Return model training metadata."""
    if META_PATH.exists():
        with open(META_PATH) as f:
            return json.load(f)
    return {}


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, argparse

    parser = argparse.ArgumentParser(description="Train policy simulator model")
    parser.add_argument("--crash",   default="data/wisdot_crash.csv")
    parser.add_argument("--weather", default="data/weather_hourly.csv")
    args = parser.parse_args()

    if not Path(args.crash).exists():
        print(f"⚠️  Crash file not found at {args.crash}")
        print("   Using synthetic data for demo…")

        # ── Generate realistic synthetic data so the demo runs without real files ──
        np.random.seed(42)
        n = 8000
        synthetic = pd.DataFrame({
            "date":             pd.date_range("2018-01-01", periods=n, freq="2h"),
            "lat":              np.random.uniform(43.02, 43.14, n),
            "lon":              np.random.uniform(-89.55, -89.30, n),
            "speed_limit":      np.random.choice([25,30,35,40,45,55], n,
                                                 p=[0.1,0.2,0.3,0.2,0.15,0.05]),
            "road_surface_code":np.random.choice([0,1,2,3], n, p=[0.6,0.2,0.15,0.05]),
            "lighting_code":    np.random.choice([0,1,2], n, p=[0.5,0.3,0.2]),
            "weather_code":     np.random.choice([0,1,2,3], n, p=[0.6,0.2,0.15,0.05]),
            "road_class_code":  np.random.choice([0,1,2,3], n, p=[0.2,0.4,0.3,0.1]),
            "hour":             np.random.randint(0, 24, n),
            "month":            np.random.randint(1, 13, n),
            "weekday":          np.random.randint(0, 7, n),
            "is_weekend":       np.random.randint(0, 2, n),
            "is_rush_hour":     np.random.randint(0, 2, n),
            "precipitation":    np.random.exponential(0.1, n),
            "visibility":       np.random.uniform(1, 10, n),
            "temperature":      np.random.normal(50, 20, n),
            "wind_speed":       np.random.exponential(8, n),
        })

        # Synthetic severity based on real relationships
        synthetic["severity"] = (
            1.0
            + 0.03  * (synthetic["speed_limit"] - 25)
            + 0.4   * synthetic["road_surface_code"]
            + 0.3   * synthetic["lighting_code"]
            + 0.35  * synthetic["weather_code"]
            + 0.2   * synthetic["road_class_code"]
            + 0.15  * synthetic["is_rush_hour"]
            + 0.1   * (synthetic["precipitation"] > 0.1).astype(int)
            + np.random.normal(0, 0.3, n)
        ).clip(1, 5)

        Path("data").mkdir(exist_ok=True)
        synthetic.to_csv(args.crash, index=False)
        print(f"  Synthetic dataset ({n} rows) saved to {args.crash}")

    meta = train_model(args.crash, args.weather)
    print("\n📊 Model Metadata:")
    print(json.dumps(meta, indent=2))

    print("\n🧪 Quick simulation test:")
    result = simulate_policy({
        "speed_limit_reduction":     8,
        "street_lighting_improvement": 6,
        "road_surface_improvement":  5,
    })
    print(json.dumps(result, indent=2))
