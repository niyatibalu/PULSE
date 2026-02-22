# 🚦 Urban Safety — What-If Policy Simulator

## Files
```
simulator_model.py   ← ML model + simulate_policy() function
simulator_api.py     ← Flask REST API (port 5050)
simulator_ui.html    ← Frontend dashboard (open in browser)
```

## Setup

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn shap scipy joblib flask flask-cors
```

### 2. Place your data files
```
data/
  wisdot_crash.csv       ← required (or it generates synthetic data)
  weather_hourly.csv     ← optional (joins to crash by date)
```

### 3. Train the model
```bash
python simulator_model.py
```
This will:
- Load wisdot_crash.csv (or generate 8,000-row synthetic data if missing)
- Engineer features (speed, lighting, road surface, weather, time-of-day, etc.)
- Train Gradient Boosting + 20-model uncertainty ensemble
- Compute SHAP feature importance
- Save to `models/policy_simulator.joblib` + `models/policy_meta.json`

### 4. Start the API
```bash
python simulator_api.py
```
Runs on http://localhost:5050

### 5. Open the frontend
Just open `simulator_ui.html` in your browser. No server needed.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Model status + metadata |
| GET | `/api/policies` | All policy knobs + presets |
| GET | `/api/baseline` | Baseline risk + model performance |
| POST | `/api/simulate` | Run simulation |
| POST | `/api/compare` | Compare multiple scenarios |
| GET | `/api/presets/<id>` | Run a named preset |

### POST /api/simulate
```json
{
  "policies": {
    "speed_limit_reduction": 8,
    "street_lighting_improvement": 6,
    "road_surface_improvement": 5
  },
  "context": {
    "weather_code": 2,
    "hour": 8
  }
}
```

Policy intensity is a **1–10 scale** (1 = minimal, 10 = maximum intervention).

### Response
```json
{
  "baseline_risk": 2.34,
  "predicted_risk": 1.61,
  "risk_reduction_pct": 31.2,
  "confidence_interval": [18.4, 44.1],
  "uncertainty_score": 7.3,
  "crash_count_reduction": { "estimated": 1404, "low": 828, "high": 1985 },
  "policy_contributions": { "speed_limit_reduction": 18.2, ... },
  "feature_impacts": { "speed_limit": -8.0, ... },
  "explanation": "Under the simulated policies...",
  "applied_policies": { ... }
}
```

---

## Available Policies

| Policy Key | Description |
|------------|-------------|
| `speed_limit_reduction` | Reduce posted speed (mph) |
| `stop_sign_increase` | Add stop signs / all-way stops |
| `street_lighting_improvement` | Upgrade street lighting |
| `road_surface_improvement` | Repave, improve drainage |
| `traffic_calming_measures` | Roundabouts, speed humps |
| `pedestrian_crossing_improvements` | HAWK beacons, crosswalks |
| `visibility_enhancement` | Better signage, delineators |
| `weather_responsive_treatment` | De-icing, anti-icing programs |

---

## How the Model Works

1. **Features**: speed limit, road surface, lighting, weather, time-of-day, day-of-week, month, weather conditions
2. **Target**: crash severity score (1–5 scale)
3. **Model**: Gradient Boosting Regressor (300 estimators, max_depth=5)
4. **Uncertainty**: 20-model Random Forest ensemble → 10th/90th percentile CI
5. **Policy effects**: Blend of (a) direct ML feature perturbation + (b) literature-calibrated severity multipliers
6. **Crash count estimate**: Risk reduction × 4,500 annual Madison crashes

## Connecting to Your Main Dashboard
Import `simulate_policy` directly:
```python
from simulator_model import simulate_policy

result = simulate_policy({
    "speed_limit_reduction": 8,
    "street_lighting_improvement": 6,
})
print(result["risk_reduction_pct"])  # e.g. 28.4
```

Or hit the API from your React/map frontend:
```js
const res = await fetch("http://localhost:5050/api/simulate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ policies: { speed_limit_reduction: 8 } })
});
const data = await res.json();
```
