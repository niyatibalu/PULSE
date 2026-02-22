"""
Urban Safety What-If Policy Simulator — Flask API
==================================================
Run:
    pip install flask flask-cors
    python simulator_api.py

Endpoints:
    GET  /api/health                    — health check + model meta
    GET  /api/policies                  — list available policy knobs
    POST /api/simulate                  — run a simulation
    GET  /api/baseline                  — baseline risk context
    POST /api/compare                   — compare multiple scenarios
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import traceback
import json
from pathlib import Path

# Import our model layer
from simulator_model import simulate_policy, get_meta, POLICY_EFFECT_MAP, load_model

app = Flask(__name__)
CORS(app)  # allow requests from the React/HTML frontend


# ── Pretty descriptions for the UI ────────────────────────────────────────────
POLICY_DESCRIPTIONS = {
    "speed_limit_reduction": {
        "label":       "Speed Limit Reduction",
        "description": "Reduce posted speed limits across the target zone.",
        "unit":        "mph reduction",
        "min":         1,
        "max":         10,
        "default":     5,
        "icon":        "🚦",
        "category":    "Speed Management",
        "research_note": "Each 1 mph reduction ≈ 2–3% crash severity decrease (WHO, 2023)",
    },
    "stop_sign_increase": {
        "label":       "Add Stop Signs / All-Way Stops",
        "description": "Install additional stop signs or convert intersections to all-way stops.",
        "unit":        "interventions (1–10 scale)",
        "min":         1,
        "max":         10,
        "default":     4,
        "icon":        "🛑",
        "category":    "Intersection Control",
        "research_note": "All-way stops reduce injury crashes ~40% at eligible intersections",
    },
    "street_lighting_improvement": {
        "label":       "Street Lighting Improvement",
        "description": "Upgrade or install new street lighting in high-risk corridors.",
        "unit":        "improvement level (1–10 scale)",
        "min":         1,
        "max":         10,
        "default":     6,
        "icon":        "💡",
        "category":    "Infrastructure",
        "research_note": "Improved lighting reduces night crashes ~30% (Elvik & Vaa, 2004)",
    },
    "road_surface_improvement": {
        "label":       "Road Surface Improvement",
        "description": "Repair potholes, improve drainage, re-stripe lanes.",
        "unit":        "improvement level (1–10 scale)",
        "min":         1,
        "max":         10,
        "default":     5,
        "icon":        "🛣️",
        "category":    "Infrastructure",
        "research_note": "Resurfacing + markings reduce crashes ~15–25%",
    },
    "traffic_calming_measures": {
        "label":       "Traffic Calming (Roundabouts / Raised Crossings)",
        "description": "Install speed humps, raised crosswalks, mini-roundabouts.",
        "unit":        "measures deployed (1–10 scale)",
        "min":         1,
        "max":         10,
        "default":     5,
        "icon":        "🔄",
        "category":    "Speed Management",
        "research_note": "Roundabouts reduce fatal+injury crashes ~75% vs signalized intersections",
    },
    "pedestrian_crossing_improvements": {
        "label":       "Pedestrian Crossing Upgrades",
        "description": "High-visibility crosswalks, pedestrian refuge islands, HAWK beacons.",
        "unit":        "improvement level (1–10 scale)",
        "min":         1,
        "max":         10,
        "default":     6,
        "icon":        "🚶",
        "category":    "Active Transportation",
        "research_note": "HAWK beacons reduce pedestrian crashes ~69%",
    },
    "visibility_enhancement": {
        "label":       "Visibility Enhancement (Signage / Delineators)",
        "description": "Improved road markings, reflective delineators, better signage.",
        "unit":        "improvement level (1–10 scale)",
        "min":         1,
        "max":         10,
        "default":     5,
        "icon":        "🪧",
        "category":    "Signage",
        "research_note": "Improved delineation reduces run-off-road crashes ~20%",
    },
    "weather_responsive_treatment": {
        "label":       "Weather-Responsive Treatment (De-icing / Plowing)",
        "description": "Enhanced winter road treatment protocols and anti-icing programs.",
        "unit":        "program intensity (1–10 scale)",
        "min":         1,
        "max":         10,
        "default":     7,
        "icon":        "❄️",
        "category":    "Weather Management",
        "research_note": "Anti-icing programs reduce snow/ice crashes ~30–50%",
    },
}

SCENARIO_PRESETS = {
    "safe_streets_lite": {
        "label":       "Safe Streets Lite",
        "description": "Low-cost, high-impact quick wins.",
        "policies": {
            "street_lighting_improvement":        5,
            "visibility_enhancement":             6,
            "pedestrian_crossing_improvements":   5,
        },
    },
    "speed_first": {
        "label":       "Speed Reduction Focus",
        "description": "Aggressive speed management package.",
        "policies": {
            "speed_limit_reduction":    8,
            "stop_sign_increase":       6,
            "traffic_calming_measures": 7,
        },
    },
    "winter_ready": {
        "label":       "Winter Safety Package",
        "description": "Prepare for Wisconsin winters.",
        "policies": {
            "weather_responsive_treatment": 9,
            "road_surface_improvement":     7,
            "street_lighting_improvement":  5,
        },
    },
    "full_intervention": {
        "label":       "Full Intervention Bundle",
        "description": "All policies at moderate-high intensity.",
        "policies": {k: 7 for k in POLICY_EFFECT_MAP},
    },
}


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    meta = get_meta()
    model_ready = Path("models/policy_simulator.joblib").exists()
    return jsonify({
        "status":      "ok" if model_ready else "model_not_trained",
        "model_ready": model_ready,
        "meta":        meta,
    })


@app.route("/api/policies")
def policies():
    return jsonify({
        "policies":  POLICY_DESCRIPTIONS,
        "presets":   SCENARIO_PRESETS,
        "categories": list({v["category"] for v in POLICY_DESCRIPTIONS.values()}),
    })


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    POST body:
    {
        "policies": {"speed_limit_reduction": 8, "street_lighting_improvement": 5},
        "context":  {"weather_code": 2, "hour": 8}   // optional
    }
    """
    body = request.get_json(force=True) or {}

    raw_policies = body.get("policies", {})
    context      = body.get("context", {})

    if not raw_policies:
        return jsonify({"error": "No policies provided. Send {'policies': {'policy_name': intensity}}"}), 400

    # Validate intensity values
    invalid = {k: v for k, v in raw_policies.items() if not (0 < float(v) <= 10)}
    if invalid:
        return jsonify({"error": f"Intensities must be 1–10. Invalid: {invalid}"}), 400

    try:
        result = simulate_policy(policies=raw_policies, scenario_context=context or None)

        # Enrich with human-readable labels
        result["policy_labels"] = {
            k: POLICY_DESCRIPTIONS.get(k, {}).get("label", k)
            for k in raw_policies
        }
        result["research_notes"] = {
            k: POLICY_DESCRIPTIONS.get(k, {}).get("research_note", "")
            for k in raw_policies
        }
        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({"error": str(e), "hint": "Run: python simulator_model.py"}), 503
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/baseline")
def baseline():
    """Return the current baseline risk context and statistics."""
    meta = get_meta()
    return jsonify({
        "baseline_severity":     meta.get("baseline_severity", 2.0),
        "annual_crash_estimate": 4500,
        "feature_importance":    meta.get("feature_importance", {}),
        "model_performance": {
            "mae":    meta.get("mae"),
            "r2":     meta.get("r2"),
            "cv_r2":  meta.get("cv_r2"),
        },
        "context": {
            "city":             "Madison, WI",
            "data_source":      "WisDOT Crash Records",
            "years_covered":    "2018–2023",
            "n_training_samples": meta.get("n_samples"),
        },
    })


@app.route("/api/compare", methods=["POST"])
def compare_scenarios():
    """
    Compare multiple named scenarios in one call.

    POST body:
    {
        "scenarios": {
            "scenario_a": {"policies": {...}, "label": "My Plan A"},
            "scenario_b": {"policies": {...}, "label": "My Plan B"}
        },
        "context": {...}   // optional shared context
    }
    """
    body      = request.get_json(force=True) or {}
    scenarios = body.get("scenarios", {})
    context   = body.get("context", {})

    if not scenarios:
        return jsonify({"error": "No scenarios provided"}), 400

    results = {}
    for name, scenario in scenarios.items():
        try:
            r = simulate_policy(
                policies=scenario.get("policies", {}),
                scenario_context=context or None,
            )
            r["label"] = scenario.get("label", name)
            results[name] = r
        except Exception as e:
            results[name] = {"error": str(e)}

    # Rank by risk reduction
    ranked = sorted(
        [(k, v) for k, v in results.items() if "risk_reduction_pct" in v],
        key=lambda x: -x[1]["risk_reduction_pct"]
    )

    return jsonify({
        "results":      results,
        "ranking":      [{"id": k, "label": v.get("label", k),
                          "risk_reduction_pct": v["risk_reduction_pct"]}
                         for k, v in ranked],
        "best_scenario": ranked[0][0] if ranked else None,
    })


@app.route("/api/presets/<preset_id>")
def run_preset(preset_id):
    """Run a named scenario preset."""
    preset = SCENARIO_PRESETS.get(preset_id)
    if not preset:
        return jsonify({
            "error": f"Unknown preset '{preset_id}'",
            "available": list(SCENARIO_PRESETS.keys())
        }), 404
    try:
        result = simulate_policy(policies=preset["policies"])
        result["preset"]      = preset_id
        result["preset_label"] = preset["label"]
        result["preset_description"] = preset["description"]
        return jsonify(result)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚦 Urban Safety Policy Simulator API")
    print("   Endpoints:")
    print("   GET  http://localhost:5050/api/health")
    print("   GET  http://localhost:5050/api/policies")
    print("   POST http://localhost:5050/api/simulate")
    print("   POST http://localhost:5050/api/compare")
    print("   GET  http://localhost:5050/api/baseline")
    print("   GET  http://localhost:5050/api/presets/<id>\n")
    app.run(debug=True, port=5050)
