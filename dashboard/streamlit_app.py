import pandas as pd
import streamlit as st
import pydeck as pdk
from pathlib import Path
from datetime import datetime
import pandas as pd
import pandas as pd

@st.cache_data
def load_edge_predictions():
    df = pd.read_csv("data/edge_predictions.csv")

    # Pick the time column automatically
    time_col = None
    if "hour" in df.columns:
        time_col = "hour"
    elif "timestamp" in df.columns:
        time_col = "timestamp"
    else:
        raise ValueError(f"edge_predictions.csv has no time column. Columns: {df.columns.tolist()}")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # Normalize to one name so the rest of your app is consistent
    if time_col != "hour":
        df = df.rename(columns={time_col: "hour"})

    return df

@st.cache_data
def load_ems_recommendations():
    return pd.read_csv("data/ems_recommendations.csv")

@st.cache_data
def load_hotspot_coverage():
    return pd.read_csv("data/hotspot_coverage.csv")

@st.cache_data
def load_ems_recs():
    return pd.read_csv("data/ems_recommendations.csv")

pred = load_edge_predictions()
latest_hour = pred["hour"].max()
latest_pred = pred[pred["hour"] == latest_hour].sort_values("risk", ascending=False)

ems = load_ems_recommendations()
cover = load_hotspot_coverage()


mean_tt = (
    cover["best_travel_time_s"]
    .replace([float("inf"), float("-inf")], pd.NA)
    .dropna()
    .mean()
)

ems = load_ems_recs()

ems_layer = pdk.Layer(
    "ScatterplotLayer",
    data=ems,
    get_position=["lon", "lat"],
    get_radius=300,
    pickable=True,
)

@st.cache_data
def load_edge_preds():
    df = pd.read_csv("data/edge_predictions.csv")
    df["hour"] = pd.to_datetime(df["hour"])
    return df

@st.cache_data
def load_latest_hotspots(n=50):
    df = pd.read_csv("data/edge_predictions.csv")
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"])
    latest_hour = df["hour"].max()
    latest = df[df["hour"] == latest_hour].sort_values("risk", ascending=False)
    return latest_hour, latest.head(n)

ems = load_ems_recommendations()
cover = load_hotspot_coverage()
latest_hour, top_hotspots = load_latest_hotspots(50)

mean_tt = (
    cover["best_travel_time_s"]
    .replace([float("inf"), float("-inf")], pd.NA)
    .dropna()
    .mean()
)

pred = load_edge_preds()
latest_hour = pred["hour"].max()
latest = pred[pred["hour"] == latest_hour].sort_values("risk", ascending=False)

# Optional: point-query the API if it's running
try:
    import requests
except Exception:
    requests = None

from app.ml.resources import recommend_ems_zones
from app.ml.fairness import equity_summary


ROOT = Path(__file__).resolve().parents[1]
DATA_SCORED = ROOT / "data" / "scored.csv"


st.set_page_config(layout="wide")
st.title("Urban Safety & Resource AI")


def load_scored(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(
            f"Missing {path}. You need to generate it first:\n\n"
            f"  python -m scripts.score_dataset\n"
        )
        st.stop()

    df = pd.read_csv(path)

    required = {"timestamp","latitude","longitude","grid_id","risk_mean","risk_uncertainty"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"scored.csv is missing columns: {sorted(missing)}")
        st.stop()

    return df


@st.cache_data
def load_and_aggregate() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_scored(DATA_SCORED)
    # Aggregate to grid-level so the map is fast + meaningful
    agg = (
        df.groupby("grid_id")
          .agg(
              latitude=("latitude", "mean"),
              longitude=("longitude", "mean"),
              risk_mean=("risk_mean", "mean"),
              risk_uncertainty=("risk_uncertainty", "mean"),
              n=("risk_mean", "size"),
          )
          .reset_index()
    )
    return df, agg


df, agg = load_and_aggregate()

st.sidebar.header("Controls")
n_units = st.sidebar.slider("Available EMS units", 1, 15, 5)

unc_cut = float(agg["risk_uncertainty"].quantile(0.80))
unc_threshold = st.sidebar.slider("High-uncertainty threshold", 0.0, float(agg["risk_uncertainty"].max()), float(min(0.08, agg["risk_uncertainty"].max())))

mode = st.sidebar.radio("View", ["Risk Map", "Point Query (API)"])

# Recommend deployment zones based on risk + uncertainty
rec = recommend_ems_zones(agg, n_units=n_units)
deploy = rec[rec["deploy_here"]].copy()

cover = pd.read_csv("data/hotspot_coverage.csv")
mean_tt = (
    cover["best_travel_time_s"]
    .replace([float("inf"), float("-inf")], pd.NA)
    .dropna()
    .mean()
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Grid cells", f"{len(agg):,}")
c2.metric("Recommended sites", f"{len(ems):,}")  # replace deploy with ems if you switched
c3.metric("Hotspots evaluated", f"{cover['best_travel_time_s'].replace([float('inf'), float('-inf')], pd.NA).dropna().shape[0]:,}")
c4.metric("Mean risk", f"{agg['risk_mean'].mean():.4f}")
c5.metric("Mean travel time", f"{mean_tt:.0f} s")

if mode == "Risk Map":
    st.subheader("Predicted Risk Heatmap (grid aggregated)")

    heat = pdk.Layer(
        "HeatmapLayer",
        data=rec,
        get_position=["longitude", "latitude"],
        get_weight="risk_mean",
        radiusPixels=60,
    )

    deploy_layer = pdk.Layer(
        "ScatterplotLayer",
        data=deploy,
        get_position=["longitude", "latitude"],
        get_radius=250,
        pickable=True,
    )

    high_unc = rec[rec["risk_uncertainty"] >= unc_threshold]
    unc_layer = pdk.Layer(
        "ScatterplotLayer",
        data=high_unc,
        get_position=["longitude", "latitude"],
        get_radius=140,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=float(rec["latitude"].mean()),
        longitude=float(rec["longitude"].mean()),
        zoom=11,
    )

    ems = pd.read_csv("data/ems_recommendations.csv")

    ems_layer = pdk.Layer(
        "ScatterplotLayer",
        data=ems,
        get_position=["lon", "lat"],
        get_radius=350,
        pickable=True,
    )

    deck = pdk.Deck(
        layers=[heat, unc_layer, deploy_layer, ems_layer],
        initial_view_state=view_state,
        tooltip={"text": "{selected_rank}. {name}\n{asset_type}"},
    )
    st.pydeck_chart(deck)
    st.subheader("Top road-segment hotspots")
    st.caption(f"Latest hour: {latest_hour}")
    st.dataframe(
        latest[["edge_id", "risk", "accidents", "y"]].head(50),
        use_container_width=True
    )
    st.subheader("Recommended EMS staging sites")
    st.dataframe(
        ems[["selected_rank", "name", "asset_type", "lat", "lon"]],
        use_container_width=True
    )

    st.subheader("Equity and coverage summary")
    eq = equity_summary(rec)
    st.dataframe(eq.head(25), use_container_width=True)

else:
    st.subheader("Point Query (calls FastAPI if running)")
    if requests is None:
        st.error("requests not installed. Install it or use Risk Map mode.")
        st.stop()

    api = st.sidebar.text_input("API base URL", "http://127.0.0.1:8000")

    lat = st.number_input("Latitude", value=float(agg["latitude"].mean()), format="%.6f")
    lon = st.number_input("Longitude", value=float(agg["longitude"].mean()), format="%.6f")
    temperature = st.number_input("Temperature", value=32.0)
    precipitation = st.number_input("Precipitation", value=0.1)
    visibility = st.number_input("Visibility", value=6.0)
    timestamp = st.text_input("Timestamp (ISO)", value=datetime.now().isoformat(timespec="seconds"))

    lag1 = st.number_input("accident_lag_1", value=0.0)
    lag3 = st.number_input("accident_lag_3", value=0.0)
    lag6 = st.number_input("accident_lag_6", value=0.0)

    payload = {
        "timestamp": timestamp,
        "latitude": float(lat),
        "longitude": float(lon),
        "temperature": float(temperature),
        "precipitation": float(precipitation),
        "visibility": float(visibility),
        "accident_lag_1": float(lag1),
        "accident_lag_3": float(lag3),
        "accident_lag_6": float(lag6),
    }

    colA, colB = st.columns(2)

    with colA:
        if st.button("Predict"):
            r = requests.post(f"{api}/predict", json=payload, timeout=15)
            st.write(r.status_code)
            st.json(r.json())

    with colB:
        if st.button("Explain"):
            r = requests.post(f"{api}/explain", json=payload, timeout=30)
            st.write(r.status_code)
            st.json(r.json())
