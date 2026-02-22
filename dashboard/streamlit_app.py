import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pydeck as pdk

try:
    import geopandas as gpd
except Exception:
    gpd = None

st.set_page_config(
    page_title="PULSE",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Outfit:wght@300;400;500;600;700;800;900&family=Syne:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

:root {
  --bg:           #05060a;
  --bg2:          #080c14;
  --bg3:          #0d1120;
  --surface:      rgba(255,255,255,0.035);
  --surfaceHov:   rgba(255,255,255,0.06);
  --border:       rgba(255,255,255,0.07);
  --borderBright: rgba(255,255,255,0.16);
  --text:         #eceef4;
  --muted:        rgba(236,238,244,0.6);
  --muted2:       rgba(236,238,244,0.3);
  --red:          #ff3d3d;
  --redGlow:      rgba(255,61,61,0.22);
  --teal:         #00d4b4;
  --tealGlow:     rgba(0,212,180,0.18);
  --amber:        #f7b731;
  --amberGlow:    rgba(247,183,49,0.18);
  --sky:          #38bdf8;
  --skyGlow:      rgba(56,189,248,0.18);
  --font-display: 'Syne', sans-serif;
  --font-body:    'Outfit', sans-serif;
  --font-mono:    'DM Mono', monospace;
}

/* ── App shell ── */
.stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
}

/* ── Remove Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 0 2.5rem 5rem !important;
  max-width: 100% !important;
}

/* ════════════════════════
   SIDEBAR
════════════════════════ */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
  min-width: 288px !important;
  max-width: 288px !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 0 !important;
  overflow-x: hidden !important;
}
[data-testid="collapsedControl"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
  border-radius: 0 10px 10px 0 !important;
  color: var(--muted) !important;
}
[data-testid="collapsedControl"]:hover {
  background: var(--bg3) !important;
  color: var(--text) !important;
}

.sb-wrap {
  padding: 1.75rem 1.4rem 2rem;
  display: flex;
  flex-direction: column;
  gap: 0;
}
.sb-logo {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 5px;
}
.sb-logo-text {
  font-family: var(--font-display);
  font-size: 30px;
  font-weight: 800;
  letter-spacing: 4px;
  color: var(--text);
}
.sb-logo-dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  background: var(--red);
  box-shadow: 0 0 14px var(--red), 0 0 28px var(--redGlow);
  flex-shrink: 0;
  animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:0.45; transform:scale(0.65); }
}
.sb-tagline {
  font-family: var(--font-mono);
  font-size: 9.5px;
  color: var(--muted2);
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-bottom: 26px;
}
.sb-divider {
  height: 1px;
  background: var(--border);
  margin: 20px 0;
}
.sb-section-label {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--muted2);
  margin-bottom: 10px;
}
.sb-status {
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
  background: var(--surface);
}
.sb-status-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--muted2);
  letter-spacing: 0.5px;
  padding: 6px 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}
.sb-status-row:last-child { border-bottom: none; padding-bottom: 0; }
.sb-status-row:first-child { padding-top: 0; }
.sb-status-val { color: var(--text); font-weight: 500; }
.sb-status-val.green { color: var(--teal); }

/* Sidebar option buttons — styled as selectable tiles */
.stButton > button {
  width: 100% !important;
  text-align: left !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--muted) !important;
  font-family: var(--font-body) !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  padding: 11px 14px !important;
  margin-bottom: 6px !important;
  transition: all 0.15s ease !important;
}
.stButton > button:hover {
  background: var(--surfaceHov) !important;
  border-color: var(--borderBright) !important;
  color: var(--text) !important;
  transform: none !important;
}

/* ════════════════════════
   HERO
════════════════════════ */
.hero {
  position: relative;
  overflow: hidden;
  padding: 3rem 2.5rem 2.5rem;
  margin: 0 -2.5rem 2.5rem;
  background:
    radial-gradient(ellipse 80% 60% at 8% 0%, rgba(255,61,61,0.13) 0%, transparent 60%),
    radial-gradient(ellipse 55% 50% at 92% 100%, rgba(0,212,180,0.08) 0%, transparent 55%),
    linear-gradient(180deg, #0a0d18 0%, #05060a 100%);
  border-bottom: 1px solid var(--border);
}
.hero::before {
  content: '';
  position: absolute;
  inset: 0;
  background-image:
    repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(255,255,255,0.016) 40px),
    repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(255,255,255,0.016) 40px);
  pointer-events: none;
}
.hero-inner {
  position: relative;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 2rem;
}
.hero-brand { max-width: 680px; }
.hero-eyebrow {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--teal);
  margin-bottom: 14px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.hero-eyebrow::before {
  content: '';
  display: inline-block;
  width: 32px; height: 1px;
  background: var(--teal);
  opacity: 0.55;
}
.hero-title {
  font-family: var(--font-display);
  font-size: clamp(64px, 7vw, 104px);
  font-weight: 800;
  line-height: 0.9;
  letter-spacing: -3px;
  color: var(--red);
  text-shadow: 0 0 60px rgba(255,61,61,0.55), 0 0 120px rgba(255,61,61,0.2);
  margin-bottom: 18px;
}
.hero-sub {
  font-size: 16px;
  color: var(--muted);
  font-weight: 300;
  line-height: 1.7;
  max-width: 500px;
  font-family: var(--font-body);
}
.hero-meta {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 10px;
}
.live-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 9px 18px;
  border: 1px solid rgba(0,212,180,0.35);
  border-radius: 999px;
  background: rgba(0,212,180,0.07);
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--teal);
  letter-spacing: 1px;
  text-transform: uppercase;
}
.live-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--teal);
  box-shadow: 0 0 10px var(--teal), 0 0 20px var(--teal);
  animation: pulse-dot 2s ease-in-out infinite;
}
.timestamp-badge {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--muted2);
  letter-spacing: 1px;
}
.location-badge {
  font-family: var(--font-body);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--muted);
}

/* ════════════════════════
   KPI STRIP
════════════════════════ */
.kpi-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-bottom: 2.5rem;
}
.kpi-card {
  position: relative;
  overflow: hidden;
  padding: 24px 26px 22px;
  border: 1px solid var(--border);
  border-radius: 22px;
  background: var(--surface);
  transition: all 0.2s ease;
  cursor: default;
}
.kpi-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  border-radius: 22px 22px 0 0;
}
.kpi-card.red::before   { background: linear-gradient(90deg, var(--red), transparent 70%); }
.kpi-card.teal::before  { background: linear-gradient(90deg, var(--teal), transparent 70%); }
.kpi-card.amber::before { background: linear-gradient(90deg, var(--amber), transparent 70%); }
.kpi-card.sky::before   { background: linear-gradient(90deg, var(--sky), transparent 70%); }
.kpi-card:hover { background: var(--surfaceHov); border-color: var(--borderBright); transform: translateY(-2px); }
.kpi-label {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--muted2);
  margin-bottom: 14px;
}
.kpi-value {
  font-family: var(--font-display);
  font-size: 54px;
  font-weight: 800;
  line-height: 1;
  color: var(--text);
  letter-spacing: -2px;
}
.kpi-value.red   { color: var(--red);   text-shadow: 0 0 40px var(--redGlow); }
.kpi-value.teal  { color: var(--teal);  text-shadow: 0 0 40px var(--tealGlow); }
.kpi-value.amber { color: var(--amber); text-shadow: 0 0 40px var(--amberGlow); }
.kpi-value.sky   { color: var(--sky);   text-shadow: 0 0 40px var(--skyGlow); }
.kpi-unit {
  font-family: var(--font-body);
  font-size: 13px;
  color: var(--muted);
  margin-top: 10px;
  font-weight: 300;
}
.kpi-trend {
  position: absolute;
  bottom: 20px; right: 22px;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--muted2);
}

/* ════════════════════════
   SECTION HEADERS
════════════════════════ */
.section-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
}
.section-label {
  font-family: var(--font-display);
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--text);
  white-space: nowrap;
}
.section-line { flex: 1; height: 1px; background: var(--border); }
.section-badge {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--muted2);
  letter-spacing: 1px;
  white-space: nowrap;
}

/* ════════════════════════
   MAP
════════════════════════ */
.map-wrap {
  border: 1px solid var(--border);
  border-radius: 24px;
  overflow: hidden;
  background: var(--bg3);
}
.map-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 22px;
  border-bottom: 1px solid var(--border);
  background: rgba(255,255,255,0.015);
}
.map-title {
  font-family: var(--font-display);
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
}
.map-meta {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--muted2);
}

/* ════════════════════════
   ALERT CARDS
════════════════════════ */
.alerts-wrap { display: flex; flex-direction: column; gap: 14px; }
.alert-card {
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 22px 24px;
  background: var(--surface);
  position: relative;
  overflow: hidden;
}
.alert-card.priority {
  border-color: rgba(255,61,61,0.22);
  background: linear-gradient(135deg, rgba(255,61,61,0.055) 0%, var(--surface) 60%);
}
.alert-card.action {
  border-color: rgba(0,212,180,0.2);
  background: linear-gradient(135deg, rgba(0,212,180,0.045) 0%, var(--surface) 60%);
}
.alert-card.info {
  border-color: rgba(56,189,248,0.18);
  background: linear-gradient(135deg, rgba(56,189,248,0.04) 0%, var(--surface) 60%);
}
.alert-type {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  margin-bottom: 8px;
  font-weight: 500;
}
.alert-type.red   { color: var(--red); }
.alert-type.teal  { color: var(--teal); }
.alert-type.sky   { color: var(--sky); }
.alert-title {
  font-family: var(--font-display);
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 10px;
  color: var(--text);
  line-height: 1.2;
}
.alert-body {
  font-family: var(--font-body);
  font-size: 14px;
  color: var(--muted);
  line-height: 1.75;
  font-weight: 400;
}
.alert-body strong { color: var(--text); font-weight: 600; }
.alert-ghost {
  position: absolute;
  top: 10px; right: 16px;
  font-family: var(--font-display);
  font-size: 58px;
  font-weight: 800;
  opacity: 0.04;
  line-height: 1;
  pointer-events: none;
}

/* ════════════════════════
   STAGING CARDS
════════════════════════ */
.staging-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}
.staging-card {
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 24px 26px;
  background: var(--surface);
  position: relative;
  overflow: hidden;
  transition: all 0.2s ease;
}
.staging-card:hover {
  background: var(--surfaceHov);
  border-color: var(--borderBright);
  transform: translateY(-2px);
}
.staging-card.primary {
  border-color: rgba(0,212,180,0.22);
  grid-column: 1 / -1;
}
.staging-card.primary::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--teal), transparent 70%);
}
.staging-rank-ghost {
  font-family: var(--font-display);
  font-size: 90px;
  font-weight: 800;
  line-height: 1;
  opacity: 0.05;
  position: absolute;
  bottom: -8px; right: 14px;
  letter-spacing: -6px;
  pointer-events: none;
}
.staging-badge {
  display: inline-flex;
  align-items: center;
  padding: 4px 12px;
  border-radius: 999px;
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: 14px;
}
.staging-badge.primary { border: 1px solid rgba(0,212,180,0.35); background: rgba(0,212,180,0.08); color: var(--teal); }
.staging-badge.support { border: 1px solid rgba(247,183,49,0.3); background: rgba(247,183,49,0.07); color: var(--amber); }
.staging-name {
  font-family: var(--font-display);
  font-size: 22px;
  font-weight: 700;
  margin-bottom: 4px;
  line-height: 1.2;
}
.staging-type {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--muted2);
  letter-spacing: 1px;
  margin-bottom: 14px;
}
.staging-story {
  font-size: 14px;
  color: var(--muted);
  line-height: 1.75;
  font-weight: 400;
}
.staging-story strong { color: var(--text); font-weight: 600; }

/* ════════════════════════
   IMPACT STATS
════════════════════════ */
.impact-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 14px;
  margin-top: 28px;
}
.impact-card {
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 24px 22px;
  background: var(--surface);
  text-align: center;
}
.impact-value {
  font-family: var(--font-display);
  font-size: 42px;
  font-weight: 800;
  color: var(--sky);
  text-shadow: 0 0 40px var(--skyGlow);
  letter-spacing: -1px;
  margin-bottom: 8px;
}
.impact-label {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--muted2);
}

/* ════════════════════════
   TABS
════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  gap: 0 !important;
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  padding-bottom: 0 !important;
  margin-bottom: 2.5rem !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -1px !important;
  border-radius: 0 !important;
  padding: 14px 26px !important;
  color: var(--muted) !important;
  font-family: var(--font-display) !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  transition: all 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text) !important; }
.stTabs [aria-selected="true"] {
  color: var(--text) !important;
  border-bottom-color: var(--red) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ════════════════════════
   BRIEFING / SITUATION
════════════════════════ */
.briefing-container { max-width: 860px; }
.briefing-block {
  border: 1px solid var(--border);
  border-radius: 22px;
  overflow: hidden;
  background: var(--surface);
  margin-bottom: 14px;
}
.briefing-header {
  padding: 18px 26px;
  border-bottom: 1px solid var(--border);
  background: rgba(255,255,255,0.018);
  display: flex;
  align-items: center;
  gap: 14px;
}
.briefing-icon {
  width: 34px; height: 34px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 15px;
  flex-shrink: 0;
}
.briefing-icon.red   { background: rgba(255,61,61,0.1); }
.briefing-icon.teal  { background: rgba(0,212,180,0.1); }
.briefing-icon.sky   { background: rgba(56,189,248,0.1); }
.briefing-header-text {
  font-family: var(--font-display);
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
}
.briefing-body {
  padding: 26px;
  font-size: 15px;
  color: var(--muted);
  line-height: 1.85;
  font-weight: 400;
  font-family: var(--font-body);
}
.briefing-body strong { color: var(--text); font-weight: 600; }

/* Hotspot items */
.hotspot-item {
  display: flex;
  align-items: center;
  gap: 18px;
  padding: 20px 24px;
  border: 1px solid var(--border);
  border-radius: 18px;
  background: var(--surface);
  margin-bottom: 10px;
  transition: all 0.15s ease;
}
.hotspot-item:hover { background: var(--surfaceHov); }
.hotspot-num {
  font-family: var(--font-display);
  font-size: 44px;
  font-weight: 800;
  line-height: 1;
  color: var(--text);
  opacity: 0.1;
  min-width: 46px;
  letter-spacing: -2px;
}
.hotspot-info { flex: 1; }
.hotspot-title {
  font-family: var(--font-display);
  font-size: 17px;
  font-weight: 700;
  margin-bottom: 5px;
}
.hotspot-desc {
  font-size: 14px;
  color: var(--muted);
  line-height: 1.6;
  font-family: var(--font-body);
}
.hotspot-risk { text-align: right; }
.hotspot-risk-val {
  font-family: var(--font-mono);
  font-size: 22px;
  font-weight: 500;
  color: var(--red);
}
.hotspot-risk-label {
  font-family: var(--font-mono);
  font-size: 9px;
  letter-spacing: 2px;
  color: var(--muted2);
  text-transform: uppercase;
}

.stCaption {
  color: var(--muted2) !important;
  font-family: var(--font-mono) !important;
  font-size: 10px !important;
  letter-spacing: 0.5px !important;
}
iframe { border-radius: 0 0 24px 24px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
PRED_CSV = Path("data/edge_predictions.csv")
EDGES_GPKG = Path("data/osm/madison_edges.gpkg")
EMS_RECS  = Path("data/ems_recommendations.csv")
COVERAGE  = Path("data/hotspot_coverage.csv")

# ─────────────────────────────────────────────
#  LOADERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_edge_predictions_gpkg():
    if not PRED_CSV.exists():
        return None, f"Missing {PRED_CSV}"
    if gpd is None:
        return None, "Missing geopandas. Install: pip install geopandas shapely pyproj fiona"
    if not EDGES_GPKG.exists():
        return None, f"Missing {EDGES_GPKG}"

    pred = pd.read_csv(PRED_CSV)
    needed = {"edge_id", "hour", "risk"}
    if not needed.issubset(pred.columns):
        return None, f"edge_predictions.csv missing columns: {sorted(list(needed - set(pred.columns)))}"

    pred["hour"] = pd.to_datetime(pred["hour"], errors="coerce")
    pred = pred.dropna(subset=["edge_id", "hour", "risk"]).copy()

    # Use latest hour from CSV (this is what your old version did)
    latest_hour = pred["hour"].max()
    pred = pred[pred["hour"] == latest_hour].copy()

    edges = gpd.read_file(EDGES_GPKG).to_crs("EPSG:4326")
    edges["edge_id"] = edges["u"].astype(str) + "_" + edges["v"].astype(str) + "_" + edges["key"].astype(str)

    gdf = edges.merge(pred[["edge_id", "hour", "risk"]], on="edge_id", how="inner")
    cent = gdf.geometry.centroid
    gdf["lat"] = cent.y
    gdf["lon"] = cent.x
    return gdf, None

@st.cache_data(show_spinner=False)
def load_ems_recommendations():
    if not EMS_RECS.exists():
        return None, f"Missing {EMS_RECS}"
    df = pd.read_csv(EMS_RECS)
    required = ["selected_rank","name","asset_type","lat","lon"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        return None, f"ems_recommendations.csv missing {miss}"
    return df, None

@st.cache_data(show_spinner=False)
def load_hotspot_coverage():
    if not COVERAGE.exists():
        return None
    return pd.read_csv(COVERAGE)

def percentile_weight(risk_series: pd.Series):
    r = risk_series.astype(float).clip(lower=0)
    p60  = float(np.nanpercentile(r, 60))
    p995 = float(np.nanpercentile(r, 99.5))
    def w(x):
        if x <= p60:  return 0.25
        if x >= p995: return 1.0
        t = (x - p60) / (p995 - p60 + 1e-9)
        return float(np.sqrt(np.clip(t, 0, 1)))
    return r.map(w), (p60, p995)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "scenario" not in st.session_state:
    st.session_state.scenario = "Normal operations"
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Heatmap + Roads"

SCENARIOS   = ["Normal operations", "Rush hour surge", "Winter conditions", "Low visibility event"]
SCEN_ICONS  = ["🟢", "🔴", "❄️", "🌫️"]
VIEW_MODES  = ["Heatmap + Roads", "Roads only"]
VIEW_ICONS  = ["🔥", "🗺️"]

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    now_sb = datetime.now()
    st.markdown("""
    <div class="sb-wrap">
      <div class="sb-logo">
        <span class="sb-logo-text">PULSE</span>
        <span class="sb-logo-dot"></span>
      </div>
      <div class="sb-tagline">EMS Command Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding: 0 0.25rem;">', unsafe_allow_html=True)

    # ── Scenario ──
    st.markdown('<div class="sb-section-label">Active Scenario</div>', unsafe_allow_html=True)
    for scen, icon in zip(SCENARIOS, SCEN_ICONS):
        active = st.session_state.scenario == scen
        label  = f"{icon}  {scen}{'  ✓' if active else ''}"
        if st.button(label, key=f"scen_{scen}"):
            st.session_state.scenario = scen
            st.rerun()

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Map view ──
    st.markdown('<div class="sb-section-label">Map View</div>', unsafe_allow_html=True)
    for vm, icon in zip(VIEW_MODES, VIEW_ICONS):
        active = st.session_state.view_mode == vm
        label  = f"{icon}  {vm}{'  ✓' if active else ''}"
        if st.button(label, key=f"view_{vm}"):
            st.session_state.view_mode = vm
            st.rerun()

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    # ── Status ──
    st.markdown('<div class="sb-section-label">System Status</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-status">
      <div class="sb-status-row">
        <span>Location</span><span class="sb-status-val">Madison, WI</span>
      </div>
      <div class="sb-status-row">
        <span>Local Time</span><span class="sb-status-val">{now_sb.strftime("%H:%M:%S")}</span>
      </div>
      <div class="sb-status-row">
        <span>Feed</span><span class="sb-status-val green">● Live</span>
      </div>
      <div class="sb-status-row">
        <span>Model</span><span class="sb-status-val">v2.4.1</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section-label">Data</div>', unsafe_allow_html=True)
    if st.button("⟳   Refresh Outputs", key="refresh_btn"):
        st.cache_data.clear()
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* Hide sidebar + collapse control */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* Remove the left padding Streamlit reserves for sidebar */
    .block-container { padding-left: 2.5rem !important; }
    </style>
    """, unsafe_allow_html=True)

scenario  = st.session_state.scenario
view_mode = st.session_state.view_mode

# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────
gdf, err  = load_edge_predictions_gpkg()
ems, err2 = load_ems_recommendations()
cov       = load_hotspot_coverage()

if err:
    st.error(err); st.stop()
if err2:
    st.error(err2); st.stop()

latest_hour = None
dfh = gdf.copy()

dfh = dfh.dropna(subset=["risk","lat","lon"]).copy()

# Drop the zero-mass that kills the heat layer
dfh = dfh.sort_values("risk", ascending=False).head(700).copy()

mean_risk = float(dfh["risk"].mean()) if len(dfh) else 0.0
max_risk  = float(dfh["risk"].max())  if len(dfh) else 0.0

mean_tt = p90_tt = None
if cov is not None and "best_travel_time_s" in cov.columns:
    tt = cov["best_travel_time_s"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(tt):
        mean_tt = float(tt.mean())
        p90_tt  = float(tt.quantile(0.90))

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
now = datetime.now()
scenario_tag = scenario.upper().replace(" ", "&nbsp;")
st.markdown(f"""
<div class="hero">
  <div class="hero-inner">
    <div class="hero-brand">
      <div class="hero-eyebrow">EMS Command Intelligence</div>
      <div class="hero-title">PULSE</div>
      <div class="hero-sub">
        City-scale crash risk radar for faster EMS staging,
        smarter pre-positioning, and safer outcomes across Madison.
      </div>
    </div>
    <div class="hero-meta">
      <div class="live-badge"><span class="live-dot"></span>Live Preview</div>
      <div class="timestamp-badge">{now.strftime("%Y-%m-%d  %H:%M:%S")}</div>
      <div class="location-badge">📍 Madison, WI</div>
      <div style="margin-top:6px;font-family:var(--font-mono);font-size:10px;color:var(--muted2);letter-spacing:2px;text-transform:uppercase;border:1px solid var(--border);border-radius:8px;padding:6px 12px;background:rgba(255,255,255,0.025);">{scenario_tag}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  KPI STRIP
# ─────────────────────────────────────────────
avg_drive_str = f"{mean_tt/60:.1f}" if mean_tt is not None else "—"
p90_str = f"p90: {p90_tt/60:.1f}m" if p90_tt else "—"

st.markdown(f"""
<div class="kpi-strip">
  <div class="kpi-card red">
    <div class="kpi-label">Hotspots Monitored</div>
    <div class="kpi-value">{len(dfh):,}</div>
    <div class="kpi-unit">road segments · {scenario}</div>
    <div class="kpi-trend">↑ active</div>
  </div>
  <div class="kpi-card amber">
    <div class="kpi-label">Average Risk Level</div>
    <div class="kpi-value amber">{mean_risk:.3f}</div>
    <div class="kpi-unit">normalized risk score</div>
    <div class="kpi-trend">per segment</div>
  </div>
  <div class="kpi-card red">
    <div class="kpi-label">Peak Hotspot Intensity</div>
    <div class="kpi-value red">{max_risk:.3f}</div>
    <div class="kpi-unit">highest risk segment</div>
    <div class="kpi-trend">⚠ critical</div>
  </div>
  <div class="kpi-card sky">
    <div class="kpi-label">Avg Drive to Hotspot</div>
    <div class="kpi-value sky">{avg_drive_str}</div>
    <div class="kpi-unit">minutes · current staging</div>
    <div class="kpi-trend">{p90_str}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  Overview  ", "  Staging  ", "  Situation  "])

BASEMAP = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
heat_points = dfh[["lat","lon","risk","geometry"]].dropna(subset=["lat","lon","risk"]).copy()
heat_points["weight"], (p60, p995) = percentile_weight(heat_points["risk"])

def build_map():
    layers = []
    if view_mode in ["Roads only", "Heatmap + Roads"]:
        try:
            road_layer = pdk.Layer(
                "GeoJsonLayer",
                data=heat_points[["risk","geometry"]].to_json(),
                stroked=True, filled=False,
                get_line_color="[255, 80, 80, 80]",
                get_line_width=3, line_width_min_pixels=1, pickable=False,
            )
            layers.append(road_layer)
        except Exception:
            pass
    if view_mode == "Heatmap + Roads":
        layers.append(pdk.Layer(
            "HeatmapLayer", data=heat_points,
            get_position='[lon, lat]', get_weight="weight",
            radiusPixels=120, intensity=1.6, threshold=0.0,
            colorRange=[[0,25,50,0],[30,80,120,100],[255,120,0,180],[255,61,61,220],[255,200,0,255]],
        ))
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=heat_points.sort_values("risk", ascending=False).head(50),
            get_position='[lon, lat]', get_radius=30,
            get_fill_color=[255,61,61,200], stroked=True,
            get_line_color=[255,200,0,180], get_line_width=2,
            pickable=True, auto_highlight=True,
        ))
    view = pdk.ViewState(
    latitude=43.0731,
    longitude=-89.4012,
    zoom=11.8,
    pitch=50,
    bearing=5,
    )
    return pdk.Deck(
        layers=layers, initial_view_state=view,
        tooltip={"html":"<div style='font-family:monospace;font-size:11px;background:#0a0d18;border:1px solid rgba(255,61,61,0.3);border-radius:8px;padding:8px 12px;color:#eceef4;'>Risk: <b style='color:#ff3d3d'>{risk}</b></div>","style":{"background":"transparent"}},
        map_style=BASEMAP,
    )

# ── TAB 1: OVERVIEW ──────────────────────────
with tab1:
    col_map, col_alerts = st.columns([2.2, 1.0], gap="large")

    with col_map:
        forecast_str = str(latest_hour) if latest_hour else "—"
        st.markdown(f"""
        <div class="map-wrap">
          <div class="map-header">
            <div class="map-title">City Risk Radar</div>
            <div class="map-meta">Forecast: {forecast_str} &nbsp;·&nbsp; {scenario}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.pydeck_chart(build_map(), use_container_width=True, height=560)

    with col_alerts:
        top5  = dfh.sort_values("risk", ascending=False).head(5)
        top_r = float(top5["risk"].iloc[0]) if len(top5) else 0.0
        n_crit = int((dfh["risk"] >= p995).sum()) if len(dfh) else 0
        st.markdown(f"""
        <div class="alerts-wrap">
          <div class="alert-card priority">
            <div class="alert-ghost">!</div>
            <div class="alert-type red">⚠&nbsp; Priority Alert</div>
            <div class="alert-title">High hotspot concentration</div>
            <div class="alert-body">Peak intensity at <strong>{top_r:.3f}</strong>. <strong>{n_crit}</strong> segments exceed the 99.5th percentile threshold. Heat layer is scaled for visual clarity — raw scores are in the Situation tab.</div>
          </div>
          <div class="alert-card action">
            <div class="alert-ghost">→</div>
            <div class="alert-type teal">✓&nbsp; Action Ready</div>
            <div class="alert-title">Staging plan is available</div>
            <div class="alert-body">Top-ranked staging sites have been computed to minimize drive times into hotspot clusters for the <strong>{scenario}</strong> scenario.</div>
          </div>
          <div class="alert-card info">
            <div class="alert-ghost">i</div>
            <div class="alert-type sky">ℹ&nbsp; How To Use</div>
            <div class="alert-title">Navigate the tabs above</div>
            <div class="alert-body">Open <strong>Staging</strong> to review and action unit placement. Use <strong>Situation</strong> for a team-ready read-aloud briefing.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 2: STAGING ───────────────────────────
with tab2:
    st.markdown("""
    <div class="section-header">
      <div class="section-label">Recommended Staging Plan</div>
      <div class="section-line"></div>
      <div class="section-badge">Optimized for current scenario</div>
    </div>
    """, unsafe_allow_html=True)

    top_sites = ems.sort_values("selected_rank").head(6).copy()
    drive_str = f"{mean_tt/60:.1f} min" if mean_tt else "—"

    st.markdown('<div class="staging-grid">', unsafe_allow_html=True)
    for _, r in top_sites.iterrows():
        rank      = int(r["selected_rank"])
        is_prim   = rank == 1
        b_cls     = "primary" if is_prim else "support"
        b_lbl     = "Primary Unit" if is_prim else f"Support Unit {rank}"
        c_cls     = "primary" if is_prim else ""
        story = f"Stage near <strong>{r['name']}</strong> to reduce drive time into the highest-risk cluster. Average hotspot coverage: <strong>{drive_str}</strong>."
        st.markdown(f"""
        <div class="staging-card {c_cls}">
          <div class="staging-rank-ghost">{rank}</div>
          <div class="staging-badge {b_cls}">{b_lbl}</div>
          <div class="staging-name">{r["name"]}</div>
          <div class="staging-type">{r["asset_type"]}</div>
          <div class="staging-story">{story}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if mean_tt is not None and p90_tt is not None:
        n_covered = len(cov) if cov is not None else 0
        st.markdown(f"""
        <div style="margin-top:2.5rem;">
          <div class="section-header">
            <div class="section-label">Expected Impact</div>
            <div class="section-line"></div>
          </div>
          <div class="impact-row">
            <div class="impact-card">
              <div class="impact-value">{mean_tt/60:.1f}m</div>
              <div class="impact-label">Avg Drive to Hotspot</div>
            </div>
            <div class="impact-card">
              <div class="impact-value">{p90_tt/60:.1f}m</div>
              <div class="impact-label">90th Pct Drive Time</div>
            </div>
            <div class="impact-card">
              <div class="impact-value">{n_covered}</div>
              <div class="impact-label">Hotspots Covered</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Impact estimates require hotspot_coverage.csv.")

# ── TAB 3: SITUATION ─────────────────────────
with tab3:
    st.markdown('<div class="briefing-container">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
      <div class="section-label">Situation Report</div>
      <div class="section-line"></div>
      <div class="section-badge">Read aloud to your team</div>
    </div>
    """, unsafe_allow_html=True)

    top      = dfh.sort_values("risk", ascending=False).head(5)
    top_risk = float(top["risk"].iloc[0]) if len(top) else 0.0
    avg_drv  = f"{mean_tt/60:.1f} min" if mean_tt is not None else "—"
    p90_drv  = f"{p90_tt/60:.1f} min" if p90_tt is not None else "—"
    hour_str = str(latest_hour) if latest_hour else "—"

    st.markdown(f"""
    <div class="briefing-block">
      <div class="briefing-header">
        <div class="briefing-icon red">⚠</div>
        <div class="briefing-header-text">Situation</div>
      </div>
      <div class="briefing-body">
        PULSE is forecasting road-segment crash risk for the next hour and highlighting
        hotspot clusters across Madison. At <strong>{hour_str}</strong>, our peak
        hotspot intensity is <strong>{top_risk:.3f}</strong>.
        Currently monitoring <strong>{len(dfh):,}</strong> road segments
        under the <strong>{scenario}</strong> scenario.
      </div>
    </div>
    <div class="briefing-block">
      <div class="briefing-header">
        <div class="briefing-icon teal">→</div>
        <div class="briefing-header-text">Plan</div>
      </div>
      <div class="briefing-body">
        Stage units at the recommended sites to reduce travel time into
        hotspot clusters. Current average drive time to hotspots is
        <strong>{avg_drv}</strong>. Under heavier demand, 90% of hotspot
        drives are under <strong>{p90_drv}</strong>.
      </div>
    </div>
    <div class="briefing-block">
      <div class="briefing-header">
        <div class="briefing-icon sky">◉</div>
        <div class="briefing-header-text">Why It Works</div>
      </div>
      <div class="briefing-body">
        Risk scores are driven by historical crashes snapped to the road network,
        weather conditions, and short-term crash momentum via lag features.
        Staging sites are then selected to minimize travel time into the
        highest-risk areas — balancing coverage across the entire hotspot footprint.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header" style="margin-top:2.5rem;">
      <div class="section-label">Top Hotspots</div>
      <div class="section-line"></div>
      <div class="section-badge">Highest-risk segments this window</div>
    </div>
    """, unsafe_allow_html=True)

    for i, row in top.reset_index(drop=True).iterrows():
        risk_val = float(row["risk"])
        st.markdown(f"""
        <div class="hotspot-item">
          <div class="hotspot-num">{i+1:02d}</div>
          <div class="hotspot-info">
            <div class="hotspot-title">Road Segment · Hotspot #{i+1}</div>
            <div class="hotspot-desc">This segment ranks among the highest-risk locations for the current forecast window. Its position falls inside a broader hotspot cluster visible on the city radar map.</div>
          </div>
          <div class="hotspot-risk">
            <div class="hotspot-risk-val">{risk_val:.3f}</div>
            <div class="hotspot-risk-label">Risk Score</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
