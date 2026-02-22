import pandas as pd
import geopandas as gpd

EDGES_GPKG = "data/osm/madison_edges.gpkg"
PRED_CSV = "data/edge_predictions.csv"
OUT_GPKG = "data/edge_predictions.gpkg"

def main():
    edges = gpd.read_file(EDGES_GPKG).to_crs("EPSG:4326")
    edges["edge_id"] = edges["u"].astype(str) + "_" + edges["v"].astype(str) + "_" + edges["key"].astype(str)

    pred = pd.read_csv(PRED_CSV)
    if "hour" not in pred.columns:
        raise ValueError("edge_predictions.csv missing 'hour'")
    pred["hour"] = pd.to_datetime(pred["hour"], errors="coerce")
    pred = pred.dropna(subset=["hour"])

    latest_hour = pred["hour"].max()
    pred_latest = pred[pred["hour"] == latest_hour].copy()

    # Merge risk onto edges
    merged = edges.merge(pred_latest[["edge_id", "risk"]], on="edge_id", how="left")

    # CRITICAL: do not create fake zeros; drop edges that were not scored
    merged = merged.dropna(subset=["risk"]).copy()

    # Store forecast hour for the UI
    merged["hour"] = latest_hour

    merged.to_file(OUT_GPKG, driver="GPKG")
    print("Saved ->", OUT_GPKG)
    print("Latest hour:", latest_hour)
    print("Rows in map layer:", len(merged))
    print("Risk describe:\n", merged["risk"].describe())

if __name__ == "__main__":
    main()