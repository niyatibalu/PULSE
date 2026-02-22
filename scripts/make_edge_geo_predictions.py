import pandas as pd
import geopandas as gpd

PRED_IN = "data/edge_predictions.csv"
EDGES_GPKG = "data/osm/madison_edges.gpkg"
OUT_GPKG = "data/edge_predictions.gpkg"

def main():
    preds = pd.read_csv(PRED_IN)
    preds["hour"] = pd.to_datetime(preds["hour"])

    edges = gpd.read_file(EDGES_GPKG)
    edges["edge_id"] = edges["u"].astype(str) + "_" + edges["v"].astype(str) + "_" + edges["key"].astype(str)

    latest_hour = preds["hour"].max()
    latest = preds[preds["hour"] == latest_hour].copy()

    g = edges.merge(latest[["edge_id","risk","accidents"]], on="edge_id", how="left")
    g["risk"] = g["risk"].fillna(0)
    g["accidents"] = g["accidents"].fillna(0)

    g.to_file(OUT_GPKG, driver="GPKG")
    print("Saved ->", OUT_GPKG)
    print("Latest hour:", latest_hour)
    print("Top risks:")
    print(latest.sort_values("risk", ascending=False).head(10))

if __name__ == "__main__":
    main()