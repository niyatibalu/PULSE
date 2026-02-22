import pandas as pd
import geopandas as gpd

TRAIN_IN = "data/edge_train_compact.csv"
TRAIN_OUT = "data/edge_train_compact.csv"   # overwrite
EDGES_GPKG = "data/osm/madison_edges.gpkg"

KEEP_COLS = ["edge_id", "speed_kph", "travel_time", "length", "highway", "lanes", "oneway", "maxspeed", "name"]

def main():
    df = pd.read_csv(TRAIN_IN, low_memory=False)

    g = gpd.read_file(EDGES_GPKG)

    # Ensure u,v,key exist
    if not set(["u","v","key"]).issubset(g.columns):
        raise ValueError(f"GPKG missing u/v/key. Columns: {g.columns.tolist()}")

    # Build edge_id = u_v_key
    g["edge_id"] = g["u"].astype(str) + "_" + g["v"].astype(str) + "_" + g["key"].astype(str)

    # Keep only useful columns
    keep = [c for c in KEEP_COLS if c in g.columns]
    g = g[keep].drop_duplicates("edge_id")

    merged = df.merge(g, on="edge_id", how="left")

    # Fill numeric columns reasonably
    for c in ["speed_kph", "travel_time", "length", "lanes"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
            merged[c] = merged[c].fillna(merged[c].median())

    # Fill categorical
    for c in ["highway", "oneway", "maxspeed", "name"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna("unknown")

    merged.to_csv(TRAIN_OUT, index=False)

    print("Saved ->", TRAIN_OUT)
    for c in ["speed_kph","travel_time","length","highway"]:
        if c in merged.columns:
            print(c, "missing rate:", float(merged[c].isna().mean()))

if __name__ == "__main__":
    main()