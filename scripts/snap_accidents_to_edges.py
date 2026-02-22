import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

ACCIDENTS_CSV = "data/accidents.csv"
EDGES_GPKG = "data/osm/madison_edges.gpkg"
OUT_CSV = "data/accidents_snapped.csv"

def main():
    # Load accidents (must contain timestamp, latitude, longitude, accident)
    df = pd.read_csv(ACCIDENTS_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Convert accidents to GeoDataFrame
    g_acc = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )

    # Load road edges (with geometry)
    edges = gpd.read_file(EDGES_GPKG)

    # Ensure both are in same CRS
    if edges.crs is None:
        # OSMnx sometimes writes no CRS in file; assume lat/lon
        edges = edges.set_crs("EPSG:4326")

    g_acc = g_acc.to_crs(edges.crs)

    # Spatial join: nearest edge to each accident point
    snapped = gpd.sjoin_nearest(
        g_acc,
        edges[["u", "v", "key", "osmid", "highway", "length", "speed_kph", "travel_time", "geometry"]],
        how="left",
        distance_col="dist_to_edge",
    )

    # Output table for ML joins
    out = snapped.drop(columns=["geometry"])
    out.to_csv(OUT_CSV, index=False)

    print(f"Saved snapped accidents -> {OUT_CSV}")
    print(out[["u","v","key","dist_to_edge"]].head())

if __name__ == "__main__":
    main()