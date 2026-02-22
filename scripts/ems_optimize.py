import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd
from pathlib import Path

GRAPHML = "data/osm/madison_drive.graphml"
PRED_IN = "data/edge_predictions.csv"
EDGES_GPKG = "data/osm/madison_edges.gpkg"

FIRE_IN = "data/fire_stations_clean.csv"
EMS_IN = "data/ems_service_map_clean.csv"

OUT_REC = "data/ems_recommendations.csv"
OUT_COVER = "data/hotspot_coverage.csv"

TOP_N_HOTSPOTS = 200
K_UNITS = 6

def main():
    print("Loading graph...")
    G_full = ox.load_graphml(GRAPHML)

    # Use largest strongly connected component for directed graph
    print("Selecting largest connected component...")
    # Convert to undirected for component detection, then keep those nodes
    Gu = G_full.to_undirected()
    lcc_nodes = max(nx.connected_components(Gu), key=len)
    G = G_full.subgraph(lcc_nodes).copy()

    print("Loading predictions...")
    pred = pd.read_csv(PRED_IN)
    pred["hour"] = pd.to_datetime(pred["hour"])
    latest_hour = pred["hour"].max()
    latest = pred[pred["hour"] == latest_hour].copy()
    latest = latest.sort_values("risk", ascending=False).head(TOP_N_HOTSPOTS)

    print("Loading edges (gpkg)...")
    edges = gpd.read_file(EDGES_GPKG)
    for c in ["u","v","key"]:
        if c not in edges.columns:
            raise ValueError(f"madison_edges.gpkg missing {c}")

    edges["edge_id"] = edges["u"].astype(str) + "_" + edges["v"].astype(str) + "_" + edges["key"].astype(str)

    hotspots = latest.merge(edges[["edge_id", "u", "v"]], on="edge_id", how="left").dropna(subset=["u","v"])
    hotspots["u"] = hotspots["u"].astype(int)

    # Keep only hotspots whose node is in the LCC
    hotspots = hotspots[hotspots["u"].isin(lcc_nodes)].copy()
    if len(hotspots) == 0:
        raise ValueError("No hotspot nodes are in the largest connected component. Something is off with edge_id merge.")

    hotspot_nodes = hotspots["u"].tolist()
    hotspot_weights = hotspots["risk"].to_numpy()

    print("Loading candidate sites (fire + EMS)...")
    fire = pd.read_csv(FIRE_IN)
    fire["asset_type"] = "fire_station"

    ems = pd.read_csv(EMS_IN)
    ems["name"] = ems["Service Name"].astype(str) if "Service Name" in ems.columns else [f"EMS_{i}" for i in range(len(ems))]
    ems["asset_type"] = "ems_service"

    candidates = pd.concat(
        [fire[["name","lat","lon","asset_type"]], ems[["name","lat","lon","asset_type"]]],
        ignore_index=True
    ).dropna(subset=["lat","lon"])

    # Keep candidates near Madison
    candidates = candidates[candidates["lat"].between(42.98, 43.20) & candidates["lon"].between(-89.60, -89.20)].copy()

    print("Snapping candidates to graph nodes...")
    candidates["node"] = ox.nearest_nodes(G_full, X=candidates["lon"].to_list(), Y=candidates["lat"].to_list())
    candidates["node"] = candidates["node"].astype(int)

    # Keep only candidates in LCC
    candidates = candidates[candidates["node"].isin(lcc_nodes)].copy()
    if len(candidates) == 0:
        raise ValueError("No candidate sites snapped into the largest connected component.")

    print(f"Candidates in LCC: {len(candidates)} | Hotspots in LCC: {len(hotspot_nodes)}")

    station_names = candidates["name"].tolist()
    station_nodes = candidates["node"].tolist()
    candidates["name"] = candidates["name"].astype(str)
    candidates = candidates[candidates["name"].str.lower() != "nan"].copy()
    
    print("Computing travel times...")
    T = []
    for s in station_nodes:
        dist = nx.single_source_dijkstra_path_length(G, int(s), weight="travel_time")
        T.append([dist.get(int(h), np.inf) for h in hotspot_nodes])
    T = np.array(T)

    # Drop hotspots unreachable from ALL stations (still possible)
    reachable = np.isfinite(T).any(axis=0)
    if not reachable.all():
        hotspots = hotspots.iloc[np.where(reachable)[0]].copy()
        hotspot_nodes = hotspots["u"].tolist()
        hotspot_weights = hotspots["risk"].to_numpy()
        T = T[:, reachable]
        print("Filtered unreachable hotspots. Remaining:", len(hotspot_nodes))

    selected = []
    best_time = np.full(len(hotspot_nodes), np.inf)

    for _ in range(min(K_UNITS, len(station_names))):
        best_pick = None
        best_obj = np.inf

        for i, name in enumerate(station_names):
            if name in selected:
                continue
            new_best = np.minimum(best_time, T[i])
            obj = np.sum(hotspot_weights * new_best)
            if obj < best_obj:
                best_obj = obj
                best_pick = i

        if best_pick is None:
            print("No further improvement possible (all remaining stations equivalent/unreachable).")
            break

        selected.append(station_names[best_pick])
        best_time = np.minimum(best_time, T[best_pick])

    rec = candidates[candidates["name"].isin(selected)].copy()
    rec["selected_rank"] = rec["name"].apply(lambda x: selected.index(x) + 1)
    rec = rec.sort_values("selected_rank")

    Path("data").mkdir(exist_ok=True)
    rec.to_csv(OUT_REC, index=False)

    cover = hotspots[["edge_id","hour","risk","accidents","y"]].copy()
    cover["best_travel_time_s"] = best_time
    cover.to_csv(OUT_COVER, index=False)

    print("Latest hour:", latest_hour)
    print("Saved ->", OUT_REC)
    print("Saved ->", OUT_COVER)
    print("Selected:", selected)
    finite = best_time[np.isfinite(best_time)]
    print("Mean hotspot travel time (s):", float(np.mean(finite)) if len(finite) else None)

if __name__ == "__main__":
    main()