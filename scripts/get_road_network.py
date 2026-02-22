# scripts/get_road_network.py
"""
Download the drivable OpenStreetMap road network inside the Madison, WI
administrative boundary (OSM relation 3352040), then export:

- data/osm/madison_drive.graphml          (graph)
- data/osm/madison_nodes.gpkg             (nodes GeoPackage)
- data/osm/madison_edges.gpkg             (edges GeoPackage)
- data/osm/madison_edges.csv              (edges CSV without geometry)
- data/osm/madison_nodes.csv              (nodes CSV without geometry)

This script is written to be robust across multiple OSMnx versions.
"""

from __future__ import annotations

from pathlib import Path
import sys

import osmnx as ox


RELATION_ID = "R3352040"  # Relation: Madison (3352040) from your OSM page
OUTDIR = Path("data/osm")


def _set_osmnx_settings() -> None:
    # Version-safe settings
    try:
        ox.settings.use_cache = True
        ox.settings.log_console = True
    except Exception:
        # Older versions may not have settings attributes
        pass


def _geocode_relation_polygon(relation_id: str):
    # Get boundary polygon from OSM relation ID
    # Version-safe wrapper around geocode_to_gdf
    try:
        gdf = ox.geocode_to_gdf(relation_id, by_osmid=True)
    except TypeError:
        # Older versions may not support by_osmid kwarg, but often accept "Rxxxx" directly
        gdf = ox.geocode_to_gdf(relation_id)
    poly = gdf.geometry.iloc[0]
    return poly


def _download_drive_network(poly):
    # Download drivable network inside polygon
    # graph_from_polygon is stable across most versions
    G = ox.graph_from_polygon(poly, network_type="drive", truncate_by_edge=True)
    return G


def _add_speeds_and_travel_times(G):
    """
    Add edge speed and travel time attributes.

    Different OSMnx versions expose these functions in different modules.
    We'll try multiple locations in a safe order.
    """
    # Newer OSMnx: ox.routing.add_edge_speeds / add_edge_travel_times
    try:
        G = ox.routing.add_edge_speeds(G)
        G = ox.routing.add_edge_travel_times(G)
        return G
    except Exception:
        pass

    # Some versions: top-level ox.add_edge_speeds / ox.add_edge_travel_times
    try:
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        return G
    except Exception:
        pass

    # Older versions: ox.speed.add_edge_speeds / add_edge_travel_times
    try:
        G = ox.speed.add_edge_speeds(G)
        G = ox.speed.add_edge_travel_times(G)
        return G
    except Exception as e:
        raise RuntimeError(
            "Could not add speeds/travel times with your installed OSMnx version. "
            "Your network is still usable (it has lengths), but travel_time will be missing.\n"
            f"Underlying error: {e}"
        )


def _graph_to_gdfs(G):
    # graph_to_gdfs location varies by version
    try:
        nodes, edges = ox.graph_to_gdfs(G)
        return nodes, edges
    except Exception:
        pass

    try:
        nodes, edges = ox.convert.graph_to_gdfs(G)
        return nodes, edges
    except Exception as e:
        raise RuntimeError(f"Could not convert graph to GeoDataFrames: {e}")


def _save_outputs(G, nodes, edges, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    # Save graphml
    try:
        ox.save_graphml(G, outdir / "madison_drive.graphml")
    except Exception:
        # Some versions: save_graphml signature wants a string path
        ox.save_graphml(G, filepath=str(outdir / "madison_drive.graphml"))

    # Save GeoPackages
    nodes.to_file(outdir / "madison_nodes.gpkg", driver="GPKG")
    edges.to_file(outdir / "madison_edges.gpkg", driver="GPKG")

    # Save CSVs (drop geometry because plain CSV can't store shapely objects)
    nodes.drop(columns=["geometry"], errors="ignore").to_csv(outdir / "madison_nodes.csv", index=False)
    edges.drop(columns=["geometry"], errors="ignore").to_csv(outdir / "madison_edges.csv", index=False)


def main():
    _set_osmnx_settings()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Fetching boundary polygon for {RELATION_ID} ...")
    poly = _geocode_relation_polygon(RELATION_ID)

    print("[2/5] Downloading drivable road network inside boundary ...")
    G = _download_drive_network(poly)

    print("[3/5] Adding speed + travel time attributes (version-safe) ...")
    G = _add_speeds_and_travel_times(G)

    print("[4/5] Converting graph to node/edge GeoDataFrames ...")
    nodes, edges = _graph_to_gdfs(G)

    print("[5/5] Saving outputs ...")
    _save_outputs(G, nodes, edges, OUTDIR)

    # Quick verification prints
    print("\n=== Done ===")
    print(f"Saved to: {OUTDIR.resolve()}")
    print(f"Nodes: {len(nodes):,} | Edges: {len(edges):,}")
    print("Edge columns include speed_kph:", "speed_kph" in edges.columns)
    print("Edge columns include travel_time:", "travel_time" in edges.columns)
    print("Example edge columns:", list(edges.columns)[:25])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR:", e)
        sys.exit(1)