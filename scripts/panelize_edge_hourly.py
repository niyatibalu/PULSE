import pandas as pd

raise SystemExit("DEPRECATED: Use make_edge_train_compact.py + train_edge_model_compact.py")


POS_IN = "data/edge_hourly.csv"          # currently positives-only
WEATHER_IN = "data/weather_hourly.csv"   # defines the hourly timeline you care about
SNAPPED_IN = "data/accidents_snapped.csv"  # gives edge attributes per edge_id
OUT = "data/edge_hourly_panel.csv"

def mode_or_first(x):
    m = x.mode()
    return m.iloc[0] if len(m) else x.iloc[0]

def main():
    pos = pd.read_csv(POS_IN)
    pos["hour"] = pd.to_datetime(pos["hour"], errors="coerce")
    if pos["hour"].isna().any():
        raise ValueError("Bad hour in edge_hourly.csv")

    # Ensure accidents numeric
    pos["accidents"] = pd.to_numeric(pos["accidents"], errors="coerce").fillna(0).astype(int)

    weather = pd.read_csv(WEATHER_IN)
    weather["hour"] = pd.to_datetime(weather["hour"], errors="coerce")
    weather = weather.dropna(subset=["hour"]).sort_values("hour")
    hours = weather["hour"].unique()

    # Only edges that appear in positives
    edges = pos["edge_id"].astype(str).unique()

    # Build full MultiIndex panel (edges x hours)
    idx = pd.MultiIndex.from_product([edges, hours], names=["edge_id", "hour"])
    panel = pd.DataFrame(index=idx).reset_index()

    # Merge in positives and fill missing as 0
    panel = panel.merge(pos[["edge_id", "hour", "accidents"]], on=["edge_id","hour"], how="left")
    panel["accidents"] = panel["accidents"].fillna(0).astype(int)

    # Add road attributes per edge from snapped file (real attributes, not made up)
    snap = pd.read_csv(SNAPPED_IN)
    # edge_id should exist in snapped; if not, reconstruct it
    if "edge_id" not in snap.columns:
        snap["edge_id"] = snap["u"].astype(str) + "_" + snap["v"].astype(str) + "_" + snap["key"].astype(str)

    # Keep one representative row per edge_id for attributes
    cols = ["edge_id"]
    for c in ["speed_kph", "travel_time", "length", "highway"]:
        if c in snap.columns:
            cols.append(c)

    attrs = snap[cols].copy()
    if "highway" in attrs.columns:
        attrs = attrs.groupby("edge_id").agg(
            speed_kph=("speed_kph","mean") if "speed_kph" in attrs.columns else ("edge_id","size"),
            travel_time=("travel_time","mean") if "travel_time" in attrs.columns else ("edge_id","size"),
            length=("length","mean") if "length" in attrs.columns else ("edge_id","size"),
            highway=("highway", mode_or_first)
        ).reset_index()
        # clean dummy cols if missing
        for c in ["speed_kph","travel_time","length"]:
            if c not in attrs.columns:
                attrs.drop(columns=[c], inplace=True, errors="ignore")
    else:
        attrs = attrs.groupby("edge_id").mean(numeric_only=True).reset_index()

    panel = panel.merge(attrs, on="edge_id", how="left")

    panel.to_csv(OUT, index=False)
    print(f"Saved -> {OUT}")
    print("Rows:", len(panel))
    print("Accidents counts:", panel["accidents"].value_counts().head().to_dict())
    print("Positive rate:", (panel["accidents"] > 0).mean())

if __name__ == "__main__":
    main()