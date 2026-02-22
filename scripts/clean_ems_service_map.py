import pandas as pd

INFILE = "data/raw/ems_service_map.csv"
OUTFILE = "data/ems_service_map_clean.csv"

def main():
    df = pd.read_csv(INFILE, encoding="utf-16", sep="\t", low_memory=False)

    # Prefer generated coordinates if present, else raw
    lat_col = "Latitude (generated)" if "Latitude (generated)" in df.columns else "Latitude"
    lon_col = "Longitude (generated)" if "Longitude (generated)" in df.columns else "Longitude"

    # Normalize numeric coords
    df["lat"] = pd.to_numeric(df.get(lat_col), errors="coerce")
    df["lon"] = pd.to_numeric(df.get(lon_col), errors="coerce")

    # Drop rows without coords (these are not usable for routing)
    df = df.dropna(subset=["lat", "lon"]).copy()

    # Keep a clean set of fields
    keep = []
    for c in [
        "Service Name",
        "Service County",
        "Service License Level",
        "Service Primary Type",
        "Primary Address City Name",
        "Elite Region",
        "Region Number",
        "Transport group",
        "Service Phone",
        "Service Email",
    ]:
        if c in df.columns:
            keep.append(c)

    out = df[keep].copy() if keep else pd.DataFrame(index=df.index)
    out["lat"] = df["lat"].astype(float)
    out["lon"] = df["lon"].astype(float)

    out.to_csv(OUTFILE, index=False)
    print("Saved ->", OUTFILE)
    print("Rows with coords:", len(out))
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()