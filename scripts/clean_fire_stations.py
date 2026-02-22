import pandas as pd

INFILE = "data/raw/fire_stations.csv"
OUTFILE = "data/fire_stations_clean.csv"

def find_col(df, options):
    cols = {c.lower(): c for c in df.columns}
    for o in options:
        if o.lower() in cols:
            return cols[o.lower()]
    return None

def main():
    df = pd.read_csv(INFILE, low_memory=False)

    lat_col = find_col(df, ["latitude","lat","y"])
    lon_col = find_col(df, ["longitude","lon","x"])
    name_col = find_col(df, ["name","station","station name","facility","location"])

    if lat_col is None or lon_col is None:
        raise ValueError(f"Missing lat/lon. Columns: {list(df.columns)}")

    df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=["lat","lon"]).copy()

    df["name"] = df[name_col].astype(str) if name_col else [f"Fire_{i}" for i in range(len(df))]
    out = df[["name","lat","lon"]].copy()

    out.to_csv(OUTFILE, index=False)
    print("Saved ->", OUTFILE)
    print("Rows:", len(out))
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()