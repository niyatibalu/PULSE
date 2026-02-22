import pandas as pd

INFILE = "data/raw/wisdot_crash.csv"
OUTFILE = "data_upload/wisdot_crash_madison.csv"

# Rough Madison bounding box (works even if county/city fields are missing)
LAT_MIN, LAT_MAX = 42.98, 43.20
LON_MIN, LON_MAX = -89.60, -89.20

# keep only these columns if they exist (shrinks file a lot)
KEEP_HINTS = [
    "crash", "date", "time", "hour", "lat", "lon", "longitude", "latitude",
    "severity", "injury", "fatal", "road", "light", "weather",
    "county", "city", "municip", "route", "speed", "surface"
]

def find_col(cols, names):
    lower = {c.lower(): c for c in cols}
    for n in names:
        if n in lower:
            return lower[n]
    return None

def main():
    # read header to detect columns
    head = pd.read_csv(INFILE, nrows=0, low_memory=False)
    cols = head.columns.tolist()

    # detect lat/lon columns
    lat_col = find_col(cols, ["latitude", "lat", "y"])
    lon_col = find_col(cols, ["longitude", "lon", "x"])

    # choose a smaller subset of columns if possible
    keep_cols = []
    for c in cols:
        cl = c.lower()
        if any(h in cl for h in KEEP_HINTS):
            keep_cols.append(c)
    # always keep lat/lon if we have them
    for c in [lat_col, lon_col]:
        if c and c not in keep_cols:
            keep_cols.append(c)

    if not keep_cols:
        keep_cols = cols  # fallback: keep everything

    # chunk read to avoid memory blowups
    out_chunks = []
    for chunk in pd.read_csv(INFILE, usecols=keep_cols, chunksize=200_000, low_memory=False):
        if lat_col and lon_col and lat_col in chunk.columns and lon_col in chunk.columns:
            chunk[lat_col] = pd.to_numeric(chunk[lat_col], errors="coerce")
            chunk[lon_col] = pd.to_numeric(chunk[lon_col], errors="coerce")
            chunk = chunk.dropna(subset=[lat_col, lon_col])
            chunk = chunk[
                (chunk[lat_col].between(LAT_MIN, LAT_MAX)) &
                (chunk[lon_col].between(LON_MIN, LON_MAX))
            ]
        # else: no lat/lon detected -> keep chunk (can't filter spatially)
        out_chunks.append(chunk)

    df = pd.concat(out_chunks, ignore_index=True) if out_chunks else pd.DataFrame(columns=keep_cols)

    # normalize column names a bit (optional)
    if lat_col and lat_col != "latitude":
        df = df.rename(columns={lat_col: "latitude"})
    if lon_col and lon_col != "longitude":
        df = df.rename(columns={lon_col: "longitude"})

    df.to_csv(OUTFILE, index=False)
    print("Saved ->", OUTFILE)
    print("Rows:", len(df))
    print("Cols:", len(df.columns))

if __name__ == "__main__":
    main()
