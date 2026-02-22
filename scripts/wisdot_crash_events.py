import pandas as pd

INFILE = "data/wisdot_crash_madison.csv"
OUTFILE = "data/events.csv"

def parse_time_hhmm(x):
    # CRSHTIME is HHMM but stored as int without leading zeros
    if pd.isna(x):
        return None
    try:
        s = str(int(float(x))).zfill(4)  # handles "626", "626.0", etc.
    except Exception:
        return None
    hh = int(s[:2])
    mm = int(s[2:])
    if hh > 23 or mm > 59:
        return None
    return f"{hh:02d}:{mm:02d}:00"

def main():
    df = pd.read_csv(INFILE, low_memory=False)

    # Required DT4000 columns for geo-coded crash events
    needed = ["CRSHDATE", "CRSHTIME", "LATDECDG", "LONDECDG"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {INFILE}: {missing}\n"
            f"Available columns (first 50): {list(df.columns)[:50]}"
        )

    # Parse date + time into timestamp
    df["date"] = pd.to_datetime(df["CRSHDATE"], errors="coerce")
    df["time_str"] = df["CRSHTIME"].apply(parse_time_hhmm)
    df["timestamp"] = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d") + " " + df["time_str"],
        errors="coerce",
    )

    # Coordinates
    df["latitude"] = pd.to_numeric(df["LATDECDG"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["LONDECDG"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["timestamp", "latitude", "longitude"])

    # OPTIONAL: keep only Madison-ish region (fast filter)
    # Comment this out if you want statewide modeling.
    df = df[(df["latitude"].between(42.98, 43.20)) & (df["longitude"].between(-89.60, -89.20))]

    # Build events table
    events = df[["timestamp", "latitude", "longitude"]].copy()
    events["accident"] = 1

    # Optional harm proxy (useful later for “harm reduction” optimization)
    if "TOTFATL" in df.columns and "TOTINJ" in df.columns:
        fatl = pd.to_numeric(df["TOTFATL"], errors="coerce").fillna(0)
        inj = pd.to_numeric(df["TOTINJ"], errors="coerce").fillna(0)
        events["harm"] = fatl * 10 + inj

    events.to_csv(OUTFILE, index=False)

    print(f"Saved -> {OUTFILE}")
    print("Rows:", len(events))
    print("Time range:", events["timestamp"].min(), "to", events["timestamp"].max())
    print(events.head())

if __name__ == "__main__":
    main()