# scripts/build_hourly_weather_noaa2022.py
import numpy as np
import pandas as pd

INFILE = "data/noaa2022.csv"
OUTFILE = "data/weather_hourly.csv"
STATION_ID = "72641014837"

def parse_tmp_dew(val):
    """
    TMP/DEW example: -0028,1  -> -2.8 C
    Missing often: 9999,9 or 99999,9
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    head = s.split(",")[0]
    try:
        n = int(head)
    except:
        return np.nan
    if n in (9999, 99999, 999999):
        return np.nan
    c = n / 10.0
    if c < -80 or c > 60:
        return np.nan
    return c

def parse_vis(val):
    """
    VIS example: 004000,1,9,9 -> 4000 meters
    Missing often 999999 or 99999
    Convert to km.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    head = s.split(",")[0]
    try:
        meters = int(head)
    except:
        return np.nan
    if meters in (9999, 99999, 999999, 9999999):
        return np.nan
    km = meters / 1000.0
    if km <= 0 or km > 80:
        return np.nan
    return km

def parse_wnd_speed(val):
    """
    WND example: 030,1,N,0051,1
    wind speed is 4th field in tenths of m/s: 0051 -> 5.1 m/s
    Missing often 9999
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    parts = s.split(",")
    if len(parts) < 4:
        return np.nan
    raw = parts[3].strip()
    if not raw.isdigit():
        return np.nan
    n = int(raw)
    if n in (9999, 99999):
        return np.nan
    ms = n / 10.0
    if ms < 0 or ms > 80:
        return np.nan
    return ms

def parse_precip_aa1(val):
    """
    AA1 example: 06,0000,2,1
    3rd field is precip depth in tenths of mm -> 2 => 0.2 mm
    Return mm.
    If missing, treat as 0.
    """
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    if not s:
        return 0.0
    parts = s.split(",")
    if len(parts) < 3:
        return 0.0
    raw = parts[2].strip()
    if not raw.isdigit():
        return 0.0
    n = int(raw)
    if n in (9999, 99999, 999999):
        return 0.0
    mm = n / 10.0
    if mm < 0 or mm > 300:
        return 0.0
    return mm

def main():
    df = pd.read_csv(INFILE, low_memory=False)

    if "STATION" in df.columns:
        df = df[df["STATION"].astype(str) == STATION_ID].copy()

    # DATE parsing
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).copy()
    df["hour"] = df["DATE"].dt.floor("h")

    # Parse fields
    for c in ["TMP","DEW","VIS","WND"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} in NOAA 2022 file.")
    # AA1 is optional
    df["temperature"] = df["TMP"].map(parse_tmp_dew)
    df["dewpoint"] = df["DEW"].map(parse_tmp_dew)
    df["visibility"] = df["VIS"].map(parse_vis)          # km
    df["wind_speed"] = df["WND"].map(parse_wnd_speed)    # m/s
    df["precipitation"] = df["AA1"].map(parse_precip_aa1) if "AA1" in df.columns else 0.0  # mm

    hourly = (
        df.groupby("hour", as_index=False)
          .agg(
              temperature=("temperature","mean"),
              dewpoint=("dewpoint","mean"),
              visibility=("visibility","mean"),
              wind_speed=("wind_speed","mean"),
              precipitation=("precipitation","sum"),
          )
          .sort_values("hour")
          .reset_index(drop=True)
    )

    hourly.to_csv(OUTFILE, index=False)
    print("Saved ->", OUTFILE)
    print("Rows:", len(hourly))
    print("Hour range:", hourly["hour"].min(), "->", hourly["hour"].max())
    print(hourly.head(5).to_string(index=False))
    print("\nDescribe:")
    print(hourly[["temperature","dewpoint","visibility","wind_speed","precipitation"]].describe())

if __name__ == "__main__":
    main()