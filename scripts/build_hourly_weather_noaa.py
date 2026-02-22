import pandas as pd
import numpy as np

INFILE = "data/raw/noaa_72641014837_2022.csv"
OUTFILE = "data/weather_hourly.csv"
STATION_ID = "72641014837"

def parse_tmp(val):
    """
    ISD TMP format commonly looks like: "+0017,1" meaning 1.7C (tenths of C).
    Sometimes it's "0017,1" or "+0017".
    We'll extract the first signed integer and divide by 10.
    """
    if pd.isna(val):
        return np.nan
    s = str(val)
    # split at comma if present
    main = s.split(",")[0]
    # keep sign and digits
    try:
        x = int(main)
        return x / 10.0
    except Exception:
        # sometimes has leading + sign
        try:
            x = int(main.replace("+", ""))
            return x / 10.0
        except Exception:
            return np.nan

def parse_vis(val):
    """
    ISD VIS format often: "16000,1,N,1" where first number is meters.
    We'll take first field as meters and convert to km.
    """
    if pd.isna(val):
        return np.nan
    s = str(val)
    first = s.split(",")[0]
    try:
        m = int(first)
        return m / 1000.0
    except Exception:
        return np.nan

def parse_wnd(val):
    """
    ISD WND format often: "180,1,0050,1" where third field is speed in tenths of m/s.
    We'll extract third field, convert to m/s then to kph.
    """
    if pd.isna(val):
        return np.nan
    parts = str(val).split(",")
    if len(parts) < 3:
        return np.nan
    try:
        sp_tenths = int(parts[2])
        ms = sp_tenths / 10.0
        return ms * 3.6
    except Exception:
        return np.nan

def parse_aa1_precip(val):
    """
    ISD AA1 is precipitation accumulation.
    Format varies, but often: "01,0000,1,..." where second field is mm in tenths?
    In many ISD docs, precipitation depth is in mm and may be in tenths.
    We'll take the second field and divide by 10 to get mm, then convert to mm (keep mm).
    If parsing fails, NaN.
    """
    if pd.isna(val):
        return np.nan
    parts = str(val).split(",")
    if len(parts) < 2:
        return np.nan
    try:
        amt = int(parts[1])
        # common convention: tenths of mm
        return amt / 10.0
    except Exception:
        return np.nan

def main():
    df = pd.read_csv(INFILE, low_memory=False)

    if "STATION" not in df.columns or "DATE" not in df.columns:
        raise ValueError(f"NOAA file must contain STATION and DATE. Columns: {list(df.columns)[:50]}")

    # Filter station (string compare is safest)
    df["STATION"] = df["STATION"].astype(str)
    df = df[df["STATION"] == STATION_ID].copy()
    if len(df) == 0:
        raise ValueError(f"No rows found for STATION={STATION_ID}")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])

    # Parse core weather from ISD packed columns
    if "TMP" not in df.columns:
        raise ValueError("Missing TMP column in NOAA file")
    if "VIS" not in df.columns:
        raise ValueError("Missing VIS column in NOAA file")
    if "WND" not in df.columns:
        raise ValueError("Missing WND column in NOAA file")

    df["temperature"] = df["TMP"].apply(parse_tmp)          # Celsius
    df["dewpoint"] = df["DEW"].apply(parse_tmp) if "DEW" in df.columns else np.nan
    df["visibility"] = df["VIS"].apply(parse_vis)           # km
    df["wind_speed"] = df["WND"].apply(parse_wnd)           # kph

    # Precip: use AA1 if present, else 0
    if "AA1" in df.columns:
        df["precipitation"] = df["AA1"].apply(parse_aa1_precip)  # mm (approx)
        df["precipitation"] = df["precipitation"].fillna(0)
    else:
        df["precipitation"] = 0.0

    # Hourly aggregation
    df["hour"] = df["DATE"].dt.floor("h")
    weather = (
        df.groupby("hour")[["temperature","precipitation","visibility","wind_speed","dewpoint"]]
          .mean()
          .reset_index()
          .sort_values("hour")
    )

    weather.to_csv(OUTFILE, index=False)
    print("Saved ->", OUTFILE)
    print("Rows:", len(weather))
    print("Hour range:", weather["hour"].min(), "to", weather["hour"].max())
    print(weather.head())

if __name__ == "__main__":
    main()