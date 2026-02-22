import re
import numpy as np
import pandas as pd

INFILE = "data/noaa_lcd.csv"          # raw NOAA LCD export
OUTFILE = "data/weather_hourly.csv"   # pipeline expects this

def f_to_c(f):
    return (f - 32.0) * (5.0 / 9.0)

def _to_float(x):
    """Extract first float from messy NOAA strings like '10.0', '10.0s', '10.0 (estimated)'."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


def _sanitize_temperature_c(x):
    x = _to_float(x)
    if np.isnan(x):
        return np.nan
    # Madison sanity range
    if x < -80 or x > 60:
        return np.nan
    return x


def _sanitize_visibility(x):
    """
    HourlyVisibility in LCD is usually in miles.
    We'll keep it as miles (consistent within pipeline) unless you prefer km.
    """
    x = _to_float(x)
    if np.isnan(x):
        return np.nan
     # sanity: visibility in miles rarely > 30-50, definitely not hundreds
    if x <= 0 or x > 80:
        return np.nan
    return x


def _sanitize_wind_speed(x):
    """
    HourlyWindSpeed in LCD is usually mph.
    """
    x = _to_float(x)
    if np.isnan(x):
        return np.nan
    if x < 0 or x > 150:
        return np.nan
    return x


def _sanitize_precip(x):
    """
    HourlyPrecipitation in LCD is typically in inches.
    Missing might be blank; traces sometimes appear as 'T'.
    """
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if not s:
        return 0.0
    if s.upper() == "T":   # trace
        return 0.0
    v = _to_float(s)
    if np.isnan(v):
        return 0.0
    if v < 0 or v > 10:    # 10 inches/hour is insane
        return 0.0
    return v

def main():
    df = pd.read_csv(INFILE, low_memory=False)

    # Filter station (you only want this one)
    if "STATION" in df.columns:
        df = df[df["STATION"].astype(str) == "72641014837"].copy()

    required = [
        "DATE",
        "HourlyDryBulbTemperature",
        "HourlyDewPointTemperature",
        "HourlyVisibility",
        "HourlyWindSpeed",
        "HourlyPrecipitation",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Columns={df.columns.tolist()[:80]}")

    # Parse datetime
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).copy()
    df["hour"] = df["DATE"].dt.floor("h")

    # Clean fields
    df["temperature"] = df["HourlyDryBulbTemperature"].map(_sanitize_temperature_c)
    df["dewpoint"] = df["HourlyDewPointTemperature"].map(_sanitize_temperature_c)
    df["visibility"] = df["HourlyVisibility"].map(_sanitize_visibility)
    df["wind_speed"] = df["HourlyWindSpeed"].map(_sanitize_wind_speed)
    df["precipitation"] = df["HourlyPrecipitation"].map(_sanitize_precip)
     # Aggregate to hourly (mean for state vars, sum for precipitation)
    hourly = (
        df.groupby("hour", as_index=False)
        .agg(
            temperature=("temperature", "mean"),
            dewpoint=("dewpoint", "mean"),
            visibility=("visibility", "mean"),
            wind_speed=("wind_speed", "mean"),
            precipitation=("precipitation", "sum"),
        )
        .sort_values("hour")
        .reset_index(drop=True)
    )

    # Convert Fahrenheit to Celsius (LCD hourly temperatures are typically °F)
    hourly["temperature"] = hourly["temperature"].map(lambda x: f_to_c(x) if pd.notna(x) else x)
    hourly["dewpoint"] = hourly["dewpoint"].map(lambda x: f_to_c(x) if pd.notna(x) else x)
    
    hourly.to_csv(OUTFILE, index=False)

    print(f"Saved -> {OUTFILE}")
    print("Rows:", len(hourly))
    print("Hour range:", hourly["hour"].min(), "to", hourly["hour"].max())
    print(hourly.head(5).to_string(index=False))
    print("\nDescribe:")
    print(hourly[["temperature", "dewpoint", "visibility", "wind_speed", "precipitation"]].describe())

if __name__ == "__main__":
    main()