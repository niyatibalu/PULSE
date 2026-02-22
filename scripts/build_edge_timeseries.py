import pandas as pd

INFILE = "data/accidents_snapped.csv"
OUTFILE = "data/edge_hourly.csv"

def main():
    df = pd.read_csv(INFILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Round to hour for spatiotemporal modeling
    df["hour"] = df["timestamp"].dt.floor("H")

    # Edge id
    df["edge_id"] = df["u"].astype(str) + "_" + df["v"].astype(str) + "_" + df["key"].astype(str)

    # Aggregate accidents per edge per hour
    hourly = (
        df.groupby(["edge_id", "hour"])
          .agg(
              accidents=("accident", "sum"),
              mean_speed=("speed_kph", "mean"),
              mean_travel_time=("travel_time", "mean"),
              mean_length=("length", "mean"),
              highway=("highway", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]),
          )
          .reset_index()
    )

    hourly.to_csv(OUTFILE, index=False)
    print(f"Saved edge-hour table -> {OUTFILE}")
    print(hourly.head())

if __name__ == "__main__":
    main()