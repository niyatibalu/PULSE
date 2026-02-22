import pandas as pd
import numpy as np

COVER_IN = "data/hotspot_coverage.csv"
REC_IN = "data/ems_recommendations.csv"

def main():
    cover = pd.read_csv(COVER_IN)
    best = cover["best_travel_time_s"].replace([np.inf, -np.inf], np.nan).dropna()

    print("Optimized mean travel time (s):", float(best.mean()))
    print("Optimized median travel time (s):", float(best.median()))
    print("Optimized 90th percentile (s):", float(best.quantile(0.90)))
    print("Hotspots evaluated:", len(best))

if __name__ == "__main__":
    main()