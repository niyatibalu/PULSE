import pandas as pd

PRED_IN = "data/edge_predictions.csv"

def main():
    df = pd.read_csv(PRED_IN)
    df["hour"] = pd.to_datetime(df["hour"])

    # Evaluate on the last month only (so it resembles deployment)
    cutoff = df["hour"].max() - pd.Timedelta(days=30)
    test = df[df["hour"] >= cutoff].copy()

    for k in [50, 100, 250, 500]:
        top = test.sort_values("risk", ascending=False).head(k)
        hits = (top["y"] == 1).sum()
        print(f"Precision@{k}: {hits}/{k} = {hits/k:.4f}")

if __name__ == "__main__":
    main()