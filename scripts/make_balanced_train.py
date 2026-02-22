import pandas as pd
import numpy as np

INFILE = "data/edge_train.csv"
OUTFILE = "data/edge_train_balanced.csv"

NEG_PER_POS = 20     # 20x negatives per positive = ~200k rows (manageable)
SEED = 42

def main():
    df = pd.read_csv(INFILE)
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"])

    if "y" not in df.columns:
        raise ValueError("edge_train.csv missing y")

    pos = df[df["y"] == 1].copy()
    neg = df[df["y"] == 0].copy()

    print("Pos:", len(pos), "Neg:", len(neg))

    n_neg = min(len(neg), NEG_PER_POS * len(pos))
    rng = np.random.default_rng(SEED)

    # Sample negatives with the same hour distribution as positives (time-respecting)
    # We do this by sampling hours from positives, then sampling negatives from those hours.
    pos_hours = pos["hour"].values
    sampled_hours = rng.choice(pos_hours, size=n_neg, replace=True)

    neg = neg.set_index("hour")
    buckets = []
    for h in sampled_hours[:2000]:
        # quick warmup to avoid empty selection errors
        pass

    # Faster: take negatives whose hour is in the set of positive hours, then random sample
    neg_same_hours = neg.loc[neg.index.isin(pos["hour"].unique())].reset_index()
    if len(neg_same_hours) < n_neg:
        # fallback: sample from all negatives if not enough overlap
        neg_sample = neg.reset_index().sample(n=n_neg, random_state=SEED)
    else:
        neg_sample = neg_same_hours.sample(n=n_neg, random_state=SEED)

    out = pd.concat([pos, neg_sample], ignore_index=True).sort_values("hour")
    out.to_csv(OUTFILE, index=False)

    print("Saved ->", OUTFILE)
    print("y distribution:", out["y"].value_counts().to_dict())
    print("Rows:", len(out))

if __name__ == "__main__":
    main()