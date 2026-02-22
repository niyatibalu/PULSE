import pandas as pd


def equity_summary(df: pd.DataFrame) -> pd.DataFrame:
    # This is a monitoring view: where risk is consistently high and uncertainty is high.
    return (
        df.groupby("grid_id")
        .agg(
            mean_risk=("risk_mean", "mean"),
            mean_uncertainty=("risk_uncertainty", "mean"),
            n=("risk_mean", "size"),
        )
        .sort_values("mean_risk", ascending=False)
        .reset_index()
    )