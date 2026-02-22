import pandas as pd


def recommend_ems_zones(df: pd.DataFrame, n_units: int) -> pd.DataFrame:
    out = df.copy()
    out["priority"] = out["risk_mean"] * (1.0 + out["risk_uncertainty"])
    out["deploy_here"] = False

    top_idx = out.sort_values("priority", ascending=False).head(n_units).index
    out.loc[top_idx, "deploy_here"] = True
    return out