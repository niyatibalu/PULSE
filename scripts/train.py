import logging
from pathlib import Path

from app.config import settings
from app.logging_config import configure_logging
from app.data.loader import load_accidents_csv
from app.ml.trainer import train_ensemble, save_bundle


def main() -> None:
    configure_logging()
    log = logging.getLogger("train")

    df = load_accidents_csv(settings.data_path)
    log.info("Loaded data: %d rows", len(df))

    bundle, auc = train_ensemble(
        df_raw=df,
        lat_bins=settings.lat_bins,
        lon_bins=settings.lon_bins,
        n_models=settings.n_models,
        seed=settings.random_seed,
    )

    Path("models").mkdir(exist_ok=True)
    save_bundle(bundle, settings.model_path)
    log.info("Saved ensemble to %s", settings.model_path)
    log.info("Time-series CV ROC-AUC: %.4f", auc)


if __name__ == "__main__":
    main()