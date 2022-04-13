import os
import sys
import logging
import hydra
import json
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger("evaluate")


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    logger.info("load data...")
    ds_gold = hydra.utils.instantiate(cfg.dataset, data=None, tokenizer=None)
    ds_gold = ds_gold.load(cfg.gold_data_path)

    ds_pred = hydra.utils.instantiate(cfg.dataset, data=None, tokenizer=None)
    ds_pred = ds_pred.load(cfg.predictions_path)

    assert len(ds_gold.data) == len(ds_pred.data)

    evaluator = hydra.utils.instantiate(cfg.evaluator)
    metrics = evaluator(ds_gold.data, ds_pred.data)
    logger.info(metrics)

    if cfg.metrics_path is not None:
        logger.info(f"saving metrics to {cfg.metrics_path}")
        with open(cfg.metrics_path, "w") as f:
            json.dump(metrics, f)
    else:
        logger.info("saving ignored due to output_path is not provided")


if __name__ == "__main__":
    main()
