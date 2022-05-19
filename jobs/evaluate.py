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

    logger.info("loading gold data...")
    ds_gold = hydra.utils.instantiate(cfg.dataset, data=None, tokenizer=None)
    ds_gold = ds_gold.load(cfg.gold_data_path)
    if cfg.filter_gold_examples:
        logger.warning(
            "filter_gold_examples=True, so gold and pred examples might mismatch. "
            "Explicitly set evaluator.allow_examples_mismatch=True"
        )
        ds_gold.filter(doc_level=True, chunk_level=False)
        cfg.evaluator.allow_examples_mismatch = True

    logger.info("loading predictions...")
    ds_pred = hydra.utils.instantiate(cfg.dataset, data=None, tokenizer=None)
    ds_pred = ds_pred.load(cfg.predictions_path)

    evaluator = hydra.utils.instantiate(cfg.evaluator)
    metric = evaluator(ds_gold.data, ds_pred.data)
    logger.info(metric.string)

    if cfg.metrics_path is not None:
        logger.info(f"saving metrics to {cfg.metrics_path}")
        with open(cfg.metrics_path, "w") as f:
            json.dump(metric.value, f)
    else:
        logger.info("saving ignored due to output_path is not provided")


if __name__ == "__main__":
    main()
