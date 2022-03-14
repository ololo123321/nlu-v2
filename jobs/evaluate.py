import os
import sys
import logging
import hydra
import json
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.io import load_collection, read_file_v3

logger = logging.getLogger("evaluate")


@hydra.main(config_path="../config", config_name="evaluate")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    logger.info("load data...")
    examples_gold = load_collection(
        data_dir=cfg.gold_data_dir,
        n=None,
        tokens_expression=None,
        ignore_bad_examples=False,
        read_fn=read_file_v3,  # TODO: конфигурировать
        verbose_fn=logger.info
    )
    examples_pred = load_collection(
        data_dir=cfg.pred_data_dir,
        n=None,
        tokens_expression=None,
        ignore_bad_examples=False,
        read_fn=read_file_v3,
        verbose_fn=logger.info
    )
    assert len(examples_gold) == len(examples_pred)

    evaluator = hydra.utils.instantiate(cfg.evaluator)
    metrics = evaluator(examples_gold, examples_pred)
    logger.info(metrics)

    if cfg.output_path is not None:
        with open(cfg.output_path, "w") as f:
            json.dump(metrics, f)
