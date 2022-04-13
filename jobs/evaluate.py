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
    # examples_gold = load_collection(
    #     data_dir=cfg.gold_data_path,
    #     n=None,
    #     tokens_expression=None,
    #     ignore_bad_examples=False,
    #     read_fn=read_file_v3,  # TODO: конфигурировать
    #     verbose_fn=logger.info
    # )
    # examples_pred = load_collection(
    #     data_dir=cfg.pred_data_dir,
    #     n=None,
    #     tokens_expression=None,
    #     ignore_bad_examples=False,
    #     read_fn=read_file_v3,
    #     verbose_fn=logger.info
    # )
    # assert len(examples_gold) == len(examples_pred)
    ds_gold = hydra.utils.instantiate(cfg.dataset, data=None, tokenizer=None)
    ds_gold = ds_gold.load(cfg.test_data_path, limit=cfg.num_examples_test)

    ds_pred = hydra.utils.instantiate(cfg.dataset, data=None, tokenizer=None)
    ds_pred = ds_pred.load(cfg.test_data_path, limit=cfg.num_examples_test)

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
