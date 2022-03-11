import os
import json
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model.base import ModeKeys
from src.model.utils import get_session

logger = logging.getLogger("train")


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    def get_dataset(data_dir, limit, mode):
        ds = hydra.utils.instantiate(cfg.dataset, data=None, tokenizer=tokenizer, mode=mode)
        ds.load(data_dir=data_dir, limit=limit)
        ds.filter()
        ds.preprocess()
        ds.check()
        return ds

    logger.info("load train data...")
    ds_train = get_dataset(data_dir=cfg.train_data_dir, limit=cfg.num_examples_train, mode=ModeKeys.TRAIN)

    logger.info("load valid data...")
    ds_valid = get_dataset(data_dir=cfg.valid_data_dir, limit=cfg.num_examples_valid, mode=ModeKeys.VALID)

    logger.info("setup model...")
    with open(os.path.join(cfg.model.pretrained_dir, "bert_config.json")) as f:
        bert_config = json.load(f)
    bert_config.update(cfg["model"]["bert"]["params_updates"])

    cfg["model"]["bert"]["pad_token_id"] = tokenizer.vocab["[PAD]"]
    cfg["model"]["bert"]["cls_token_id"] = tokenizer.vocab["[CLS]"]
    cfg["model"]["bert"]["sep_token_id"] = tokenizer.vocab["[SEP]"]
    cfg["model"]["bert"]["params"] = bert_config
    cfg["training"]["num_train_samples"] = sum(len(x.chunks) for x in ds_train.data)

    # TODO: подгрузка чекпоинта
    sess = get_session()
    model_cls = hydra.utils.instantiate(cfg.model_cls)
    model = model_cls(sess=sess, config=cfg)
    model.build(mode=ModeKeys.TRAIN)
    model.reset_weights(bert_dir=cfg.model.pretrained_dir)

    model.train(
        examples_train=ds_train.data,
        examples_valid=ds_valid.data,
        model_dir=cfg.output_dir,
        scope_to_save=None,
        verbose=True,
        verbose_fn=None
    )


if __name__ == "__main__":
    main()
