import os
import json
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model.base import ModeKeys
from src.model.utils import get_session
from src.data.io import to_brat_v2

logger = logging.getLogger("predict")


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    logger.info("load data...")
    ds = hydra.utils.instantiate(cfg.dataset, data=None, tokenizer=tokenizer, mode=ModeKeys.TEST)
    ds.load(data_dir=cfg.test_data_dir, limit=cfg.num_examples_test)  # TODO: в конфиг
    # TODO: прояснить логику: с одной стороны мы должны обработать каждый входной файл,
    #  с другой - не каждый файл может быть возможно засунуть в модель
    #  (например, пример без сущностей не засунуть в cr и re)
    ds.filter()
    ds.preprocess()
    ds.check()

    logger.info("setup model...")
    with open(os.path.join(cfg.model.pretrained_dir, "bert_config.json")) as f:
        bert_config = json.load(f)
    bert_config.update(cfg["model"]["bert"]["params_updates"])

    cfg["model"]["bert"]["pad_token_id"] = tokenizer.vocab["[PAD]"]
    cfg["model"]["bert"]["cls_token_id"] = tokenizer.vocab["[CLS]"]
    cfg["model"]["bert"]["sep_token_id"] = tokenizer.vocab["[SEP]"]
    cfg["model"]["bert"]["params"] = bert_config

    # TODO: подгрузка чекпоинта
    sess = get_session()
    model_cls = hydra.utils.instantiate(cfg.model_cls)
    model = model_cls(sess=sess, config=cfg)
    model.build(mode=ModeKeys.TRAIN)
    model.reset_weights(bert_dir=cfg.model.pretrained_dir)

    model.predict(ds.data)  # TODO: добавить специфичные для модели kwargs

    logger.info("saving predictions to brat format...")
    to_brat_v2(ds.data, output_dir=cfg.output_dir)


if __name__ == "__main__":
    main()
