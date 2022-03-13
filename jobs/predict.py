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
    ds = hydra.utils.instantiate(cfg.dataset, data=None, tokenizer=tokenizer)
    ds.load(data_dir=cfg.test_data_dir, limit=cfg.num_examples_test)  # TODO: в конфиг
    # TODO: прояснить логику: с одной стороны мы должны обработать каждый входной файл,
    #  с другой - не каждый файл возможно засунуть в модель
    #  (например, пример без сущностей не засунуть в cr и re). Поэтому, наверное, здесь лучше
    #  делать check, чтоб вылетала ошибка. С другой стороны можно как-то сделать так, чтобы документ
    #  просто копировался в выход, не проходя через модель
    ds.check()
    ds.preprocess()
    # TODO: здесь фильтровать только куски, но не документы
    ds.check()

    logger.info("setup model...")
    with open(os.path.join(cfg.model.pretrained_dir, "bert_config.json")) as f:
        bert_config = json.load(f)
    bert_config.update(cfg["model"]["bert"]["params_updates"])

    cfg["model"]["bert"]["pad_token_id"] = tokenizer.vocab["[PAD]"]
    cfg["model"]["bert"]["cls_token_id"] = tokenizer.vocab["[CLS]"]
    cfg["model"]["bert"]["sep_token_id"] = tokenizer.vocab["[SEP]"]
    cfg["model"]["bert"]["params"] = bert_config

    sess = get_session()
    model_cls = hydra.utils.instantiate(cfg.model_cls)
    model = model_cls.load(
        sess=sess,
        model_dir=cfg.model_dir,
        scope_to_load=cfg.scope_to_load,
        mode=ModeKeys.TEST
    )

    model.predict(ds.data)  # TODO: добавить специфичные для модели kwargs

    logger.info("saving predictions to brat format...")
    to_brat_v2(ds.data, output_dir=cfg.output_dir)


if __name__ == "__main__":
    main()
