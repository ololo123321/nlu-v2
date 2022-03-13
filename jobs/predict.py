import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model.base import ModeKeys
from src.model.utils import get_session
from src.data.io import to_brat_v2

logger = logging.getLogger("predict")


@hydra.main(config_path="../config", config_name="predict")
def main(cfg: DictConfig):
    # load config
    cfg_base = OmegaConf.load(os.path.join(cfg.model_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg_base, cfg)

    print(OmegaConf.to_yaml(cfg))
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    logger.info("load data...")
    ds = hydra.utils.instantiate(cfg.dataset, data=None, tokenizer=tokenizer)

    # 1. подгрузка примеров
    # 2. очистка примеров от лишней разметки, если таковая есть
    # 3. проверка на согласованность разметки и текста
    # 4. препроцессинг
    # 5. фильтрация на уровне кусков. Фильтрации на уровне документов не будет,
    #    потому что в случае косяков на уровне документов вылезла бы AssertionError на шаге 3.
    ds = ds \
        .load(data_dir=cfg.test_data_dir, limit=cfg.num_examples_test) \
        .clear() \
        .check() \
        .preprocess() \
        .filter(doc_level=False)

    logger.info("setup model...")
    sess = get_session()
    model_cls = hydra.utils.instantiate(cfg.model_cls)
    model = model_cls(sess=sess, config=cfg)
    model.build(mode=ModeKeys.TEST)
    model.restore_weights(model_dir=cfg.model_dir, scope=None)  # TODO: прояснить логику со scope

    model.predict(ds.data)  # TODO: добавить специфичные для модели kwargs

    logger.info(f"saving predictions to {cfg.output_dir}")
    to_brat_v2(ds.data, output_dir=cfg.output_dir)


if __name__ == "__main__":
    main()
