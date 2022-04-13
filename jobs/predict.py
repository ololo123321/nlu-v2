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
def main(overrides: DictConfig):
    # Некоторые параметры конфига модели зависят от выборки:
    # * число лейблов в задаче ner, re
    # * маппинги index -> label в тех же задачах

    # Также некоторе параметры архитектуры могут быть переопределены в командной строке при обучении:
    # * число и размерность новых слоёв
    # * дропауты (на инференсе нужно знать, какой дропаут был при обучении)

    # Для того, чтобы гарантировать отсутствие конфлитктов между обучением и инференсом решил сделать следующее:
    # * конфиг обучения сохранять в папку модели в файле config.yaml
    # * при инференсе подгружать этот конфиг, и обновлять в нём только специфичные для инференса поля,
    #   представленные в файле predict.yaml

    # load config
    cfg = OmegaConf.load(os.path.join(overrides.model_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, overrides)

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
    #    Но всё равно лучше явно написать doc_level=False, чтоб было понятней.
    ds = ds \
        .load(cfg.test_data_path, limit=cfg.num_examples_test) \
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
    ds.save(cfg.path_out)


if __name__ == "__main__":
    main()
