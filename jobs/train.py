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


def maybe_update_config(cfg, encodings, tokenizer):
    if "parser" in cfg["model"]:
        cfg["model"]["parser"]["biaffine_type"]["num_labels"] = len(encodings["rel_enc"])
        cfg["model"]["bert"]["root_token_id"] = tokenizer.vocab["[unused1]"]
    if "ner" in cfg["model"]:
        if "biaffine" in cfg["model"]["ner"]:  # ner as span prediction
            cfg["model"]["ner"]["biaffine"]["num_labels"] = len(encodings["ner_enc"])
        else:  # ner as sequence labeling
            cfg["model"]["ner"]["num_labels"] = len(encodings["ner_enc"])


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    def get_dataset(path, limit):
        ds = hydra.utils.instantiate(cfg.dataset, data=None, tokenizer=tokenizer)
        # 1. подгрузка примеров
        # 2. фильтрация на уровне документов
        # 3. препроцессинг с разбиением документов на кусочки (с возможным перекрытием)
        # 4. фильтрация на уровне кусочков (длина, наличие сущностей в случае re и cr)
        ds = ds \
            .load(path, limit=limit) \
            .filter() \
            .preprocess() \
            .filter()
        return ds

    logger.info("load train data...")
    ds_train = get_dataset(cfg.train_data_path, limit=cfg.num_examples_train)
    encodings = ds_train.fit()

    if encodings:
        with open(os.path.join(cfg.output_dir, "encodings.json"), "w") as f:
            json.dump(encodings, f, indent=4, ensure_ascii=False)

    logger.info("load valid data...")
    ds_valid = get_dataset(cfg.valid_data_path, limit=cfg.num_examples_valid)

    logger.info("setup model...")
    with open(os.path.join(cfg.model.pretrained_dir, "bert_config.json")) as f:
        bert_config = json.load(f)
    bert_config.update(cfg["model"]["bert"]["params_updates"])

    cfg["model"]["bert"]["pad_token_id"] = tokenizer.vocab["[PAD]"]
    cfg["model"]["bert"]["cls_token_id"] = tokenizer.vocab["[CLS]"]
    cfg["model"]["bert"]["sep_token_id"] = tokenizer.vocab["[SEP]"]
    cfg["model"]["bert"]["params"] = DictConfig(bert_config)
    cfg["training"]["num_train_samples"] = sum(len(x.chunks) for x in ds_train.data)

    maybe_update_config(cfg, encodings=encodings, tokenizer=tokenizer)

    # save config
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # TODO: подгрузка чекпоинта
    sess = get_session()
    model_cls = hydra.utils.instantiate(cfg.model_cls)
    model = model_cls(sess=sess, config=cfg, **encodings)
    model.build(mode=ModeKeys.TRAIN)
    model.reset_weights(bert_dir=cfg.model.pretrained_dir)

    model.train(
        examples_train=ds_train.data,
        examples_valid=ds_valid.data,
        model_dir=cfg.output_dir,
        scope_to_save=cfg.scope_to_save,
        verbose=True,
        verbose_fn=None
    )


if __name__ == "__main__":
    main()
