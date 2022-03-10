import random
import os
import json
import shutil
from typing import Dict, List, Callable, Tuple, Iterable
from itertools import chain, groupby, combinations
from abc import ABC, abstractmethod
from collections import defaultdict

tf = import_tensorflow()
import tensorflow_hub as hub
import numpy as np

from bert.modeling import BertModel, BertConfig
from bert.optimization import create_optimizer

from src.model.utils import (
    get_batched_coords_from_labels,
    get_labels_mask,
    get_dense_labels_from_indices,
    get_entity_embeddings,
    get_padded_coords_3d,
    upper_triangular,
    get_entity_embeddings_concat_half,
    get_entity_pairs_mask,
    get_sent_pairs_to_predict_for,
    get_span_indices,
    get_additive_mask
)
from src.model.layers import GraphEncoder, GraphEncoderInputs, StackedBiRNN, MLP
from src.data.base import Example, Arc, Entity
from src.data.postprocessing import get_valid_spans
from src.data.io import to_conll
from src.metrics import (
    classification_report,
    classification_report_ner,
    f1_precision_recall_support,
    get_coreferense_resolution_metrics
)
from src.utils import get_entity_spans, get_connected_components, parse_conll_metrics, train_test_split


class BaseModel(ABC):
    """
    Interface for all models
    """

    model_scope = "model"
    ner_scope = "ner"
    re_scope = "re"

    def __init__(self, sess, config=None, ner_enc=None, re_enc=None):
        self.sess = sess
        self.config = config
        self.ner_enc = ner_enc
        self.re_enc = re_enc

        if config is not None:
            if "ner" in config["model"] and "re" in config["model"]:
                assert config["model"]["ner"]["loss_coef"] + config["model"]["re"]["loss_coef"] > 0.0

        if self.ner_enc is not None:
            self.inv_ner_enc = {v: k for k, v in self.ner_enc.items()}
        else:
            self.inv_ner_enc = None

        if self.re_enc is not None:
            self.inv_re_enc = {v: k for k, v in self.re_enc.items()}
        else:
            self.re_enc = None

        self.loss = None
        self.train_op = None

    # специфичные для каждой модели методы

    @abstractmethod
    def _build_graph(self):
        """построение вычислительного графа (без loss и train_op)"""

    @abstractmethod
    def _get_feed_dict(self, examples: List[Example], mode: str) -> Dict:
        """mode: {train, valid, test} (см. ModeKeys)"""

    @abstractmethod
    def _set_placeholders(self):
        pass

    @abstractmethod
    def _set_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def _set_train_op(self):
        pass

    @abstractmethod
    def predict(
            self,
            examples: List[Example],
            window: int = 1,
            batch_size: int = 16,
            **kwargs
    ) -> None:
        """
        Вся логика инференса должна быть реализована здесь.
        Предполагается, что модель училась не на целых документах, а на кусках (chunks).
        Следовательно, предикт модель должна делать тоже на уровне chunks.
        Но в конечном итоге нас интересуют предсказанные лейблы на исходных документах (examples).
        Поэтому схема такая:
        1. получить модельные предикты на уровне chunks
        2. аггрегировать результат из п.1 и записать на уровне examples

        :param examples: исходные документы. атрибут chunks должен быть заполнен!
        :param window: размер куска (в предложениях)
        :param batch_size:
        :param kwargs:
        :return:
        """

    @abstractmethod
    def evaluate(self, examples: List[Example], batch_size: int = 16) -> Dict:
        """
        Возвращаемый словарь должен обязательно содержать ключи "score" и "loss"
        :param examples: исходные документы. атрибут chunks должен быть заполнен!
        :param batch_size:
        :return:
        """

    # общие методы для всех моделей

    def build(self):
        with tf.variable_scope(self.model_scope):
            self._build_graph()
            self._set_loss()
            self._set_train_op()
        self.reset_weights()

    # альтернативная версия данной функции вынесена в src._old.wip
    def train(
            self,
            examples_train: List[Example],
            examples_valid: List[Example],
            train_op_name: str = "train_op",
            model_dir: str = None,
            scope_to_save: str = None,
            verbose: bool = True,
            verbose_fn: Callable = None,
    ):
        """

        :param examples_train:
        :param examples_valid:
        :param train_op_name:
        :param model_dir:
        :param scope_to_save:
        :param verbose:
        :param verbose_fn: вход - словарь с метриками (выход self.evaluate); выход - None. функция должна вывести
                           релевантные метрики в stdout
        :return:

        нет возможности переопределять batch_size, потому что есть следующая зависимость:
        batch_size -> num_train_steps -> lr schedule for adamw
        поэтому если хочется изменить batch_size, то нужно переопределить train_op. иными словами, проще сделать так:
        tf.reset_default_graph()
        sess = tf.Session()
        model = ...
        model.build()
        model.initialize()
        """
        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
            checkpoint_path = os.path.join(model_dir, "model.ckpt")
            print(f"model dir {model_dir} created")
        else:
            checkpoint_path = None
            print("model dir is None, so checkpoints will not be saved")

        if checkpoint_path is not None:
            if scope_to_save is not None:
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_to_save)
            else:
                var_list = tf.trainable_variables()
            saver = tf.train.Saver(var_list)
        else:
            saver = None

        # релзиовано так для возможности выбора train_op: обычная или с аккумуляцией градиентов.
        # TODO: реализовать вторую (уже реализовывал, когда решал dependency parsing, нужно скопипастить сюда)
        train_op = getattr(self, train_op_name)

        chunks_train = []
        for x in examples_train:
            assert len(x.chunks) > 0, f"[{x.id}] example didn't split by chunks!"
            chunks_train += x.chunks

        chunks_valid = []
        for x in examples_valid:
            assert len(x.chunks) > 0, f"[{x.id}] example didn't split by chunks!"
            chunks_valid += x.chunks

        batch_size = self.config["training"]["batch_size"]
        num_epoch_steps = int(len(chunks_train) / batch_size)
        best_score = -1
        num_steps_wo_improvement = 0
        verbose_fn = verbose_fn if verbose_fn is not None else print
        train_loss = []

        for epoch in range(self.config["training"]["num_epochs"]):
            for _ in range(num_epoch_steps):
                if len(chunks_train) > batch_size:
                    chunks_batch = random.sample(chunks_train, batch_size)
                else:
                    chunks_batch = chunks_train
                feed_dict = self._get_feed_dict(chunks_batch, mode=ModeKeys.TRAIN)
                try:
                    _, loss = self.sess.run([train_op, self.loss], feed_dict=feed_dict)
                    train_loss.append(loss)
                except Exception as e:
                    print("current batch:", [x.id for x in chunks_batch])
                    raise e

            # pycharm bug:
            # Cannot find reference {mean, std} in __init__.pyi | __init__.pxd
            # so, np.mean(train_loss) highlights yellow
            print(f"epoch {epoch} finished. mean train loss: {np.array(train_loss).mean()}. evaluation starts.")
            performance_info = self.evaluate(examples=examples_valid, batch_size=batch_size)
            if verbose:
                verbose_fn(performance_info)
            score = performance_info["score"]

            print("current score:", score)

            if score > best_score:
                print("!!! new best score:", score)
                best_score = score
                num_steps_wo_improvement = 0

                if saver is not None:
                    saver.save(self.sess, checkpoint_path)
                    print(f"saved new head to {checkpoint_path}")
            else:
                num_steps_wo_improvement += 1
                print("best score:", best_score)
                print("steps wo improvement:", num_steps_wo_improvement)

                if num_steps_wo_improvement == self.config["training"]["max_epochs_wo_improvement"]:
                    print("training finished due to max number of steps wo improvement encountered.")
                    break

            print("=" * 50)

        if saver is not None:
            print(f"restoring model from {checkpoint_path}")
            saver.restore(self.sess, checkpoint_path)

    def cross_validate(
            self,
            examples: List[Example],
            folds: Iterable,
            valid_frac: float = 0.15,
            verbose_fn: Callable = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        :param examples:
        :param folds:
        :param valid_frac:
        :param verbose_fn:
        :return:
        """
        for x in examples:
            assert len(x.chunks) > 0, f"[{x.id}] example didn't split by chunks!"

        scores_valid = []
        scores_test = []

        verbose_fn = verbose_fn if verbose_fn is not None else print

        for i, (train_files, test_files) in enumerate(folds):
            print(f"FOLDS {i}")

            train_files_set = set(train_files)
            test_files_set = set(test_files)

            examples_train_valid = [x for x in examples if x.filename in train_files_set]
            examples_test = [x for x in examples if x.filename in test_files_set]

            train_frac = 1.0 - valid_frac
            examples_train, examples_valid = train_test_split(
                examples_train_valid, train_frac=train_frac, seed=228
            )

            # TODO: lr schedule depends on num train steps, which depends on num train sample and batch size.

            self.reset_weights()

            self.train(
                examples_train=examples_train,
                examples_valid=examples_valid,
                train_op_name="train_op",
                model_dir=None,
                scope_to_save=None,
                verbose=False,
                verbose_fn=verbose_fn
            )

            # TODO: batch_size для инференса вынести в config
            d_valid = self.evaluate(examples=examples_valid, batch_size=16)
            verbose_fn(d_valid)
            d_test = self.evaluate(examples=examples_test, batch_size=16)
            verbose_fn(d_test)

            scores_valid.append(d_valid["re"]["metrics"]["micro"]["f1"])
            scores_test.append(d_test["re"]["metrics"]["micro"]["f1"])

            print("=" * 80)

        # pycharm bug:
        # Cannot find reference {mean, std} in __init__.pyi | __init__.pxd
        # so, np.mean(scores) highlights yellow
        scores_valid = np.array(scores_valid)
        scores_test = np.array(scores_test)

        print(f"scores valid: {scores_valid} (mean {scores_valid.mean()}, std {scores_valid.std()})")
        print(f"scores test: {scores_test} (mean {scores_test.mean()}, std {scores_test.std()})")

        return scores_valid, scores_test

    # TODO: разделить на сохранение весов и сохранение конфигов
    def save(self, model_dir: str, force: bool = True, scope_to_save: str = None):
        assert self.config is not None
        assert self.sess is not None

        if force:
            try:
                shutil.rmtree(model_dir)
            except:
                pass
        else:
            assert not os.path.exists(model_dir), \
                f'dir {model_dir} already exists. exception raised due to flag "force" was set to "True"'

        os.makedirs(model_dir, exist_ok=True)

        if scope_to_save is not None:
            scope = scope_to_save
        else:
            scope = self.model_scope

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list)
        checkpoint_path = os.path.join(model_dir, "model.ckpt")
        saver.save(self.sess, checkpoint_path)

        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

        if self.ner_enc is not None:
            with open(os.path.join(model_dir, "ner_enc.json"), "w") as f:
                json.dump(self.ner_enc, f, indent=4)

        if self.re_enc is not None:
            with open(os.path.join(model_dir, "re_enc.json"), "w") as f:
                json.dump(self.re_enc, f, indent=4)

    @classmethod
    def load(cls, sess, model_dir: str, scope_to_load: str = None):

        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)

        ner_enc_path = os.path.join(model_dir, "ner_enc.json")
        if os.path.exists(ner_enc_path):
            with open(ner_enc_path) as f:
                ner_enc = json.load(f)
        else:
            ner_enc = None

        re_enc_path = os.path.join(model_dir, "re_enc.json")
        if os.path.exists(re_enc_path):
            with open(re_enc_path) as f:
                re_enc = json.load(f)
        else:
            re_enc = None

        model = cls(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

        model.build()

        if scope_to_load is not None:
            scope = scope_to_load
        else:
            scope = model.model_scope

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.train.Saver(var_list)
        checkpoint_path = os.path.join(model_dir, "model.ckpt")
        saver.restore(sess, checkpoint_path)

        return model

    def reset_weights(self):
        global_vars = tf.global_variables()
        init_op = tf.variables_initializer(global_vars)
        self.sess.run(init_op)


# TODO: сделать базовый класс для joint моделей. потом от него отнаследовать ElmoJoinModel


class BertJointModel(BaseModel):
    """
    1. Поиск сущностей и триггеров событий (flat ner)
    2. Поиск отношений между сущностями и аргументов событий

    https://arxiv.org/abs/1812.11275
    """

    def __init__(self, sess, config=None, ner_enc=None, re_enc=None):
        """
        config = {
            "model": {
                "bert": {
                    "dir": "~/bert",
                    "dim": 768,
                    "attention_probs_dropout_prob": 0.5,  # default 0.1
                    "hidden_dropout_prob": 0.1,
                    "dropout": 0.1,
                    "scope": "bert",
                    "pad_token_id": 0,
                    "cls_token_id": 1,
                    "sep_token_id": 2
                },
                "ner": {
                    "use_crf": True,
                    "num_labels": 7,
                    "no_entity_id": 0,
                    "start_ids": [1, 2, 3],  # id лейблов первых токенов сущностей. нужно для векторизации сущностей
                    "prefix_joiner": "-",
                    "loss_coef": 1.0,
                    "use_birnn": True,
                    "rnn": {
                        "num_layers": 1,
                        "cell_dim": 128,
                        "dropout": 0.5,
                        "recurrent_dropout": 0.0
                    }
                },
                "re": {
                    "no_relation_id": 0,
                    "loss_coef": 10.0,
                    "use_birnn": True,
                    "use_entity_emb": True,
                    "use_entity_emb_layer_norm": True,
                    "entity_emb_dropout": 0.2,
                    "rnn": {
                        "num_layers": 1,
                        "cell_dim": 128,
                        "dropout": 0.5,
                        "recurrent_dropout": 0.0
                    },
                    "biaffine": {
                        "num_mlp_layers": 1,
                        "activation": "relu",
                        "head_dim": 128,
                        "dep_dim": 128,
                        "dropout": 0.3,
                        "num_labels": 7,
                    }
                }
            },
            "training": {
                "num_epochs": 100,
                "batch_size": 16,
                "max_epochs_wo_improvement": 10
            },
            "optimizer": {
                "init_lr": 2e-5,
                "num_train_steps": 100000,
                "num_warmup_steps": 10000
            }
        }
        """
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

        # PLACEHOLDERS
        # bert
        self.input_ids_ph = None
        self.input_mask_ph = None
        self.segment_ids_ph = None

        # ner
        self.first_pieces_coords_ph = None
        self.num_pieces_ph = None  # для обучаемых с нуля рекуррентных слоёв
        self.num_tokens_ph = None  # для crf
        self.ner_labels_ph = None

        # re
        self.re_labels_ph = None

        # common
        self.training_ph = None

        # TENSORS
        self.loss_ner = None
        self.loss_re = None
        self.ner_logits_train = None
        self.transition_params = None
        self.ner_preds_inference = None
        self.re_logits_train = None
        self.re_labels_true_entities = None
        self.re_labels_pred_entities = None
        self.num_entities = None
        self.num_entities_pred = None

        # LAYERS
        self.bert_dropout = None
        self.birnn_ner = None
        self.birnn_re = None
        self.dense_ner_labels = None
        self.ner_emb = None
        self.ner_emb_layer_norm = None
        self.ner_emb_dropout = None
        self.entity_pairs_enc = None

        # OPS
        self.train_op_head = None

    def _build_graph(self):
        self._set_placeholders()

        # N - batch size
        # D - bert dim
        # T_pieces - число bpe-сиволов (включая [CLS] и [SEP])
        # T_tokens - число токенов (не вклчая [CLS] и [SEP])
        with tf.variable_scope(self.model_scope):
            bert_out_train = self._build_bert(training=True)  # [N, T_pieces, D]
            bert_out_pred = self._build_bert(training=False)  # [N, T_pieces, D]

            self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

            # ner
            with tf.variable_scope(self.ner_scope):
                if self.config["model"]["ner"]["use_birnn"]:
                    self.birnn_ner = StackedBiRNN(**self.config["model"]["ner"]["rnn"])

                num_labels = self.config["model"]["ner"]["num_labels"]
                self.dense_ner_labels = tf.keras.layers.Dense(num_labels)

                self.ner_logits_train, _, self.transition_params = self._build_ner_head(bert_out=bert_out_train)
                _, self.ner_preds_inference, _ = self._build_ner_head(bert_out=bert_out_pred)

            # re
            with tf.variable_scope(self.re_scope):
                if self.config["model"]["re"]["use_entity_emb"]:
                    bert_dim = self.config["model"]["bert"]["dim"]
                    self.ner_emb = tf.keras.layers.Embedding(num_labels, bert_dim)
                    if self.config["model"]["re"]["use_entity_emb_layer_norm"]:
                        self.ner_emb_layer_norm = tf.keras.layers.LayerNormalization()
                    self.ner_emb_dropout = tf.keras.layers.Dropout(self.config["model"]["re"]["entity_emb_dropout"])

                if self.config["model"]["re"]["use_birnn"]:
                    self.birnn_re = StackedBiRNN(**self.config["model"]["re"]["rnn"])

                self.entity_pairs_enc = GraphEncoder(**self.config["model"]["re"]["biaffine"])

                self.re_logits_train, self.num_entities = self._build_re_head(
                    bert_out=bert_out_train, ner_labels=self.ner_labels_ph
                )
                re_logits_true_entities, _ = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=self.ner_labels_ph
                )
                re_logits_pred_entities, self.num_entities_pred = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=self.ner_preds_inference
                )

                self.re_labels_true_entities = tf.argmax(re_logits_true_entities, axis=-1)
                self.re_labels_pred_entities = tf.argmax(re_logits_pred_entities, axis=-1)

            self._set_loss()
            self._set_train_op()

    def evaluate(self, examples: List[Example], batch_size: int = 16) -> Dict:
        """
        metrics = {
            "ner": {},
            "re": {},
            "total": {}
        }
        """
        y_true_ner = []
        y_pred_ner = []

        y_true_re = []
        y_pred_re = []

        no_rel_id = self.config["model"]["re"]["no_relation_id"]

        loss = 0.0
        loss_ner = 0.0
        loss_re = 0.0
        num_batches = 0

        for start in range(0, len(examples), batch_size):
            end = start + batch_size
            examples_batch = examples[start:end]
            feed_dict = self._get_feed_dict(examples_batch, mode=ModeKeys.VALID)
            loss_i, loss_ner_i, loss_re_i, ner_labels_pred, rel_labels_pred, num_entities = self.sess.run(
                [
                    self.loss,
                    self.loss_ner,
                    self.loss_re,
                    self.ner_preds_inference,
                    self.re_labels_true_entities,
                    self.num_entities
                ],
                feed_dict=feed_dict
            )
            loss += loss_i
            loss_ner += loss_ner_i
            loss_re += loss_re_i

            for i, x in enumerate(examples_batch):
                # ner
                y_true_ner_i = []
                y_pred_ner_i = []
                for j, t in enumerate(x.tokens):
                    y_true_ner_i.append(t.labels[0])
                    y_pred_ner_i.append(self.inv_ner_enc[ner_labels_pred[i, j]])
                y_true_ner.append(y_true_ner_i)
                y_pred_ner.append(y_pred_ner_i)

                # re
                num_entities_i = num_entities[i]
                # этот assert может не выполняться в случае, когда редкие сущности игнорятся
                # assert num_entities_i == len(x.entities), f"[{x.id}] {num_entities_i} != {len(x.entities)}"
                arcs_true = np.full((num_entities_i, num_entities_i), no_rel_id, dtype=np.int32)

                for arc in x.arcs:
                    assert arc.head_index is not None
                    assert arc.dep_index is not None
                    arcs_true[arc.head_index, arc.dep_index] = arc.rel_id

                arcs_pred = rel_labels_pred[i, :num_entities_i, :num_entities_i]
                y_true_re += [self.inv_re_enc[j] for j in arcs_true.flatten()]
                y_pred_re += [self.inv_re_enc[j] for j in arcs_pred.flatten()]

            num_batches += 1

        # loss
        # TODO: учитывать, что последний батч может быть меньше. тогда среднее не совсем корректно так считать
        loss /= num_batches
        loss_ner /= num_batches
        loss_re /= num_batches

        # ner
        joiner = self.config["model"]["ner"]["prefix_joiner"]
        ner_metrics_entity_level = classification_report_ner(y_true=y_true_ner, y_pred=y_pred_ner, joiner=joiner)
        y_true_ner_flat = list(chain(*y_true_ner))
        y_pred_ner_flat = list(chain(*y_pred_ner))
        ner_metrics_token_level = classification_report(
            y_true=y_true_ner_flat, y_pred=y_pred_ner_flat, trivial_label="O"
        )

        # re
        re_metrics = classification_report(y_true=y_true_re, y_pred=y_pred_re, trivial_label="O")

        # total
        # сделано так, чтобы случайный скор на таске с нулевым loss_coef не вносил подгрешность в score.
        # невозможность равенства нулю коэффициентов при лоссах на обоих тасках рассмотрена в BaseModel.__init__
        if self.config["model"]["ner"]["loss_coef"] == 0.0:
            score = re_metrics["micro"]["f1"]
        elif self.config["model"]["re"]["loss_coef"] == 0.0:
            score = ner_metrics_entity_level["micro"]["f1"]
        else:
            score = ner_metrics_entity_level["micro"]["f1"] * 0.5 + re_metrics["micro"]["f1"] * 0.5

        performance_info = {
            "ner": {
                "loss": loss_ner,
                "metrics": {
                    "entity_level": ner_metrics_entity_level,
                    "token_level": ner_metrics_token_level
                }
            },
            "re": {
                "loss": loss_re,
                "metrics": re_metrics,
            },
            "loss": loss,
            "score": score
        }

        return performance_info

    # TODO: реалзиовать случай window > 1
    def predict(self, examples: List[Example], window: int = 1, batch_size: int = 16, **kwargs):
        """
        инференс. примеры не должны содержать разметку токенов и пар сущностей!
        сделано так для того, чтобы не было непредсказуемых результатов.

        ner - запись лейблов в Token.labels
        re - создание новых инстансов Arc и запись их в Example.arcs
        """
        assert window == 1, "logic with window > 1 is not implemented :("

        # проверка примеров
        chunks = []
        for x in examples:
            assert len(x.arcs) == 0, f"[{x.id}] arcs are already annotated"
            assert len(x.chunks) > 0, f"[{x.id}] didn't split by chunks"
            for t in x.tokens:
                assert len(t.labels) == 0, f"[{x.id}] tokens are already annotated"
            chunks += x.chunks

        id2example = {x.id: x for x in examples}

        no_rel_id = self.config["model"]["re"]["no_relation_id"]

        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            chunks_batch = chunks[start:end]
            feed_dict = self._get_feed_dict(chunks_batch, mode=ModeKeys.TEST)
            ner_labels_pred, rel_labels_pred, num_entities = self.sess.run(
                [self.ner_preds_inference, self.re_labels_pred_entities, self.num_entities_pred],
                feed_dict=feed_dict
            )

            m = max(len(x.tokens) for x in chunks_batch)
            assert m == ner_labels_pred.shape[1], f'{m} != {ner_labels_pred.shape[1]}'

            for i, chunk in enumerate(chunks_batch):
                example = id2example[chunk.parent]
                ner_labels_i = []
                # ner
                for j, t in enumerate(chunk.tokens):
                    id_label = ner_labels_pred[i, j]
                    label = self.inv_ner_enc[id_label]
                    ner_labels_i.append(label)

                entities_chunk = []
                tag2spans = get_entity_spans(labels=ner_labels_i, joiner=self.config["model"]["ner"]["prefix_joiner"])
                for label, spans in tag2spans.items():
                    for span in spans:
                        start_abs = chunk.tokens[span.start].index_abs
                        end_abs = chunk.tokens[span.end].index_abs
                        tokens = example.tokens[start_abs:end_abs + 1]
                        t_first = tokens[0]
                        t_last = tokens[-1]
                        text = example.text[t_first.span_rel.start:t_last.span_rel.end]
                        id_entity = 'T' + str(len(example.entities))
                        entity = Entity(
                            id=id_entity,
                            label=label,
                            text=text,
                            tokens=tokens,
                        )
                        example.entities.append(entity)
                        entities_chunk.append(entity)

                # re
                entities_sorted = sorted(entities_chunk, key=lambda e: (e.tokens[0].index_rel, e.tokens[-1].index_rel))
                arcs_pred = rel_labels_pred[i, :num_entities[i], :num_entities[i]]
                for j, k in zip(*np.where(arcs_pred != no_rel_id)):
                    id_label = arcs_pred[j, k]
                    id_head = entities_sorted[j].id
                    id_dep = entities_sorted[k].id
                    assert isinstance(id_head, str)  # чтоб не подсвечивалось жёлтым ниже
                    assert isinstance(id_dep, str)
                    id_arc = "R" + str(len(example.arcs))
                    arc = Arc(id=id_arc, head=id_head, dep=id_dep, rel=self.inv_re_enc[id_label])
                    example.arcs.append(arc)

    def reset_weights(self):
        super().reset_weights()

        bert_dir = self.config["model"]["bert"]["dir"]
        bert_scope = self.config["model"]["bert"]["scope"]
        var_list = {
            self._actual_name_to_checkpoint_name(x.name): x for x in tf.trainable_variables()
            if x.name.startswith(f"{self.model_scope}/{bert_scope}")
        }
        saver = tf.train.Saver(var_list)
        checkpoint_path = os.path.join(bert_dir, "bert_model.ckpt")
        saver.restore(self.sess, checkpoint_path)

    def set_train_op_head(self):
        """
        [опционально] операция для предобучения только новых слоёв
        TODO: по-хорошему нужно global_step обновлять до нуля, если хочется продолжать обучение с помощью train_op.
         иначе learning rate будет считаться не совсем ожидаемо
        """
        tvars = [
            x for x in tf.trainable_variables()
            if x.name.startswith(f"{self.model_scope}/{self.ner_scope}")
            or x.name.startswith(f"{self.model_scope}/{self.re_scope}")
        ]
        opt = tf.train.AdamOptimizer()
        grads = tf.gradients(self.loss, tvars)
        self.train_op_head = opt.apply_gradients(zip(grads, tvars))

    def _get_feed_dict(self, examples: List[Example], mode: str):
        assert self.ner_enc is not None
        assert self.re_enc is not None

        # bert
        input_ids = []
        input_mask = []
        segment_ids = []

        # ner
        first_pieces_coords = []
        num_pieces = []
        num_tokens = []
        ner_labels = []

        # re
        re_labels = []

        # filling
        for i, x in enumerate(examples):
            input_ids_i = []
            input_mask_i = []
            segment_ids_i = []
            first_pieces_coords_i = []

            # [CLS]
            input_ids_i.append(self.config["model"]["bert"]["cls_token_id"])
            input_mask_i.append(1)
            segment_ids_i.append(0)

            ner_labels_i = []
            ptr = 1

            # tokens
            for t in x.tokens:
                first_pieces_coords_i.append((i, ptr))
                num_pieces_ij = len(t.pieces)
                input_ids_i += t.token_ids
                input_mask_i += [1] * num_pieces_ij
                segment_ids_i += [0] * num_pieces_ij
                label = t.labels[0]
                if mode != ModeKeys.TEST:
                    id_label = self.ner_enc[label]
                    ner_labels_i.append(id_label)  # ner решается на уровне токенов!
                ptr += num_pieces_ij

            # [SEP]
            input_ids_i.append(self.config["model"]["bert"]["sep_token_id"])
            input_mask_i.append(1)
            segment_ids_i.append(0)

            # relations
            if mode != ModeKeys.TEST:
                for arc in x.arcs:
                    assert arc.head_index is not None
                    assert arc.dep_index is not None
                    id_rel = self.re_enc[arc.rel]
                    re_labels.append((i, arc.head_index, arc.dep_index, id_rel))

            # write
            num_pieces.append(len(input_ids_i))
            num_tokens.append(len(x.tokens))
            input_ids.append(input_ids_i)
            input_mask.append(input_mask_i)
            segment_ids.append(segment_ids_i)
            ner_labels.append(ner_labels_i)
            first_pieces_coords.append(first_pieces_coords_i)

        # padding
        pad_token_id = self.config["model"]["bert"]["pad_token_id"]
        pad_label_id = self.config["model"]["ner"]["no_entity_id"]
        num_tokens_max = max(num_tokens)
        num_pieces_max = max(num_pieces)
        for i in range(len(examples)):
            input_ids[i] += [pad_token_id] * (num_pieces_max - num_pieces[i])
            input_mask[i] += [0] * (num_pieces_max - num_pieces[i])
            segment_ids[i] += [0] * (num_pieces_max - num_pieces[i])
            ner_labels[i] += [pad_label_id] * (num_tokens_max - num_tokens[i])
            first_pieces_coords[i] += [(i, 0)] * (num_tokens_max - num_tokens[i])

        if len(re_labels) == 0:
            re_labels.append((0, 0, 0, 0))

        training = mode == ModeKeys.TRAIN

        d = {
            # bert
            self.input_ids_ph: input_ids,
            self.input_mask_ph: input_mask,
            self.segment_ids_ph: segment_ids,

            # ner
            self.first_pieces_coords_ph: first_pieces_coords,
            self.num_pieces_ph: num_pieces,
            self.num_tokens_ph: num_tokens,

            # common
            self.training_ph: training
        }

        if mode != ModeKeys.TEST:
            d[self.ner_labels_ph] = ner_labels
            d[self.re_labels_ph] = re_labels

        return d

    def _set_placeholders(self):
        # bert inputs
        self.input_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")

        # ner inputs
        # [id_example, id_piece]
        self.first_pieces_coords_ph = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name="first_pieces_coords")
        self.num_pieces_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_pieces")
        self.num_tokens_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_tokens")
        self.ner_labels_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ner_labels")

        # re inputs
        # [id_example, id_head, id_dep, id_rel]
        self.re_labels_ph = tf.placeholder(dtype=tf.int32, shape=[None, 4], name="re_labels")

        # common inputs
        self.training_ph = tf.placeholder(dtype=tf.bool, shape=None, name="training_ph")

    def _set_loss(self):
        self.loss_ner = self._get_ner_loss()
        self.loss_re = self._get_re_loss()
        self.loss = self.loss_ner + self.loss_re

    def _set_train_op(self):
        num_samples = self.config["training"]["num_train_samples"]
        batch_size = self.config["training"]["batch_size"]
        num_epochs = self.config["training"]["num_epochs"]
        num_train_steps = int(num_samples / batch_size) * num_epochs
        warmup_proportion = self.config["optimizer"]["warmup_proportion"]
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        init_lr = self.config["optimizer"]["init_lr"]
        self.train_op = create_optimizer(
            loss=self.loss,
            init_lr=init_lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=False
        )

    def _build_bert(self, training):
        bert_dir = self.config["model"]["bert"]["dir"]
        bert_scope = self.config["model"]["bert"]["scope"]
        reuse = not training
        with tf.variable_scope(bert_scope, reuse=reuse):
            bert_config = BertConfig.from_json_file(os.path.join(bert_dir, "bert_config.json"))
            bert_config.attention_probs_dropout_prob = self.config["model"]["bert"]["attention_probs_dropout_prob"]
            bert_config.hidden_dropout_prob = self.config["model"]["bert"]["hidden_dropout_prob"]
            model = BertModel(
                config=bert_config,
                is_training=training,
                input_ids=self.input_ids_ph,
                input_mask=self.input_mask_ph,
                token_type_ids=self.segment_ids_ph
            )
            x = model.get_sequence_output()
        return x

    def _build_ner_head(self,  bert_out):
        """
        bert_out -> dropout -> stacked birnn (optional) -> dense(num_labels) -> crf (optional)
        :param bert_out:
        :return:
        """
        use_crf = self.config["model"]["ner"]["use_crf"]
        num_labels = self.config["model"]["ner"]["num_labels"]

        # dropout
        if (self.birnn_ner is None) or (self.config["model"]["ner"]["rnn"]["dropout"] == 0.0):
            x = self.bert_dropout(bert_out, training=self.training_ph)
        else:
            x = bert_out

        # birnn
        if self.birnn_ner is not None:
            sequence_mask = tf.sequence_mask(self.num_pieces_ph)
            x = self.birnn_ner(x, training=self.training_ph, mask=sequence_mask)

        # pieces -> tokens
        # сделано так для того, чтобы в ElmoJointModel не нужно было переопределять данный метод
        if self.first_pieces_coords_ph is not None:
            x = tf.gather_nd(x, self.first_pieces_coords_ph)  # [N, num_tokens_tokens, bert_dim or cell_dim * 2]

        # label logits
        logits = self.dense_ner_labels(x)

        # label ids
        if use_crf:
            with tf.variable_scope("crf", reuse=tf.AUTO_REUSE):
                transition_params = tf.get_variable("transition_params", [num_labels, num_labels], dtype=tf.float32)
            pred_ids, _ = tf.contrib.crf.crf_decode(logits, transition_params, self.num_tokens_ph)
        else:
            pred_ids = tf.argmax(logits, axis=-1)
            transition_params = None

        return logits, pred_ids, transition_params

    def _build_re_head(self, bert_out: tf.Tensor, ner_labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x, num_entities = self._get_entities_representation(bert_out=bert_out, ner_labels=ner_labels)

        # encoding of pairs
        inputs = GraphEncoderInputs(head=x, dep=x)
        logits = self.entity_pairs_enc(inputs=inputs, training=self.training_ph)  # [N, num_ent, num_ent, num_relation]
        return logits, num_entities

    def _get_entities_representation(self, bert_out: tf.Tensor, ner_labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        bert_out ->
        ner_labels -> x_ner

        Выход - логиты отношений

        Пусть v_context - контекстный вектор первого токена сущности или триггера события,
              v_label - обучаемый с нуля вектор лейбла или триггера события
              v_entity - обучаемый с нуля вектор именной сущности

        Есть несколько способов векторизации сущностей и триггеров событий:

        1. v_context
        2. v_context + v_label
        3. сущнсоть - v_entity, триггер - v_context + v_label

        :param bert_out: tf.Tensor of shape [batch_size, num_pieces_max, bert_dim] and type tf.float32
        :param ner_labels: tf.Tensor of shape [batch_size, num_tokens_max] and type tf.int32
        :return:
        """
        # dropout
        bert_out = self.bert_dropout(bert_out, training=self.training_ph)

        # pieces -> tokens
        x_bert = tf.gather_nd(bert_out, self.first_pieces_coords_ph)  # [batch_size, num_tokens, bert_dim]

        if self.ner_emb is not None:
            x_emb = self._get_ner_embeddings(ner_labels=ner_labels)
            x = x_bert + x_emb
        else:
            x = x_bert

        if self.birnn_re is not None:
            sequence_mask = tf.sequence_mask(self.num_tokens_ph)
            x = self.birnn_re(x, training=self.training_ph, mask=sequence_mask)  # [N, num_tokens, cell_dim * 2]

        # вывод координат первых токенов сущностей
        start_ids = tf.constant(self.config["model"]["ner"]["start_ids"], dtype=tf.int32)
        coords, num_entities = get_batched_coords_from_labels(
            labels_2d=ner_labels, values=start_ids, sequence_len=self.num_tokens_ph
        )

        # tokens -> entities
        x = tf.gather_nd(x, coords)   # [batch_size, num_entities_max, bert_bim or cell_dim * 2]
        return x, num_entities

    def _get_ner_loss(self):
        use_crf = self.config["model"]["ner"]["use_crf"]
        if use_crf:
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                inputs=self.ner_logits_train,
                tag_indices=self.ner_labels_ph,
                sequence_lengths=self.num_tokens_ph,
                transition_params=self.transition_params
            )
            loss = -tf.reduce_mean(log_likelihood)
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.ner_labels_ph, logits=self.ner_logits_train
            )
            loss = tf.reduce_mean(loss)

        loss *= self.config["model"]["ner"]["loss_coef"]
        return loss

    def _get_re_loss(self):
        no_rel_id = self.config["model"]["re"]["no_relation_id"]
        logits_shape = tf.shape(self.re_logits_train)
        labels_shape = logits_shape[:3]
        labels = get_dense_labels_from_indices(indices=self.re_labels_ph, shape=labels_shape, no_label_id=no_rel_id)
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=self.re_logits_train
        )  # [batch_size, num_entities, num_entities]

        sequence_mask = tf.sequence_mask(self.num_entities, maxlen=logits_shape[1], dtype=tf.float32)
        mask = sequence_mask[:, None, :] * sequence_mask[:, :, None]

        masked_per_example_loss = per_example_loss * mask
        total_loss = tf.reduce_sum(masked_per_example_loss)
        num_pairs = tf.cast(tf.reduce_sum(mask), tf.float32)
        num_pairs = tf.maximum(num_pairs, 1.0)
        loss = total_loss / num_pairs
        loss *= self.config["model"]["re"]["loss_coef"]
        return loss

    def _actual_name_to_checkpoint_name(self, name: str) -> str:
        bert_scope = self.config["model"]["bert"]["scope"]
        prefix = f"{self.model_scope}/{bert_scope}/"
        name = name[len(prefix):]
        name = name.replace(":0", "")
        return name

    def _get_ner_embeddings(self, ner_labels):
        x_emb = self.ner_emb(ner_labels)
        if self.ner_emb_layer_norm is not None:
            x_emb = self.ner_emb_layer_norm(x_emb)
        x_emb = self.ner_emb_dropout(x_emb, training=self.training_ph)
        return x_emb


class BertForRelationExtraction(BertJointModel):
    """
    ner уже решён.
    сущности заменены на соответствующие лейблы: Иван Иванов работает в ООО "Ромашка". -> [PER] работает в [ORG].
    """
    def __init__(self, sess, config=None, ner_enc=None, re_enc=None, entity_label_to_token_id=None):
        # ner_enc - для сохранения интерфейса
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

        self.entity_label_to_token_id = entity_label_to_token_id

        # так как решается только relation extraction, вывод числа сущностей выводится однозначно.
        # поэтому их можно прокидывать через плейсхолдер
        self.num_entities_ph = None
        self.entity_coords_ph = None

    def _build_graph(self):
        self._set_placeholders()

        # N - batch size
        # D - bert dim
        # T_pieces - число bpe-сиволов (включая [CLS] и [SEP])
        # T_tokens - число токенов (не вклчая [CLS] и [SEP])
        with tf.variable_scope(self.model_scope):
            bert_out_train = self._build_bert(training=True)  # [N, T_pieces, D]
            bert_out_pred = self._build_bert(training=False)  # [N, T_pieces, D]

            self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

            with tf.variable_scope(self.re_scope):
                if self.config["model"]["re"]["use_birnn"]:
                    self.birnn_re = StackedBiRNN(**self.config["model"]["re"]["rnn"])

                self.entity_pairs_enc = GraphEncoder(**self.config["model"]["re"]["biaffine"])

                self.re_logits_train, _ = self._build_re_head(
                    bert_out=bert_out_train, ner_labels=self.ner_labels_ph
                )
                re_logits_true_entities, _ = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=self.ner_labels_ph
                )

                self.re_labels_true_entities = tf.argmax(re_logits_true_entities, axis=-1)

            self.num_entities = self.num_entities_ph
            self._set_loss()
            self._set_train_op()

    def evaluate(self, examples: List[Example], batch_size: int = 16) -> Dict:
        """
        metrics = {
            "ner": {},
            "re": {},
            "total": {}
        }
        """
        y_true_re = []
        y_pred_re = []

        no_rel_id = self.config["model"]["re"]["no_relation_id"]

        loss = 0.0
        num_batches = 0

        for start in range(0, len(examples), batch_size):
            end = start + batch_size
            examples_batch = examples[start:end]
            feed_dict = self._get_feed_dict(examples_batch, mode=ModeKeys.VALID)
            loss_i, rel_labels_pred = self.sess.run([self.loss, self.re_labels_true_entities], feed_dict=feed_dict)
            loss += loss_i

            for i, x in enumerate(examples_batch):
                # re TODO: рассмотреть случаи num_events == 0
                num_entities_i = len(x.entities)
                arcs_true = np.full((num_entities_i, num_entities_i), no_rel_id, dtype=np.int32)

                for arc in x.arcs:
                    assert arc.head_index is not None
                    assert arc.dep_index is not None
                    arcs_true[arc.head_index, arc.dep_index] = arc.rel_id

                arcs_pred = rel_labels_pred[i, :num_entities_i, :num_entities_i]
                assert arcs_pred.shape[0] == num_entities_i, f"{arcs_pred.shape[0]} != {num_entities_i}"
                assert arcs_pred.shape[1] == num_entities_i, f"{arcs_pred.shape[1]} != {num_entities_i}"
                y_true_re += [self.inv_re_enc[j] for j in arcs_true.flatten()]
                y_pred_re += [self.inv_re_enc[j] for j in arcs_pred.flatten()]

            num_batches += 1

        # loss
        # TODO: учитывать, что последний батч может быть меньше. тогда среднее не совсем корректно так считать
        loss /= num_batches

        # TODO: хардкод "O"
        re_metrics = classification_report(y_true=y_true_re, y_pred=y_pred_re, trivial_label="O")

        # total
        performance_info = {
            "loss": loss,
            "metrics": re_metrics,
            "score": re_metrics["micro"]["f1"]
        }

        return performance_info

    def predict(self, examples: List[Example], window: int = 1, batch_size: int = 16, **kwargs):
        """
        Оценка качества на уровне документа.
        :param examples: документы
        :param chunks: куски (stride 1). предполагаетя, что для каждого документа из examples должны быть куски в chunks
        :param window: размер кусков (в предложениях)
        :param batch_size:
        :return:
        """
        # проверка на то, то в примерах нет рёбер
        # TODO: как-то обработать случай отсутствия сущнсоетй

        id2example = {}
        id_to_num_sentences = {}
        chunks = []
        for x in examples:
            assert len(x.arcs) == 0
            assert len(x.chunks) > 0
            id2example[x.id] = x
            id_to_num_sentences[x.id] = x.tokens[-1].id_sent + 1
            chunks += x.chunks

        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            chunks_batch = chunks[start:end]
            feed_dict = self._get_feed_dict(chunks_batch, mode=ModeKeys.VALID)
            re_labels_pred = self.sess.run(self.re_labels_true_entities, feed_dict=feed_dict)

            for i in range(len(chunks_batch)):
                chunk = chunks_batch[i]
                parent = id2example[chunk.parent]

                num_sentences = id_to_num_sentences[chunk.parent]
                end_rel = chunk.tokens[-1].id_sent - chunk.tokens[0].id_sent
                assert end_rel < window, f"[{chunk.id}] relative end {end_rel} >= window size {window}"
                is_first = chunk.tokens[0].id_sent == 0
                is_last = chunk.tokens[-1].id_sent == num_sentences - 1
                pairs = get_sent_pairs_to_predict_for(end=end_rel, is_first=is_first, is_last=is_last, window=window)

                num_entities_i = len(chunk.entities)
                arcs_pred = re_labels_pred[i, :num_entities_i, :num_entities_i]
                index2entity = {entity.index: entity for entity in chunk.entities}
                assert len(index2entity) == num_entities_i

                # предсказанные лейблы, которые можно получить из предиктов для кусочка chunk
                for id_sent_rel_a, id_sent_rel_b in pairs:
                    id_sent_abs_a = id_sent_rel_a + chunk.tokens[0].id_sent
                    id_sent_abs_b = id_sent_rel_b + chunk.tokens[0].id_sent
                    for idx_head, idx_dep in zip(*np.where(arcs_pred != 0)):
                        head = index2entity[idx_head]
                        dep = index2entity[idx_dep]
                        id_sent_head = head.tokens[0].id_sent
                        id_sent_dep = dep.tokens[0].id_sent
                        if (id_sent_head == id_sent_abs_a and id_sent_dep == id_sent_abs_b) or \
                                (id_sent_head == id_sent_abs_b and id_sent_dep == id_sent_abs_a):
                            id_arc = "R" + str(len(parent.arcs))
                            id_label = arcs_pred[idx_head, idx_dep]
                            arc = Arc(
                                id=id_arc,
                                head=head.id,
                                dep=dep.id,
                                rel=self.inv_re_enc[id_label]
                            )
                            parent.arcs.append(arc)

    def save(self, model_dir: str, force: bool = True, scope_to_save: str = None):
        assert self.entity_label_to_token_id is not None
        super().save(model_dir=model_dir, force=force, scope_to_save=scope_to_save)
        with open(os.path.join(model_dir, "entity_label_to_token_id.json"), "w") as f:
            json.dump(self.entity_label_to_token_id, f, indent=4)

    @classmethod
    def load(cls, sess, model_dir: str, scope_to_load: str = None):
        model = super().load(sess=sess, model_dir=model_dir, scope_to_load=scope_to_load)

        with open(os.path.join(model_dir, "entity_label_to_token_id.json")) as f:
            model.entity_label_to_token_id = json.load(f)

        return model

    def _get_feed_dict(self, examples: List[Example], mode: str):
        # bert
        input_ids = []
        input_mask = []
        segment_ids = []

        entity_coords = []
        num_pieces = []
        num_entities = []
        re_labels = []

        # filling
        for i, x in enumerate(examples):
            input_ids_i = []
            input_mask_i = []
            segment_ids_i = []
            entity_coords_i = []

            # [CLS]
            input_ids_i.append(self.config["model"]["bert"]["cls_token_id"])
            input_mask_i.append(1)
            segment_ids_i.append(0)

            ptr = 1

            if len(x.entities) == 0:
                for t in x.tokens:
                    num_pieces_ij = len(t.pieces)
                    input_ids_i += t.token_ids
                    input_mask_i += [1] * num_pieces_ij
                    segment_ids_i += [0] * num_pieces_ij
                    ptr += num_pieces_ij
            else:
                sorted_entities = sorted(x.entities, key=lambda e: e.tokens[0].index_rel)
                idx_start = 0
                for entity in sorted_entities:
                    idx_end = entity.tokens[0].index_rel
                    for t in x.tokens[idx_start:idx_end]:
                        # кусочки токена TODO: копипаста
                        num_pieces_ij = len(t.pieces)
                        input_ids_i += t.token_ids
                        input_mask_i += [1] * num_pieces_ij
                        segment_ids_i += [0] * num_pieces_ij
                        ptr += num_pieces_ij

                    # кусочек сущности
                    entity_coords_i.append((i, ptr))
                    input_ids_i.append(self.entity_label_to_token_id[entity.label])
                    input_mask_i.append(1)
                    segment_ids_i.append(0)
                    ptr += 1

                    # обновление границы
                    idx_start = entity.tokens[-1].index_rel + 1

                for t in x.tokens[idx_start:]:
                    # кусочки токена TODO: копипаста
                    num_pieces_ij = len(t.pieces)
                    input_ids_i += t.token_ids
                    input_mask_i += [1] * num_pieces_ij
                    segment_ids_i += [0] * num_pieces_ij
                    ptr += num_pieces_ij

            # [SEP]
            input_ids_i.append(self.config["model"]["bert"]["sep_token_id"])
            input_mask_i.append(1)
            segment_ids_i.append(0)

            # relations
            for arc in x.arcs:
                assert arc.head_index is not None
                assert arc.dep_index is not None
                re_labels.append((i, arc.head_index, arc.dep_index, arc.rel_id))

            # write
            num_pieces.append(len(input_ids_i))
            num_entities.append(len(x.entities))
            input_ids.append(input_ids_i)
            input_mask.append(input_mask_i)
            segment_ids.append(segment_ids_i)
            entity_coords.append(entity_coords_i)

        # padding
        pad_token_id = self.config["model"]["bert"]["pad_token_id"]
        num_pieces_max = max(num_pieces)
        num_entities_max = max(num_entities)
        for i in range(len(examples)):
            input_ids[i] += [pad_token_id] * (num_pieces_max - num_pieces[i])
            input_mask[i] += [0] * (num_pieces_max - num_pieces[i])
            segment_ids[i] += [0] * (num_pieces_max - num_pieces[i])
            entity_coords[i] += [(i, 0)] * (num_entities_max - num_entities[i])

        if len(re_labels) == 0:
            re_labels.append((0, 0, 0, 0))

        if len(entity_coords) == 0:
            entity_coords.append([(0, 0)])

        training = mode == ModeKeys.TRAIN

        d = {
            # bert
            self.input_ids_ph: input_ids,
            self.input_mask_ph: input_mask,
            self.segment_ids_ph: segment_ids,

            self.num_pieces_ph: num_pieces,
            self.num_entities_ph: num_entities,
            self.entity_coords_ph: entity_coords,
            self.re_labels_ph: re_labels,
            self.training_ph: training
        }
        return d

    def _set_placeholders(self):
        # bert inputs
        self.input_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")

        self.first_pieces_coords_ph = tf.placeholder(
            dtype=tf.int32, shape=[None, None, 2], name="first_pieces_coords"
        )  # [id_example, id_piece]
        self.num_pieces_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_pieces")
        self.num_entities_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_entities")
        self.entity_coords_ph = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name="entity_coords")
        self.re_labels_ph = tf.placeholder(
            dtype=tf.int32, shape=[None, 4], name="re_labels"
        )  # [id_example, id_head, id_dep, id_rel]

        # common inputs
        self.training_ph = tf.placeholder(dtype=tf.bool, shape=None, name="training_ph")

    def _set_loss(self):
        self.loss_re = self._get_re_loss()
        self.loss = self.loss_re

    def _get_entities_representation(self, bert_out: tf.Tensor, ner_labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.birnn_re is not None:
            sequence_mask = tf.sequence_mask(self.num_pieces_ph)
            x = self.birnn_re(bert_out, training=self.training_ph, mask=sequence_mask)  # [N, num_tokens, cell_dim * 2]
        else:
            x = bert_out

        x = tf.gather_nd(x, self.entity_coords_ph)   # [batch_size, num_entities_max, bert_bim or cell_dim * 2]
        return x, self.num_entities_ph


class BertJointModelV2(BertJointModel):
    def __init__(self, sess, config=None, ner_enc=None, re_enc=None):
        """
        Изменена логика векторизации сущностей:
        v1: token embeddings + ner label embeddings -> rnn -> first token emdeeing
        v2: см. описание ф-ии _vectorize_whole_entities

        Изменения в config:
        - model.ner.start_ids
        + model.ner.{token_start_ids, entity_start_ids, event_start_ids}
        """
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

    def _get_entities_representation(self, bert_out: tf.Tensor, ner_labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        assert self.ner_emb is not None
        assert self.ner_emb_dropout is not None
        assert self.birnn_re is not None

        # dropout
        bert_out = self.bert_dropout(bert_out, training=self.training_ph)

        # pieces -> tokens
        x_bert = tf.gather_nd(bert_out, self.first_pieces_coords_ph)  # [batch_size, num_tokens, bert_dim]

        # tokens -> [tokens_wo_entity_tokens; entity_labels]
        x, coords, num_tokens_new, num_entities = self._vectorize_whole_entities(
            x_tokens=x_bert,
            ner_labels=ner_labels,
        )

        sequence_mask = tf.sequence_mask(num_tokens_new)
        x = self.birnn_re(x, training=self.training_ph, mask=sequence_mask)  # [N, num_tokens, cell_dim * 2]

        # tokens -> entities
        x = tf.gather_nd(x, coords)
        return x, num_entities

    def _vectorize_whole_entities(self, x_tokens, ner_labels):
        """
        Векторизация сущностей, инвариантная к их значениям.

        * Пусть r(A, B) - отношение между A и B.
          Если A и B - именные сущности, то важны их лейблы и контекст, но не важны конкретные значения.
          Например, в предложении `ООО "Ромашка" обанкротила ООО "Одуванчик"` названия организаций
          `ООО "Ромашка"` и `ООО "Одуванчик"` можно заменить на `[ORG]`, чтоб модель не переобучалась под конкретные
          названия.
        * Словоформа триггера события важна: например, если пример выше заменить на `[ORG] [BANKRUPTCY] [ORG]`,
          то не будет ясно, кто конкретно стал банкротом.
        * В то же время было бы полезно модели понять, к какому конкретно событию относится триггер, чтоб проще учить
          условное распределение на роли. Например, если модель будет знать, что слово "обанкротила" является триггером
          события "банкротство", то ей будет проще понять, что роли могут быть только {банкрот, тот_кто_банкротит}, потому
          что в разметке других ролей нет у данного события.

        emb types:
        0 - эмбеддинг токена
        1 - эмбеддинг сущности
        2 - эмбеддинг токена + эмбеддинг сущности

        tokens:     Компания	ООО     "	    Ромашка     "	    обанкротила     Газпром    .
        labels:     O		    B_ORG	I_ORG	I_ORG	    I_ORG	B_BANKRUPTCY    B_ORG      O
        mask:       True        True    False   False       False   True            True       True
        emb_type    0           1       None    None        None    2               1          0

        Args:
            x_tokens: tf.Tensor of shape [batch_size, num_tokens_max, hidden] and dtype tf.float32 - векторизованные токены
            ner_labels: tf.Tensor of shape [batch_size, num_tokens_max] and dtype tf.int32 - ner лейблы

        Returns:
            x: tf.Tensor of shape [batch_size, num_tokens_new_max, hidden] and dtype tf.float32
            coords: то же, что и в get_padded_coords
            num_entities: то же, что и в get_padded_coords
        """
        no_entity_id = self.config["model"]["ner"]["no_entity_id"]
        token_start_ids = tf.constant([no_entity_id] + self.config["model"]["ner"]["token_start_ids"], dtype=tf.int32)
        entity_start_ids = tf.constant(self.config["model"]["ner"]["entity_start_ids"], dtype=tf.int32)
        event_start_ids = tf.constant(self.config["model"]["ner"]["event_start_ids"], dtype=tf.int32)

        coords, num_tokens_new = get_batched_coords_from_labels(
            labels_2d=ner_labels, values=token_start_ids, sequence_len=self.num_tokens_ph
        )  # [batch_size, max_num_tokens_new, 2], [batch_size]
        x_tokens_new = tf.gather_nd(x_tokens, coords)  # [batch_size, max_num_tokens_new, d]
        ner_labels_new = tf.gather_nd(ner_labels, coords)  # [batch_size, max_num_tokens_new]

        x_emb = self._get_ner_embeddings(ner_labels=ner_labels_new)
        x_tokens_plus_emb = x_tokens_new + x_emb

        # маски таковы, что sum(masks) = ones_like(ner_labels_new)
        mask_tok = tf.equal(ner_labels_new, no_entity_id)  # O
        mask_entity = get_labels_mask(labels_2d=ner_labels_new, values=entity_start_ids, sequence_len=num_tokens_new)
        mask_event = get_labels_mask(labels_2d=ner_labels_new, values=event_start_ids, sequence_len=num_tokens_new)

        mask_tok = tf.cast(tf.expand_dims(mask_tok, -1), tf.float32)
        mask_entity = tf.cast(tf.expand_dims(mask_entity, -1), tf.float32)
        mask_event = tf.cast(tf.expand_dims(mask_event, -1), tf.float32)

        # merge
        x_new = x_tokens_new * mask_tok + x_emb * mask_entity + x_tokens_plus_emb * mask_event

        # coords of entities and events
        entity_and_event_start_ids = tf.concat([entity_start_ids, event_start_ids], axis=-1)
        coords_new, num_entities_new = get_batched_coords_from_labels(
            labels_2d=ner_labels_new, values=entity_and_event_start_ids, sequence_len=num_tokens_new
        )
        return x_new, coords_new, num_tokens_new, num_entities_new


class BertJointModelWithNestedNer(BertJointModel):
    """
    https://arxiv.org/abs/2005.07150 - ner
    https://arxiv.org/abs/1812.11275 - re
    """
    def __init__(self, sess, config=None, ner_enc=None, re_enc=None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

        self.ner_logits_inference = None
        self.tokens_pair_enc = None

    def _build_graph(self):
        self._set_placeholders()

        # N - batch size
        # D - bert dim
        # T_pieces - число bpe-сиволов (включая [CLS] и [SEP])
        # T_tokens - число токенов (не вклчая [CLS] и [SEP])
        with tf.variable_scope(self.model_scope):
            bert_out_train = self._build_bert(training=True)  # [N, T_pieces, D]
            bert_out_pred = self._build_bert(training=False)  # [N, T_pieces, D]

            self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["bert"]["dropout"])

            with tf.variable_scope(self.ner_scope):
                if self.config["model"]["ner"]["use_birnn"]:
                    self.birnn_ner = StackedBiRNN(**self.config["model"]["ner"]["rnn"])

                self.tokens_pair_enc = GraphEncoder(**self.config["model"]["ner"]["biaffine"])

                self.ner_logits_train = self._build_ner_head(bert_out=bert_out_train)
                self.ner_logits_inference = self._build_ner_head(bert_out=bert_out_pred)

            # re
            with tf.variable_scope(self.re_scope):
                if self.config["model"]["re"]["use_entity_emb"]:
                    num_entities = self.config["model"]["ner"]["biaffine"]["num_labels"]
                    if self.config["model"]["re"]["use_birnn"]:
                        emb_dim = self.config["model"]["re"]["rnn"]["cell_dim"] * 2
                    else:
                        emb_dim = self.config["model"]["bert"]["dim"]
                    self.ner_emb = tf.keras.layers.Embedding(num_entities, emb_dim)
                    if self.config["model"]["re"]["use_entity_emb_layer_norm"]:
                        self.ner_emb_layer_norm = tf.keras.layers.LayerNormalization()
                    self.ner_emb_dropout = tf.keras.layers.Dropout(self.config["model"]["re"]["entity_emb_dropout"])

                if self.config["model"]["re"]["use_birnn"]:
                    self.birnn_re = StackedBiRNN(**self.config["model"]["re"]["rnn"])

                self.entity_pairs_enc = GraphEncoder(**self.config["model"]["re"]["biaffine"])

                shape = tf.shape(self.ner_logits_train)[:-1]
                no_entity_id = self.config["model"]["ner"]["no_entity_id"]
                ner_labels_dense = get_dense_labels_from_indices(
                    indices=self.ner_labels_ph, shape=shape, no_label_id=no_entity_id
                )

                ner_preds_inference = tf.argmax(self.ner_logits_inference, axis=-1, output_type=tf.int32)

                self.re_logits_train, self.num_entities = self._build_re_head(
                    bert_out=bert_out_train, ner_labels=ner_labels_dense
                )

                re_logits_true_entities, _ = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=ner_labels_dense
                )
                re_logits_pred_entities, self.num_entities_pred = self._build_re_head(
                    bert_out=bert_out_pred, ner_labels=ner_preds_inference
                )

                self.re_labels_true_entities = tf.argmax(re_logits_true_entities, axis=-1)
                self.re_labels_pred_entities = tf.argmax(re_logits_pred_entities, axis=-1)

            self._set_loss()
            self._set_train_op()

    def evaluate(self, examples: List[Example], batch_size: int = 16) -> Dict:
        y_true_ner = []
        y_pred_ner = []

        y_true_re = []
        y_pred_re = []

        no_entity_id = self.config["model"]["ner"]["no_entity_id"]
        no_rel_id = self.config["model"]["re"]["no_relation_id"]

        loss = 0.0
        loss_ner = 0.0
        loss_re = 0.0
        num_batches = 0

        for start in range(0, len(examples), batch_size):
            end = start + batch_size
            examples_batch = examples[start:end]
            feed_dict = self._get_feed_dict(examples_batch, training=False)
            loss_i, loss_ner_i, loss_re_i, ner_logits, num_entities, re_labels_pred = self.sess.run([
                self.loss,
                self.loss_ner,
                self.loss_re,
                self.ner_logits_inference,
                self.num_entities,
                self.re_labels_true_entities
            ], feed_dict=feed_dict)
            loss += loss_i
            loss_ner += loss_ner_i
            loss_re += loss_re_i

            for i, x in enumerate(examples_batch):
                # ner
                num_tokens = len(x.tokens)
                spans_true = np.full((num_tokens, num_tokens), no_entity_id, dtype=np.int32)

                for entity in x.entities:
                    start = entity.tokens[0].index_rel
                    end = entity.tokens[-1].index_rel
                    spans_true[start, end] = entity.label_id

                spans_pred = np.full((num_tokens, num_tokens), no_entity_id, dtype=np.int32)
                ner_logits_i = ner_logits[i, :num_tokens, :num_tokens, :]
                spans_filtered = get_valid_spans(logits=ner_logits_i,  is_flat_ner=False)
                for span in spans_filtered:
                    spans_pred[span.start, span.end] = span.label

                y_true_ner += [self.inv_ner_enc[j] for j in spans_true.flatten()]
                y_pred_ner += [self.inv_ner_enc[j] for j in spans_pred.flatten()]

                # re
                num_entities_i = num_entities[i]
                assert num_entities_i == len(x.entities)
                arcs_true = np.full((num_entities_i, num_entities_i), no_rel_id, dtype=np.int32)

                for arc in x.arcs:
                    assert arc.head_index is not None
                    assert arc.dep_index is not None
                    arcs_true[arc.head_index, arc.dep_index] = arc.rel_id

                # TODO: проверить, что в re_labels_pred сущности отсортированы по спанам
                arcs_pred = re_labels_pred[i, :num_entities_i, :num_entities_i]
                assert arcs_pred.shape[0] == num_entities_i, f"{arcs_pred.shape[0]} != {num_entities_i}"
                assert arcs_pred.shape[1] == num_entities_i, f"{arcs_pred.shape[1]} != {num_entities_i}"
                y_true_re += [self.inv_re_enc[j] for j in arcs_true.flatten()]
                y_pred_re += [self.inv_re_enc[j] for j in arcs_pred.flatten()]

            num_batches += 1

        # loss
        # TODO: учитывать, что последний батч может быть меньше. тогда среднее не совсем корректно так считать
        loss /= num_batches
        loss_ner /= num_batches
        loss_re /= num_batches

        # ner
        trivial_label = self.inv_ner_enc[no_entity_id]
        ner_metrics = classification_report(y_true=y_true_ner, y_pred=y_pred_ner, trivial_label=trivial_label)

        # re
        re_metrics = classification_report(y_true=y_true_re, y_pred=y_pred_re, trivial_label="O")

        # total TODO: копипаста из родительского класса
        # сделано так, чтобы случайный скор на таске с нулевым loss_coef не вносил подгрешность в score.
        # невозможность равенства нулю коэффициентов при лоссах на обоих тасках рассмотрена в BaseModel.__init__
        if self.config["model"]["ner"]["loss_coef"] == 0.0:
            score = re_metrics["micro"]["f1"]
        elif self.config["model"]["re"]["loss_coef"] == 0.0:
            score = ner_metrics["micro"]["f1"]
        else:
            score = ner_metrics["micro"]["f1"] * 0.5 + re_metrics["micro"]["f1"] * 0.5

        performance_info = {
            "ner": {
                "loss": loss_ner,
                "metrics": ner_metrics
            },
            "re": {
                "loss": loss_re,
                "metrics": re_metrics,
            },
            "loss": loss,
            "score": score
        }

        return performance_info

    # TODO: перенести эту логику в evaluate
    def evaluate_doc_level(
            self,
            examples: List[Example],
            chunks: List[Example],
            window: int,
            batch_size: int = 16
    ):
        """
        Оценка качества на уровне документа.
        :param examples: документы
        :param chunks: куски (stride 1). предполагаетя, что для каждого документа из examples должны быть куски в chunks
        :param window: размер кусков (в предложениях)
        :param batch_size:
        :return:
        """
        id_to_num_sentences = {x.id: x.tokens[-1].id_sent + 1 for x in examples}
        id2rel = {v: k for k, v in self.re_enc.items()}

        # (id_example, id_head, id_dep, rel)
        y_true = set()
        for x in examples:
            for arc in x.arcs:
                y_true.add((x.id, arc.head, arc.dep, arc.rel))

        y_pred = set()

        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            chunks_batch = chunks[start:end]
            feed_dict = self._get_feed_dict(chunks_batch, training=False)
            re_labels_pred = self.sess.run(self.re_labels_true_entities, feed_dict=feed_dict)

            for i in range(len(chunks_batch)):
                chunk = chunks_batch[i]

                num_entities_chunk = len(chunk.entities)
                entities_sorted = sorted(chunk.entities, key=lambda e: (e.tokens[0].index_rel, e.tokens[-1].index_rel))
                entity_sent_ids_abs = np.array([entity.tokens[0].id_sent for entity in entities_sorted])
                entity_sent_ids_rel = entity_sent_ids_abs - chunk.tokens[0].id_sent

                num_sentences = id_to_num_sentences[chunk.parent]
                end_rel = chunk.tokens[-1].id_sent - chunk.tokens[0].id_sent
                assert end_rel < window, f"[{chunk.id}] relative end {end_rel} >= window size {window}"
                is_first = chunk.tokens[0].id_sent == 0
                is_last = chunk.tokens[-1].id_sent == num_sentences - 1
                pairs = get_sent_pairs_to_predict_for(end=end_rel, is_first=is_first, is_last=is_last, window=window)

                # предсказанные лейблы, которые можно получить из предиктов для кусочка chunk
                re_labels_pred_i = re_labels_pred[i, :num_entities_chunk, :num_entities_chunk]
                for id_sent_rel_a, id_sent_rel_b in pairs:
                    mask = get_entity_pairs_mask(entity_sent_ids_rel, id_sent_rel_a, id_sent_rel_b)
                    re_labels_pred_i_masked = re_labels_pred_i * mask
                    for idx_head, idx_dep in zip(*np.where(re_labels_pred_i_masked != 0)):
                        id_head = entities_sorted[idx_head].id
                        id_dep = entities_sorted[idx_dep].id
                        id_rel = re_labels_pred_i_masked[idx_head, idx_dep]
                        y_pred.add((chunk.parent, id_head, id_dep, id2rel[id_rel]))

        tp = len(y_true & y_pred)
        fp = len(y_pred) - tp
        fn = len(y_true) - tp
        d = f1_precision_recall_support(tp=tp, fp=fp, fn=fn)
        return d, y_true, y_pred

    def _set_placeholders(self):
        # bert inputs
        self.input_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.segment_ids_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")

        # ner inputs
        self.first_pieces_coords_ph = tf.placeholder(
            dtype=tf.int32, shape=[None, None, 2], name="first_pieces_coords"
        )  # [id_example, id_piece]
        self.num_pieces_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_pieces")
        self.num_tokens_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_tokens")
        self.ner_labels_ph = tf.placeholder(
            dtype=tf.int32, shape=[None, 4], name="ner_labels"
        )  # [id_example, start, end, label]

        # re
        self.re_labels_ph = tf.placeholder(
            dtype=tf.int32, shape=[None, 4], name="re_labels"
        )  # [id_example, head, dep, rel]

        # common inputs
        self.training_ph = tf.placeholder(dtype=tf.bool, shape=None, name="training_ph")

    def _build_ner_head(self,  bert_out):
        bert_out = self.bert_dropout(bert_out, training=self.training_ph)

        # pieces -> tokens
        x = tf.gather_nd(bert_out, self.first_pieces_coords_ph)  # [batch_size, num_tokens, bert_dim]

        if self.birnn_ner is not None:
            sequence_mask = tf.sequence_mask(self.num_tokens_ph)
            x = self.birnn_ner(x, training=self.training_ph, mask=sequence_mask)  # [N, num_tokens, cell_dim * 2]

        # encoding of pairs
        inputs = GraphEncoderInputs(head=x, dep=x)
        logits = self.tokens_pair_enc(inputs=inputs, training=self.training_ph)  # [N, num_tok, num_tok, num_entities]
        return logits

    def _get_entities_representation(self, bert_out: tf.Tensor, ner_labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """


        bert_out - [batch_size, num_pieces, bert_dim]
        ner_labels - [batch_size, num_tokens, num_tokens]

        logits - [batch_size, num_entities_max, bert_bim or cell_dim * 2]
        num_entities - [batch_size]
        """
        # dropout
        bert_out = self.bert_dropout(bert_out, training=self.training_ph)

        # pieces -> tokens
        x = tf.gather_nd(bert_out, self.first_pieces_coords_ph)  # [batch_size, num_tokens, bert_dim]

        # birnn
        if self.birnn_re is not None:
            sequence_mask = tf.sequence_mask(self.num_tokens_ph)
            x = self.birnn_re(x, training=self.training_ph, mask=sequence_mask)  # [N, num_tokens, cell_dim * 2]
            d_model = self.config["model"]["re"]["rnn"]["cell_dim"] * 2
        else:
            d_model = self.config["model"]["bert"]["dim"]

        # маскирование
        num_tokens = tf.shape(ner_labels)[1]
        mask = upper_triangular(num_tokens, dtype=tf.int32)
        ner_labels *= mask[None, :, :]

        # векторизация сущностей
        no_entity_id = self.config["model"]["ner"]["no_entity_id"]
        span_mask = tf.not_equal(ner_labels, no_entity_id)  # [batch_size, num_tokens, num_tokens]
        start_coords, end_coords, num_entities = get_padded_coords_3d(mask_3d=span_mask)
        if self.config["model"]["re"]["entity_emb_type"] == 0:
            # требуется специальный токен начала и окончания последовательности
            entity_emb_fn = get_entity_embeddings
        elif self.config["model"]["re"]["entity_emb_type"] == 1:
            entity_emb_fn = get_entity_embeddings_concat_half
        else:
            raise
        x_entity = entity_emb_fn(x=x, d_model=d_model, start_coords=start_coords, end_coords=end_coords)

        # добавление эмбеддингов лейблов сущностей
        if self.config["model"]["re"]["use_entity_emb"]:
            entity_coords = tf.concat([start_coords, end_coords[:, :, -1:]], axis=-1)
            ner_labels_2d = tf.gather_nd(ner_labels, entity_coords)
            ner_labels_2d *= tf.sequence_mask(num_entities, dtype=tf.int32)

            x_emb = self.ner_emb(ner_labels_2d)
            x_entity += x_emb

        return x_entity, num_entities

    def _get_ner_loss(self):
        """"
        1 1 1
        0 1 1
        0 0 1
        i - start, j - end
        """
        # per example loss
        no_entity_id = self.config["model"]["ner"]["no_entity_id"]
        logits_shape = tf.shape(self.ner_logits_train)
        labels_shape = logits_shape[:3]
        labels = get_dense_labels_from_indices(indices=self.ner_labels_ph, shape=labels_shape, no_label_id=no_entity_id)
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=self.ner_logits_train
        )  # [batch_size, num_tokens, num_tokens]

        # mask
        maxlen = logits_shape[1]
        span_mask = upper_triangular(maxlen, dtype=tf.float32)
        sequence_mask = tf.sequence_mask(self.num_tokens_ph, dtype=tf.float32)  # [batch_size, num_tokens]
        mask = span_mask[None, :, :] * sequence_mask[:, None, :] * sequence_mask[:, :, None]  # [batch_size, num_tokens, num_tokens]

        masked_per_example_loss = per_example_loss * mask
        total_loss = tf.reduce_sum(masked_per_example_loss)
        num_valid_spans = tf.cast(tf.reduce_sum(mask), tf.float32)
        loss = total_loss / num_valid_spans

        loss *= self.config["model"]["ner"]["loss_coef"]
        return loss

    # TODO: много копипасты!
    def _get_feed_dict(self, examples: List[Example], training: bool):
        # bert
        input_ids = []
        input_mask = []
        segment_ids = []

        # ner
        first_pieces_coords = []
        num_pieces = []
        num_tokens = []
        ner_labels = []

        # re
        re_labels = []

        # filling
        for i, x in enumerate(examples):
            input_ids_i = []
            input_mask_i = []
            segment_ids_i = []
            first_pieces_coords_i = []

            # [CLS]
            input_ids_i.append(self.config["model"]["bert"]["cls_token_id"])
            input_mask_i.append(1)
            segment_ids_i.append(0)

            ptr = 1

            # tokens
            for t in x.tokens:
                first_pieces_coords_i.append((i, ptr))
                num_pieces_ij = len(t.pieces)
                input_ids_i += t.token_ids
                input_mask_i += [1] * num_pieces_ij
                segment_ids_i += [0] * num_pieces_ij
                ptr += num_pieces_ij

            # [SEP]
            input_ids_i.append(self.config["model"]["bert"]["sep_token_id"])
            input_mask_i.append(1)
            segment_ids_i.append(0)

            # ner
            for entity in x.entities:
                start = entity.tokens[0].index_rel
                end = entity.tokens[-1].index_rel
                label = entity.label_id
                assert isinstance(label, int)
                ner_labels.append((i, start, end, label))

            # re
            for arc in x.arcs:
                assert arc.head_index is not None
                assert arc.dep_index is not None
                assert arc.rel_id is not None
                re_labels.append((i, arc.head_index, arc.dep_index, arc.rel_id))

            # write
            num_pieces.append(len(input_ids_i))
            num_tokens.append(len(x.tokens))
            input_ids.append(input_ids_i)
            input_mask.append(input_mask_i)
            segment_ids.append(segment_ids_i)
            first_pieces_coords.append(first_pieces_coords_i)

        # padding
        pad_token_id = self.config["model"]["bert"]["pad_token_id"]
        num_tokens_max = max(num_tokens)
        num_pieces_max = max(num_pieces)
        for i in range(len(examples)):
            input_ids[i] += [pad_token_id] * (num_pieces_max - num_pieces[i])
            input_mask[i] += [0] * (num_pieces_max - num_pieces[i])
            segment_ids[i] += [0] * (num_pieces_max - num_pieces[i])
            first_pieces_coords[i] += [(i, 0)] * (num_tokens_max - num_tokens[i])

        if len(ner_labels) == 0:
            ner_labels.append((0, 0, 0, 0))

        if len(re_labels) == 0:
            re_labels.append((0, 0, 0, 0))

        d = {
            self.input_ids_ph: input_ids,
            self.input_mask_ph: input_mask,
            self.segment_ids_ph: segment_ids,
            self.first_pieces_coords_ph: first_pieces_coords,
            self.num_pieces_ph: num_pieces,
            self.num_tokens_ph: num_tokens,
            self.ner_labels_ph: ner_labels,
            self.re_labels_ph: re_labels,
            self.training_ph: training
        }
        return d


# TODO: проверить работоспособность
class ElmoJointModel(BertJointModel):
    def __init__(self, sess, config=None, ner_enc=None, re_enc=None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

        self.tokens_ph = None
        self.ner_logits = None

    # TODO: мб эту часть можно было бы отрефакторить так, чтобы копипасты было меньше
    def _build_graph(self):
        self._set_placeholders()

        with tf.variable_scope(self.model_scope):
            elmo_out = self._build_elmo()  # [N, num_tokens, D]

            self.bert_dropout = tf.keras.layers.Dropout(self.config["model"]["elmo"]["dropout"])

            # ner
            with tf.variable_scope(self.ner_scope):
                if self.config["model"]["ner"]["use_birnn"]:
                    self.birnn_ner = StackedBiRNN(**self.config["model"]["ner"]["rnn"])

                num_labels = self.config["model"]["ner"]["num_labels"]
                self.dense_ner_labels = tf.keras.layers.Dense(num_labels)

                self.ner_logits_train, self.ner_preds_inference, self.transition_params = self._build_ner_head(bert_out=elmo_out)

            # re
            with tf.variable_scope(self.re_scope):
                if self.config["model"]["re"]["use_entity_emb"]:
                    elmo_dim = self.config["model"]["elmo"]["dim"]
                    self.ner_emb = tf.keras.layers.Embedding(num_labels, elmo_dim)
                    if self.config["model"]["re"]["use_entity_emb_layer_norm"]:
                        self.ner_emb_layer_norm = tf.keras.layers.LayerNormalization()
                    self.ner_emb_dropout = tf.keras.layers.Dropout(self.config["model"]["re"]["entity_emb_dropout"])

                if self.config["model"]["re"]["use_birnn"]:
                    self.birnn_re = StackedBiRNN(**self.config["model"]["re"]["rnn"])

                self.entity_pairs_enc = GraphEncoder(**self.config["model"]["re"]["biaffine"])

                self.re_logits_train, self.num_entities = self._build_re_head(
                    bert_out=elmo_out, ner_labels=self.ner_labels_ph
                )
                re_logits_pred_entities, _ = self._build_re_head(
                    bert_out=elmo_out, ner_labels=self.ner_preds_inference
                )

                self.re_labels_true_entities = tf.argmax(self.re_logits_train, axis=-1)
                self.re_labels_pred_entities = tf.argmax(re_logits_pred_entities, axis=-1)

            self._set_loss()
            self._set_train_op()

    def _get_entities_representation(self, bert_out: tf.Tensor, ner_labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        bert_out ->
        ner_labels -> x_ner

        Выход - логиты отношений

        Пусть v_context - контекстный вектор первого токена сущности или триггера события,
              v_label - обучаемый с нуля вектор лейбла или триггера события
              v_entity - обучаемый с нуля вектор именной сущности

        Есть несколько способов векторизации сущностей и триггеров событий:

        1. v_context
        2. v_context + v_label
        3. сущнсоть - v_entity, триггер - v_context + v_label

        :param bert_out: tf.Tensor of shape [batch_size, num_pieces_max, bert_dim] and type tf.float32
        :param ner_labels: tf.Tensor of shape [batch_size, num_tokens_max] and type tf.int32
        :return:
        """
        # dropout
        bert_out = self.bert_dropout(bert_out, training=self.training_ph)

        if self.ner_emb is not None:
            x_emb = self._get_ner_embeddings(ner_labels=ner_labels)
            x = bert_out + x_emb
        else:
            x = bert_out

        if self.birnn_re is not None:
            sequence_mask = tf.sequence_mask(self.num_tokens_ph)
            x = self.birnn_re(x, training=self.training_ph, mask=sequence_mask)  # [N, num_tokens, cell_dim * 2]

        # вывод координат первых токенов сущностей
        start_ids = tf.constant(self.config["model"]["ner"]["start_ids"], dtype=tf.int32)
        coords, num_entities = get_batched_coords_from_labels(
            labels_2d=ner_labels, values=start_ids, sequence_len=self.num_tokens_ph
        )

        # tokens -> entities
        x = tf.gather_nd(x, coords)   # [batch_size, num_entities_max, bert_bim or cell_dim * 2]
        return x, num_entities

    def _get_feed_dict(self, examples: List[Example], training: bool):
        # elmo
        tokens = []

        # ner
        num_tokens = []
        ner_labels = []

        # re
        re_labels = []

        # filling
        for i, x in enumerate(examples):
            tokens_i = []
            ner_labels_i = []

            # tokens
            for t in x.tokens:
                tokens_i.append(t.text)
                ner_labels_i.append(t.label_ids[0])

            # relations
            for arc in x.arcs:
                assert arc.head_index is not None
                assert arc.dep_index is not None
                re_labels.append((i, arc.head_index, arc.dep_index, arc.rel_id))

            # write
            tokens.append(tokens_i)
            num_tokens.append(len(x.tokens))
            ner_labels.append(ner_labels_i)

        # padding
        pad_label_id = self.config["model"]["ner"]["no_entity_id"]
        num_tokens_max = max(num_tokens)
        for i in range(len(examples)):
            tokens[i] += ['<pad>'] * (num_tokens_max - num_tokens[i])
            ner_labels[i] += [pad_label_id] * (num_tokens_max - num_tokens[i])

        if len(re_labels) == 0:
            re_labels.append((0, 0, 0, 0))

        d = {
            # elmo
            self.tokens_ph: tokens,

            # ner
            self.num_tokens_ph: num_tokens,
            self.ner_labels_ph: ner_labels,

            # re
            self.re_labels_ph: re_labels,

            # common
            self.training_ph: training
        }
        return d

    def _set_placeholders(self):
        # elmo inputs
        self.tokens_ph = tf.placeholder(dtype=tf.string, shape=[None, None], name="tokens")

        # ner inputs
        self.num_tokens_ph = tf.placeholder(dtype=tf.int32, shape=[None], name="num_tokens")
        self.ner_labels_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ner_labels")

        # re inputs
        # [id_example, id_head, id_dep, id_rel]
        self.re_labels_ph = tf.placeholder(dtype=tf.int32, shape=[None, 4], name="re_labels")

        # common inputs
        self.training_ph = tf.placeholder(dtype=tf.bool, shape=None, name="training_ph")

    def _build_elmo(self):
        elmo = hub.Module(self.config["model"]["elmo"]["dir"], trainable=False)
        input_dict = {
            "tokens": self.tokens_ph,
            "sequence_len": self.num_tokens_ph
        }
        x = elmo(input_dict, signature="tokens", as_dict=True)["elmo"]
        return x

    def _set_train_op(self):
        self.set_train_op_head()
        self.train_op = self.train_op_head

    def initialize(self):
        self.init_uninitialized()
