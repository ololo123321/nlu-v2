from typing import Dict, List

import numpy as np

from src.data.base import Example, Entity, Arc, NO_LABEL, NO_LABEL_ID
from src.model.base import ModeKeys
from src.model.utils import get_sent_pairs_to_predict_for
from src.model.relation_extraction import BertForRelationExtraction
from src.metrics import classification_report, classification_report_ner
from src.utils import batches_gen, log, tf, classification_report_to_string, get_entity_spans


class BertForNerAsSequenceLabelingAndRelationExtraction(BertForRelationExtraction):
    """
    требуется найти и сущности, и отношения между ними.
    https://arxiv.org/abs/1812.11275
    TODO: реализовать src.utils.get_entity_spans как tf.Operation (https://www.tensorflow.org/guide/create_op).
     тогда и в случае sequence labeling можно при векторизации учитывать вектор последнего токена сущности

    векторизация сущностей: start + label_emb, потому что по построению нет гарантии наличия лейбла L_<ENTITY>
    для соответствующего лейбла B-<ENTITY>.
    """
    ner_scope = "ner"

    def __init__(self, sess: tf.Session = None, config: Dict = None, ner_enc: Dict = None, re_enc: Dict = None):
        super().__init__(sess=sess, config=config, re_enc=re_enc)
        self.ner_enc = ner_enc

        # PLACEHOLDERS
        self.ner_labels_sequence_ph = None

        # TENSORS
        # * logits
        self.re_logits_train = None
        # * labels
        self.re_labels_valid = None
        self.re_labels_test = None
        self.ner_labels_pred = None
        # * ner loss
        self.loss_ner = None
        self.total_loss_ner = None
        self.loss_denominator_ner = None
        # re loss
        self.loss_re = None
        self.total_loss_re = None
        # LAYERS
        self.dense_ner_labels = None

        # VARIABLES
        self.transition_params = None

    def _build_graph(self):
        """
        добавление ner головы
        """
        self._build_embedder()
        with tf.variable_scope(self.ner_scope):
            self._build_ner_head()
        with tf.variable_scope(self.re_scope):
            self._build_re_head()

    # TODO: копипаста из BertForNerAsSequenceLabeling
    def _build_ner_head(self):
        self.ner_logits_train, _, self.transition_params = self._build_ner_head_fn(bert_out=self.bert_out_train)
        _, self.ner_preds_inference, _ = self._build_ner_head_fn(bert_out=self.bert_out_pred)

    def _build_re_head(self):
        """
        с добавлением головы ner валидация и тест стали различаться
        """
        self.re_logits_train, self.num_entities = self._build_re_head_fn(self.bert_out_train, self.ner_labels_ph)
        re_logits_valid, _ = self._build_re_head_fn(self.bert_out_pred, self.ner_labels_ph)
        re_logits_test, _ = self._build_re_head_fn(self.bert_out_pred, self.ner_labels_pred)

        self.re_labels_valid = tf.argmax(re_logits_valid, axis=-1)
        self.re_labels_test = tf.argmax(re_logits_test, axis=-1)

    # TODO: копипаста из BertForNerAsSequenceLabeling
    def _build_ner_head_fn(self,  bert_out):
        """
        bert_out -> dropout -> stacked birnn (optional) -> dense(num_labels) -> crf (optional)
        :param bert_out: [batch_size, num_pieces, D]
        :return:
        """
        use_crf = self.config["model"]["ner"]["use_crf"]
        num_labels = self.config["model"]["ner"]["num_labels"]

        x = self._get_token_level_embeddings(bert_out=bert_out)  # [batch_size, num_tokens, bert_dim or cell_dim * 2]

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

    def _set_placeholders(self):
        super()._set_placeholders()
        self.ner_labels_sequence_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ner_labels")

    def _set_layers(self):
        """
        dense слой для ner
        """
        super()._set_layers()
        self.dense_ner_labels = tf.keras.layers.Dense(self.config["model"]["ner"]["num_labels"])

    def _set_loss(self, *args, **kwargs):
        """
        довабление ner лосса
        """
        # ner
        self.loss_ner, self.total_loss_ner, self.loss_denominator_ner = self._get_ner_loss()

        # re
        self.loss_re, _, _, _ = super()._get_re_loss()

        # joint
        w_ner = 0.5  # TODO: в конфиг
        w_re = 0.5  # TODO: в конфиг
        self.loss = self.loss_ner * w_ner + self.loss_re * w_re

    # TODO: копипаста из BertForNerAsSequenceLabeling
    def _get_ner_loss(self):
        use_crf = self.config["model"]["ner"]["use_crf"]
        if use_crf:
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                inputs=self.ner_logits_train,
                tag_indices=self.ner_labels_ph,
                sequence_lengths=self.num_tokens_ph,
                transition_params=self.transition_params
            )
            per_example_loss = -log_likelihood
            total_loss = tf.reduce_sum(per_example_loss)
            num_sequences = tf.shape(self.ner_logits_train)[0]
            loss = total_loss / tf.cast(num_sequences, tf.float32)
            total_loss = total_loss
            loss_denominator = num_sequences
        else:
            per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.ner_labels_ph, logits=self.ner_logits_train
            )
            mask = tf.sequence_mask(self.num_tokens_ph, dtype=tf.float32)
            masked_per_example_loss = per_example_loss * mask
            total_loss = tf.reduce_sum(masked_per_example_loss)
            total_num_tokens = tf.reduce_sum(self.num_tokens_ph)
            loss = total_loss / tf.cast(total_num_tokens, tf.float32)
            total_loss = total_loss
            loss_denominator = total_num_tokens
        return loss, total_loss, loss_denominator

    # TODO: копипаста из BertForNerAsSequenceLabeling
    def _get_feed_dict(self, examples: List[Example], mode: str) -> Dict:
        """
        ner лейблы в формате, необходимом для обучения ner головы
        """
        d = super()._get_feed_dict(examples=examples, mode=mode)

        if mode != ModeKeys.TEST:
            ner_labels = []
            for x in examples:
                ner_labels_i = []
                for t in x.tokens:
                    id_label = self.ner_enc[t.label]
                    ner_labels_i.append(id_label)  # ner решается на уровне токенов!
                ner_labels.append(ner_labels_i)

            num_tokens_max = max(d[self.num_tokens_ph].num_tokens)
            pad_label_id = self.config["model"]["ner"]["no_entity_id"]
            for i in range(len(examples)):
                ner_labels[i] += [pad_label_id] * (num_tokens_max - d[self.num_tokens_ph].num_tokens[i])

            d[self.ner_labels_sequence_ph] = ner_labels
        return d

    @log
    def evaluate(self, examples: List[Example], batch_size: int = 16) -> Dict:
        # проверка примеров
        chunks = []
        id_to_num_sentences = {}
        id2example = {}
        for x in examples:
            # assert len(x.chunks) > 0, f"[{x.id}] didn't split by chunks"
            for chunk in x.chunks:
                assert chunk.parent is not None, f"[{x.id}] parent for chunk {chunk.id} is not set. " \
                    f"It is not a problem, but must be set for clarity"
                chunks.append(chunk)
            id_to_num_sentences[x.id] = x.tokens[-1].id_sent + 1
            id2example[x.id] = x

        assert len(id2example) == len(examples), f"examples must have unique ids, " \
            f"but got {len(id2example)} unique ids among {len(examples)} examples"

        # ner labels
        y_true_ner = []
        y_true_flat = []
        y_pred_ner = []
        y_pred_flat = []

        # re labels
        y_true = []
        y_pred = []

        # total loss
        loss = 0.0

        # ner loss
        loss_ner = 0.0
        loss_denominator_ner = 0

        # re loss
        loss_re = 0.0
        loss_denominator_re = 0

        gen = batches_gen(
            examples=chunks,
            max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"],
            pieces_level=True
        )
        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.VALID)
            loss_re_i, loss_ner_i, d_ner, ner_labels_pred, re_labels_pred = self.sess.run([
                self.total_loss_re,
                self.total_loss_ner,
                self.loss_denominator_ner,
                self.ner_labels_pred,
                self.re_labels_test
            ], feed_dict=feed_dict)

            loss_re += loss_re_i
            loss_ner += loss_ner_i
            loss_denominator_ner += d_ner

            for i, x in enumerate(batch):
                # ner
                y_true_i = []
                y_pred_i = []
                for j, t in enumerate(x.tokens):
                    y_true_i.append(t.label)
                    y_pred_i.append(self.inv_ner_enc[ner_labels_pred[i, j]])
                y_true_ner.append(y_true_i)
                y_true_flat += y_true_i
                y_pred_ner.append(y_pred_i)
                y_pred_flat += y_pred_i

                # re
                num_entities_i = len(x.entities)
                num_entities_i_squared = num_entities_i ** 2
                loss_denominator_re += num_entities_i_squared
                y_true_i = [NO_LABEL] * num_entities_i_squared

                for arc in x.arcs:
                    assert arc.head_index is not None
                    assert arc.dep_index is not None
                    y_true_i[num_entities_i * arc.head_index + arc.dep_index] = arc.rel
                y_true += y_true_i

                labels_pred_i = re_labels_pred[i, :num_entities_i, :num_entities_i]
                assert labels_pred_i.shape[0] == num_entities_i, f"{labels_pred_i.shape[0]} != {num_entities_i}"
                assert labels_pred_i.shape[1] == num_entities_i, f"{labels_pred_i.shape[1]} != {num_entities_i}"

                y_pred_i = [NO_LABEL] * num_entities_i_squared
                for head_index, dep_index in zip(*np.where(labels_pred_i != NO_LABEL_ID)):
                    id_label = labels_pred_i[head_index, dep_index]
                    y_pred_i[num_entities_i * head_index + dep_index] = self.inv_re_enc[id_label]
                y_pred += y_pred_i

        # ner
        loss_ner /= loss_denominator_ner
        ner_metrics = {
            "entity_level": classification_report_ner(y_true=y_true, y_pred=y_pred),
            "token_level": classification_report(y_true=y_true_flat, y_pred=y_pred_flat)
        }

        # re
        loss_re /= loss_denominator_re
        re_metrics = classification_report(y_true=y_true, y_pred=y_pred)

        # TODO: посчитать итоговый loss
        # TODO: записать куда-то loss_ner, loss_re
        # TODO: если какой-то таск выключен, то в score сделать его вес нулевым
        # total
        performance_info = {
            "loss": loss,
            "score": ner_metrics["entity_level"]["micro"]["f1"] * 0.5 + re_metrics["micro"]["f1"] * 0.5,
            "metrics": {
                "ner": ner_metrics,
                "re": re_metrics
            }
        }
        return performance_info

    @log
    def predict(self, examples: List[Example], **kwargs) -> None:
        # TODO: как-то обработать случай отсутствия сущнсоетй

        # проверка примеров
        chunks = []
        id_to_num_sentences = {}
        id2example = {}
        for x in examples:
            assert len(x.arcs) == 0
            # assert len(x.chunks) > 0, f"[{x.id}] didn't split by chunks"
            for chunk in x.chunks:
                assert chunk.parent is not None, f"[{x.id}] parent for chunk {chunk.id} is not set. " \
                    f"It is not a problem, but must be set for clarity"
                chunks.append(chunk)
            id_to_num_sentences[x.id] = x.tokens[-1].id_sent + 1
            id2example[x.id] = x

        assert len(id2example) == len(examples), f"examples must have unique ids, " \
            f"but got {len(id2example)} unique ids among {len(examples)} examples"

        window = self.config["inference"]["window"]

        gen = batches_gen(
            examples=chunks,
            max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"],
            pieces_level=True
        )
        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.TEST)
            ner_labels_pred, re_labels_pred = self.sess.run([
                self.ner_labels_pred,
                self.re_labels_test
            ], feed_dict=feed_dict)  # [N, T], [N, E, E]

            for i in range(len(batch)):
                chunk = batch[i]
                parent = id2example[chunk.parent]

                # ner
                ner_labels_i = []
                for j, t in enumerate(chunk.tokens):
                    id_label = ner_labels_pred[i, j]
                    label = self.inv_ner_enc[id_label]
                    ner_labels_i.append(label)

                tag2spans = get_entity_spans(labels=ner_labels_i)
                for label, spans in tag2spans.items():
                    for span in spans:
                        start_abs = chunk.tokens[span.start].index_abs
                        end_abs = chunk.tokens[span.end].index_abs
                        tokens = parent.tokens[start_abs:end_abs + 1]
                        t_first = tokens[0]
                        t_last = tokens[-1]
                        text = parent.text[t_first.span_rel.start:t_last.span_rel.end]
                        id_entity = 'T' + str(len(parent.entities))
                        entity = Entity(
                            id=id_entity,
                            label=label,
                            text=text,
                            tokens=tokens,
                        )
                        parent.entities.append(entity)
                        chunk.entities.append(entity)

                index2entity = sorted(
                    chunk.entities, key=lambda e: (e.tokens[0].span_rel.start, e.tokens[-1].span_rel.end)
                )

                # re
                num_sentences = id_to_num_sentences[chunk.parent]
                end_rel = chunk.tokens[-1].id_sent - chunk.tokens[0].id_sent
                assert end_rel < window, f"[{chunk.id}] relative end {end_rel} >= window size {window}"
                is_first = chunk.tokens[0].id_sent == 0
                is_last = chunk.tokens[-1].id_sent == num_sentences - 1
                pairs = get_sent_pairs_to_predict_for(end=end_rel, is_first=is_first, is_last=is_last, window=window)

                num_entities_i = len(chunk.entities)
                arcs_pred = re_labels_pred[i, :num_entities_i, :num_entities_i]
                assert len(index2entity) == num_entities_i, f'[{chunk.id}] {len(index2entity)} != {num_entities_i}'

                # предсказанные лейблы, которые можно получить из предиктов для кусочка chunk
                for id_sent_rel_a, id_sent_rel_b in pairs:
                    id_sent_abs_a = id_sent_rel_a + chunk.tokens[0].id_sent
                    id_sent_abs_b = id_sent_rel_b + chunk.tokens[0].id_sent
                    for idx_head, idx_dep in zip(*np.where(arcs_pred != NO_LABEL_ID)):
                        head = index2entity[idx_head]
                        dep = index2entity[idx_dep]
                        id_sent_head = head.tokens[0].id_sent
                        id_sent_dep = dep.tokens[0].id_sent
                        if (id_sent_head == id_sent_abs_a and id_sent_dep == id_sent_abs_b) or \
                                (id_sent_head == id_sent_abs_b and id_sent_dep == id_sent_abs_a):
                            id_arc = "R" + str(len(parent.arcs))
                            id_label = arcs_pred[idx_head, idx_dep]
                            rel = self.inv_re_enc[id_label]
                            arc = Arc(
                                id=id_arc,
                                head=head.id,
                                dep=dep.id,
                                rel=rel
                            )
                            parent.arcs.append(arc)

    def verbose_fn(self, metrics: Dict) -> None:
        self.logger.info(f'loss: {metrics["loss"]}')
        self.logger.info(f'score: {metrics["score"]}')
        self.logger.info("ner (entity-level):")
        self.logger.info('\n' + classification_report_to_string(metrics["ner"]["metrics"]["entity_level"]))
        self.logger.info("re:")
        self.logger.info('\n' + classification_report_to_string(metrics["re"]["metrics"]))


class BertForNerAsDependencyParsingAndRelationExtraction:
    """
    требуется найти и сущности, и отношения между ними.

    векторизация сущностей: [start, end, attn, label_emb]
    """
    pass
