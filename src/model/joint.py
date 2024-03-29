from typing import Dict, List

import numpy as np

from src.data.base import Example, Entity, Arc, NO_LABEL, NO_LABEL_ID
from src.model.base import ModeKeys
from src.model.utils import get_sent_pairs_to_predict_for, get_batched_coords_from_labels
from src.model.relation_extraction import BertForRelationExtraction
from src.model.layers import GraphEncoderInputs
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

        # TENSORS
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
        _, self.ner_labels_pred, _ = self._build_ner_head_fn(bert_out=self.bert_out_pred)

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
            pred_ids = tf.argmax(logits, axis=-1, output_type=tf.int32)  # default is int64
            transition_params = None
        return logits, pred_ids, transition_params

    def _build_re_head_fn(self,  bert_out, ner_labels):
        """
        ner_labels - [N, T] of type tf.int32 (в родительской версии были [num_entities_total, 4])
        """
        x = self._get_token_level_embeddings(bert_out=bert_out)  # [batch_size, num_tokens, d_bert]

        # вывод координат первых токенов сущностей
        # list на случай, если config - DictConfig, а не  dict
        start_ids = tf.constant(list(self.config["model"]["ner"]["start_ids"]), dtype=tf.int32)
        coords, num_entities = get_batched_coords_from_labels(
            labels_2d=ner_labels, values=start_ids, sequence_len=self.num_tokens_ph
        )

        # tokens -> entities
        x = tf.gather_nd(x, coords)  # [batch_size, num_entities_max, bert_bim or cell_dim * 2]

        # entity pairs scores
        inputs = GraphEncoderInputs(head=x, dep=x)
        logits = self.entity_pairs_enc(inputs, training=self.training_ph)  # [batch_size, num_ent, num_ent, num_rel]
        return logits, num_entities

    def _set_ner_labels_ph(self):
        self.ner_labels_ph = tf.placeholder(tf.int32, shape=[None, None], name="ner_labels")

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
        self.loss_re, self.total_loss_re, _, _ = super()._get_re_loss()

        # joint
        self.loss = self.loss_ner * self.config["train"]["loss_coef_ner"] + self.loss_re * self.config["train"]["loss_coef_re"]

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

    def _get_feed_dict(self, examples: List[Example], mode: str) -> Dict:
        assert self.ner_enc is not None
        assert self.re_enc is not None

        bert_inputs = self._get_bert_input_for_feed_dict(examples)

        d = {
            self.input_ids_ph: bert_inputs.input_ids,
            self.input_mask_ph: bert_inputs.input_mask,
            self.segment_ids_ph: bert_inputs.segment_ids,
            self.first_pieces_coords_ph: bert_inputs.first_pieces_coords,
            self.num_pieces_ph: bert_inputs.num_pieces,
            self.num_tokens_ph: bert_inputs.num_tokens,
            self.training_ph: mode == ModeKeys.TRAIN
        }
        if mode == ModeKeys.TEST:
            return d

        # add labels in case of train or valid mode
        ner_labels = []
        re_labels = []
        assigned_relations = set()
        for i, x in enumerate(examples):
            # entities
            ner_labels_i = []
            for t in x.tokens:
                id_label = self.ner_enc.get(t.label, NO_LABEL_ID)
                if t.label not in self.ner_enc:
                    self.logger.warning(f'[{x.id}] No label {t.label} in ner_enc. Return NO_LABEL_ID={NO_LABEL_ID}')
                ner_labels_i.append(id_label)
            ner_labels.append(ner_labels_i)

            # relations
            for arc in x.arcs:
                if arc.rel not in self.re_enc.keys():
                    self.logger.warning(f'[{x.id}] relation {arc.id} has unknown label: {arc.rel}')
                    continue
                assert arc.head_index is not None
                assert arc.dep_index is not None
                idx = i, arc.head_index, arc.dep_index
                assert idx not in assigned_relations, \
                    f'duplicated relation: {idx} (batch, head, dep). ' \
                    f'This will cause nan loss due to incorrect dense labels construction in with tf.scatter_nd'
                id_rel = self.re_enc.get(arc.rel, NO_LABEL_ID)
                if arc.rel not in self.re_enc:
                    self.logger.warning(f'[{x.id}] No label {arc.rel} in re_enc. Return NO_LABEL_ID={NO_LABEL_ID}')
                re_labels.append((*idx, id_rel))
                assigned_relations.add(idx)

        # padding
        num_tokens_max = max(bert_inputs.num_tokens)
        pad_label_id = self.config["model"]["ner"]["no_entity_id"]
        for i in range(len(examples)):
            ner_labels[i] += [pad_label_id] * (num_tokens_max - bert_inputs.num_tokens[i])

        if len(re_labels) == 0:
            re_labels.append((0, 0, 0, 0))

        d[self.ner_labels_ph] = ner_labels
        d[self.re_labels_ph] = re_labels

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

        # хотя и нужно в цикле по x.arcs проверять вхождение в этот список,
        # это не должно быть затратно по времени, потому что для большинства примеров данный список будет пустым,
        # а с каждым новым примером этот список очищается.
        invalid_entity_indices = []

        def _maybe_shift_index(index):
            """
            Нужна для корректного построения вектора y_true для задачи relation extraction
            при условии наличия новых сущностей в валидационной выборке, что значит то, что
            граф связей на инференсе строился не над всеми сущностями.

            * если есть новые сущности, то могут поехать arc.{head, dep}_index
            * чтоб восстановить верное соответствие, нужно их сдвинуть.
            * пример: entity_indices = [0, 1, 2, 3], invalid_entity_indices = [0, 2]
              тогда entity_indices нужно изменить следующим образом:
              0 -> delete; 1 -> shift left by 1; 2 - delete, 3 - shift left by 2.
            """
            if len(invalid_entity_indices) == 0:
                return index
            res = index
            for idx in invalid_entity_indices:
                assert idx != index
                if idx < index:
                    res -= 1
            return res

        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.VALID)
            loss_re_i, loss_ner_i, d_ner, ner_labels_pred, re_labels_pred = self.sess.run([
                self.total_loss_re,
                self.total_loss_ner,
                self.loss_denominator_ner,
                self.ner_labels_pred,
                self.re_labels_valid
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
                num_entities_i = 0
                for e in x.entities:
                    if self._is_valid_entity(e):
                        num_entities_i += 1
                    else:
                        invalid_entity_indices.append(e.index)

                num_entities_i_squared = num_entities_i ** 2
                loss_denominator_re += num_entities_i_squared
                y_true_i = [NO_LABEL] * num_entities_i_squared
                for arc in x.arcs:
                    assert arc.head_index is not None
                    assert arc.dep_index is not None
                    if (arc.head_index in invalid_entity_indices) or (arc.dep_index in invalid_entity_indices):
                        continue
                    head_index = _maybe_shift_index(arc.head_index)
                    dep_index = _maybe_shift_index(arc.dep_index)
                    y_true_i[num_entities_i * head_index + dep_index] = arc.rel
                y_true += y_true_i
                invalid_entity_indices.clear()

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
            "entity_level": classification_report_ner(y_true=y_true_ner, y_pred=y_pred_ner),
            "token_level": classification_report(y_true=y_true_flat, y_pred=y_pred_flat)
        }

        # re
        loss_re /= loss_denominator_re
        re_metrics = classification_report(y_true=y_true, y_pred=y_pred)

        # total
        loss = loss_ner * self.config["train"]["loss_coef_ner"] + loss_re * self.config["train"]["loss_coef_re"]
        if self.config["train"]["loss_coef_ner"] == 0.0:
            w_ner = 0
        else:
            w_ner = 0.5
        if self.config["train"]["loss_coef_re"] == 0.0:
            w_re = 0
        else:
            w_re = 0.5
        score = ner_metrics["entity_level"]["micro"]["f1"] * w_ner + re_metrics["micro"]["f1"] * w_re
        performance_info = {
            "loss": loss,
            "loss_ner": loss_ner,
            "loss_re": loss_re,
            "score": score,
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
        self.logger.info('\n' + classification_report_to_string(metrics["metrics"]["ner"]["entity_level"]))
        self.logger.info("re:")
        self.logger.info('\n' + classification_report_to_string(metrics["metrics"]["re"]))

    def _is_valid_entity(self, entity: Entity) -> bool:
        """
        * отношения строятся над парами сущностей
        * число таких пар - n^2, где n - число таких лейблов l в последовательности лейблов токенов,
          что l in self.config["model"]["ner]["start_ids"]
        * если какие-то l отсутстовали в обучении, то их нет в ner_enc. в таком случае на его место ставится лейбл NO_LABEL
        * таким образом, если в валидационном примере есть новая сущность,
          то не будет гарантироваться биекция между следующими списками:
          - [ner_enc[e.tokens[0].label] for e in x.entities]
          - [i for i in label_ids if i in start_ids]
        * решение: на валидации отильтровать сущности, которых нет в обучении
        """
        return entity.tokens[0].label in self.ner_enc.keys()


class BertForNerAsDependencyParsingAndRelationExtraction:
    """
    требуется найти и сущности, и отношения между ними.

    векторизация сущностей: [start, end, attn, label_emb]
    """
    pass
