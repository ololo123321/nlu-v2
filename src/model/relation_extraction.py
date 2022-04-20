from typing import Dict, List

import numpy as np

from src.data.base import Example, Arc, NO_LABEL, NO_LABEL_ID
from src.model.base import BaseModelRelationExtraction, BaseModelBert, ModeKeys, BaseModelNerAndRelationExtracion
from src.model.layers import GraphEncoder, GraphEncoderInputs
from src.model.utils import get_entities_representation, get_sent_pairs_to_predict_for
from src.metrics import classification_report
from src.utils import batches_gen, log, tf, classification_report_to_string


class BertForRelationExtraction(BaseModelRelationExtraction, BaseModelBert):
    """
    сущности известны в виде [start, end, label]
    так как сущности уже известны, они подаются в модель в виде четвёрок (i, start, end, label),
    где i - номер примера.

    векторизация сущностей: [start, end, label_emb]
    """
    def __init__(self, sess: tf.Session = None, config: Dict = None, ner_enc: Dict = None, re_enc: Dict = None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

        # PLACEHOLDERS
        self.ner_labels_ph = None
        self.re_labels_ph = None

        # TENSORS
        self.logits_train = None
        self.num_entities = None
        self.total_loss = None
        self.labels_pred = None

        # LAYERS
        self.entity_emb = None
        self.entity_emb_layer_norm = None
        self.entity_emb_dropout = None
        self.entity_pairs_enc = None

        # DEBUG
        self.per_example_loss = None
        self.loss_mask = None

    def _build_re_head(self):
        self.logits_train, self.num_entities = self._build_re_head_fn(bert_out=self.bert_out_train)
        logits_pred, _ = self._build_re_head_fn(bert_out=self.bert_out_pred)
        self.labels_pred = tf.argmax(logits_pred, axis=-1)

    def _set_placeholders(self):
        super()._set_placeholders()
        self.ner_labels_ph = tf.placeholder(tf.int32, shape=[None, 4], name="ner_labels")  # [i, start, end, label]
        self.re_labels_ph = tf.placeholder(tf.int32, shape=[None, 4], name="re_labels")  # [i, id_head, id_dep, label]

    def _set_layers(self):
        """
        {
            "entity_emb": {
                "use": True,
                "params": {
                    "dim": 16,  # в lee et. al. использовали размерность 20: https://github.com/kentonl/e2e-coref/blob/9d1ee1972f6e34eb5d1dcbb1fd9b9efdf53fc298/experiments.conf#L42
                    "num_labels": 10,
                    "merge_mode": "concat"  # {"add", "concat"},
                    "dropout": 0.3
                }
            }
        }
        :return:
        """
        super()._set_layers()
        self.entity_pairs_enc = GraphEncoder(**self.config["model"]["re"]["biaffine"])

        if self.config["model"]["re"]["entity_emb"]["use"]:
            params = self.config["model"]["re"]["entity_emb"]["params"]
            assert params["merge_mode"] == "concat"
            self.entity_emb = tf.keras.layers.Embedding(params["num_labels"], params["dim"])
            self.entity_emb_dropout = tf.keras.layers.Dropout(params["dropout"])

    def _build_re_head_fn(self,  bert_out):
        x = self._get_token_level_embeddings(bert_out=bert_out)  # [batch_size, num_tokens, d_bert]

        # entity embeddings
        entity_emb_layer = self._entity_emb_fn if self.config["model"]["re"]["entity_emb"]["use"] else None
        x, num_entities = get_entities_representation(
            x=x, ner_labels=self.ner_labels_ph, sparse_labels=True, ff_attn=None, entity_emb_layer=entity_emb_layer
        )  # [batch_size, num_ent, D * 3]

        inputs = GraphEncoderInputs(head=x, dep=x)
        logits = self.entity_pairs_enc(inputs, training=self.training_ph)  # [batch_size, num_ent, num_ent, num_rel]
        return logits, num_entities

    def _entity_emb_fn(self, x):
        assert self.entity_emb is not None
        x = self.entity_emb(x)
        x = self.entity_emb_dropout(x, training=self.training_ph)
        return x

    def _set_loss(self, *args, **kwargs):
        assert self.config["model"]["re"]["no_relation_id"] == 0
        logits_shape = tf.shape(self.logits_train)
        labels = tf.scatter_nd(
            indices=self.re_labels_ph[:, :-1], updates=self.re_labels_ph[:, -1], shape=logits_shape[:-1]
        )
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=self.logits_train
        )  # [batch_size, num_entities, num_entities]

        sequence_mask = tf.sequence_mask(self.num_entities, maxlen=logits_shape[1], dtype=tf.float32)
        mask = sequence_mask[:, None, :] * sequence_mask[:, :, None]

        masked_per_example_loss = per_example_loss * mask
        total_loss = tf.reduce_sum(masked_per_example_loss)
        num_pairs = tf.cast(tf.reduce_sum(self.num_entities ** 2), tf.float32)
        num_pairs = tf.maximum(num_pairs, 1.0)
        self.loss = total_loss / num_pairs
        self.total_loss = total_loss

        # DEBUG
        self.per_example_loss = per_example_loss
        self.loss_mask = mask

    def _get_feed_dict(self, examples: List[Example], mode: str) -> Dict:
        assert self.ner_enc is not None
        assert self.re_enc is not None

        bert_inputs = self._get_bert_input_for_feed_dict(examples)

        ner_labels = []
        re_labels = []
        assigned_relations = set()

        # filling
        for i, x in enumerate(examples):
            # entities
            for entity in x.entities:
                if entity.label not in self.ner_enc.keys():
                    self.logger.warning(f'[{x.id}] entity {entity.id} has unknown label: {entity.label}')
                    continue
                start = entity.tokens[0].index_rel
                end = entity.tokens[-1].index_rel
                label = self.ner_enc[entity.label]
                ner_labels.append((i, start, end, label))

            # relations
            if mode != ModeKeys.TEST:
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
                    id_rel = self.re_enc[arc.rel]
                    re_labels.append((*idx, id_rel))
                    assigned_relations.add(idx)

        if len(ner_labels) == 0:
            ner_labels.append((0, 0, 0, 0))

        d = {
            self.input_ids_ph: bert_inputs.input_ids,
            self.input_mask_ph: bert_inputs.input_mask,
            self.segment_ids_ph: bert_inputs.segment_ids,
            self.first_pieces_coords_ph: bert_inputs.first_pieces_coords,
            self.num_pieces_ph: bert_inputs.num_pieces,
            self.num_tokens_ph: bert_inputs.num_tokens,
            self.training_ph: mode == ModeKeys.TRAIN,
            self.ner_labels_ph: ner_labels
        }

        if mode != ModeKeys.TEST:
            if len(re_labels) == 0:
                re_labels.append((0, 0, 0, 0))

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

        y_true = []
        y_pred = []

        loss = 0.0
        loss_denominator = 0

        chunks = []
        for x in examples:
            for chunk in x.chunks:
                chunks.append(chunk)

        gen = batches_gen(
            examples=chunks,
            max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"],
            pieces_level=True
        )
        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.VALID)
            loss_i, labels_pred = self.sess.run([self.total_loss, self.labels_pred], feed_dict=feed_dict)
            loss += loss_i

            for i, x in enumerate(batch):
                num_entities_i = len(x.entities)
                num_entities_i_squared = num_entities_i ** 2
                loss_denominator += num_entities_i_squared
                y_true_i = [NO_LABEL] * num_entities_i_squared

                for arc in x.arcs:
                    assert arc.head_index is not None
                    assert arc.dep_index is not None
                    y_true_i[num_entities_i * arc.head_index + arc.dep_index] = arc.rel
                y_true += y_true_i

                labels_pred_i = labels_pred[i, :num_entities_i, :num_entities_i]
                assert labels_pred_i.shape[0] == num_entities_i, f"{labels_pred_i.shape[0]} != {num_entities_i}"
                assert labels_pred_i.shape[1] == num_entities_i, f"{labels_pred_i.shape[1]} != {num_entities_i}"

                y_pred_i = [NO_LABEL] * num_entities_i_squared
                for head_index, dep_index in zip(*np.where(labels_pred_i != NO_LABEL_ID)):
                    id_label = labels_pred_i[head_index, dep_index]
                    y_pred_i[num_entities_i * head_index + dep_index] = self.inv_re_enc[id_label]
                y_pred += y_pred_i

        loss /= loss_denominator
        re_metrics = classification_report(y_true=y_true, y_pred=y_pred)

        # total
        performance_info = {
            "loss": loss,
            "metrics": re_metrics,
            "score": re_metrics["micro"]["f1"]
        }
        return performance_info

    def verbose_fn(self, metrics: Dict) -> None:
        self.logger.info(f'loss: {metrics["loss"]}')
        self.logger.info(f'score: {metrics["score"]}')
        self.logger.info("metrics:")
        self.logger.info('\n' + classification_report_to_string(metrics["metrics"]))

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

        chunks = []
        for x in examples:
            for chunk in x.chunks:
                chunks.append(chunk)

        gen = batches_gen(
            examples=chunks,
            max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"],
            pieces_level=True
        )
        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.TEST)
            re_labels_pred = self.sess.run(self.labels_pred, feed_dict=feed_dict)  # [N, E, E]

            for i in range(len(batch)):
                chunk = batch[i]
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
                            arc = Arc(id=id_arc, head=head.id, dep=dep.id, rel=rel)
                            parent.arcs.append(arc)


class BertForRelationExtractionDroppedEntities:
    """
    https://arxiv.org/abs/1907.10529, секция Relation Extraction

    векторизация сущностей: entity_label_emb
    """
    pass


class BertForNerAsSequenceLabelingAndRelationExtraction(BaseModelNerAndRelationExtracion, BaseModelBert):
    """
    требуется найти и сущности, и отношения между ними.
    https://arxiv.org/abs/1812.11275
    TODO: реализовать src.utils.get_entity_spans как tf.Operation (https://www.tensorflow.org/guide/create_op).
     тогда и в случае sequence labeling можно при векторизации учитывать вектор последнего токена сущности

    векторизация сущностей: start + label_emb, потому что по построению нет гарантии наличия лейбла L_<ENTITY>
    для соответствующего лейбла B-<ENTITY>.
    """
    def __init__(self, sess: tf.Session = None, config: Dict = None, ner_enc: Dict = None, re_enc: Dict = None):
        super().__init__(sess=sess, config=config, ner_enc=ner_enc, re_enc=re_enc)

        # PLACEHOLDERS
        self.ner_labels_ph = None
        self.re_labels_ph = None

        # TENSORS
        self.re_logits_train = None
        self.num_entities = None
        self.re_labels_valid = None
        self.re_labels_test = None
        self.total_loss = None
        self.ner_labels_pred = None

        # LAYERS
        self.entity_emb = None
        self.entity_emb_layer_norm = None
        self.entity_emb_dropout = None
        self.entity_pairs_enc = None

    def _build_re_head(self):
        self.re_logits_train, self.num_entities = self._build_re_head_fn(self.bert_out_train, self.ner_labels_ph)
        re_logits_valid, _ = self._build_re_head_fn(self.bert_out_pred, self.ner_labels_ph)
        re_logits_test, _ = self._build_re_head_fn(self.bert_out_pred, self.ner_labels_pred)

        self.re_labels_valid = tf.argmax(re_logits_valid, axis=-1)
        self.re_labels_test = tf.argmax(re_logits_test, axis=-1)

    def _build_ner_head(self):
        pass

    def _set_placeholders(self):
        super()._set_placeholders()
        self.ner_labels_ph = tf.placeholder(tf.int32, shape=[None, 4], name="ner_labels")  # [i, start, end, label]
        self.re_labels_ph = tf.placeholder(tf.int32, shape=[None, 4], name="re_labels")  # [i, id_head, id_dep, label]

    def _set_layers(self):
        """
        {
            "entity_emb": {
                "use": True,
                "params": {
                    "dim": 16,  # в lee et. al. использовали размерность 20: https://github.com/kentonl/e2e-coref/blob/9d1ee1972f6e34eb5d1dcbb1fd9b9efdf53fc298/experiments.conf#L42
                    "num_labels": 10,
                    "merge_mode": "concat"  # {"add", "concat"},
                    "dropout": 0.3
                }
            }
        }
        :return:
        """
        super()._set_layers()
        self.entity_pairs_enc = GraphEncoder(**self.config["model"]["re"]["biaffine"])

        if self.config["model"]["re"]["entity_emb"]["use"]:
            params = self.config["model"]["re"]["entity_emb"]["params"]
            assert params["merge_mode"] == "concat"
            self.entity_emb = tf.keras.layers.Embedding(params["num_labels"], params["dim"])
            self.entity_emb_dropout = tf.keras.layers.Dropout(params["dropout"])

    def _build_re_head_fn(self,  bert_out, ner_labels):
        """

        :param bert_out:
        :param ner_labels: [N, T], int32
        :return:
        """
        x = self._get_token_level_embeddings(bert_out=bert_out)  # [batch_size, num_tokens, d_bert]

        # entity embeddings
        x, num_entities = get_entities_representation(
            x=x, ner_labels=ner_labels, sparse_labels=True, ff_attn=None, entity_emb_layer=self._entity_emb_fn
        )  # [batch_size, num_ent, D * 3]

        inputs = GraphEncoderInputs(head=x, dep=x)
        logits = self.entity_pairs_enc(inputs, training=self.training_ph)  # [batch_size, num_ent, num_ent, num_rel]
        return logits, num_entities

    def _entity_emb_fn(self, x):
        assert self.entity_emb is not None
        x = self.entity_emb(x)
        x = self.entity_emb_dropout(x, training=self.training_ph)
        return x

    def _set_loss(self, *args, **kwargs):
        assert self.config["model"]["re"]["no_relation_id"] == 0
        logits_shape = tf.shape(self.logits_train)
        labels = tf.scatter_nd(
            indices=self.re_labels_ph[:, :-1], updates=self.re_labels_ph[:, -1], shape=logits_shape[:-1]
        )
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=self.logits_train
        )  # [batch_size, num_entities, num_entities]

        sequence_mask = tf.sequence_mask(self.num_entities, maxlen=logits_shape[1], dtype=tf.float32)
        mask = sequence_mask[:, None, :] * sequence_mask[:, :, None]

        masked_per_example_loss = per_example_loss * mask
        total_loss = tf.reduce_sum(masked_per_example_loss)
        num_pairs = tf.cast(tf.reduce_sum(self.num_entities ** 2), tf.float32)
        num_pairs = tf.maximum(num_pairs, 1.0)
        self.loss = total_loss / num_pairs
        self.total_loss = total_loss

    def _get_feed_dict(self, examples: List[Example], mode: str) -> Dict:
        assert self.ner_enc is not None
        assert self.re_enc is not None

        bert_inputs = self._get_bert_input_for_feed_dict(examples)

        ner_labels = []
        re_labels = []

        # filling
        for i, x in enumerate(examples):
            # entities
            for entity in x.entities:
                start = entity.tokens[0].index_rel
                end = entity.tokens[-1].index_rel
                label = self.ner_enc[entity.label]
                ner_labels.append((i, start, end, label))

            # relations
            if mode != ModeKeys.TEST:
                for arc in x.arcs:
                    assert arc.head_index is not None
                    assert arc.dep_index is not None
                    id_rel = self.re_enc[arc.rel]
                    re_labels.append((i, arc.head_index, arc.dep_index, id_rel))

        if len(ner_labels) == 0:
            ner_labels.append((0, 0, 0, 0))

        d = {
            self.input_ids_ph: bert_inputs.input_ids,
            self.input_mask_ph: bert_inputs.input_mask,
            self.segment_ids_ph: bert_inputs.segment_ids,
            self.first_pieces_coords_ph: bert_inputs.first_pieces_coords,
            self.num_pieces_ph: bert_inputs.num_pieces,
            self.num_tokens_ph: bert_inputs.num_tokens,
            self.training_ph: mode == ModeKeys.TRAIN,
            self.ner_labels_ph: ner_labels
        }

        if mode != ModeKeys.TEST:
            if len(re_labels) == 0:
                re_labels.append((0, 0, 0, 0))

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

        y_true = []
        y_pred = []

        loss = 0.0
        loss_denominator = 0

        gen = batches_gen(
            examples=chunks,
            max_tokens_per_batch=self.config["inference"]["max_tokens_per_batch"],
            pieces_level=True
        )
        for batch in gen:
            feed_dict = self._get_feed_dict(batch, mode=ModeKeys.VALID)
            loss_i, labels_pred = self.sess.run([self.total_loss, self.labels_pred], feed_dict=feed_dict)
            loss += loss_i

            for i, x in enumerate(batch):
                num_entities_i = len(x.entities)
                num_entities_i_squared = num_entities_i ** 2
                loss_denominator += num_entities_i_squared
                y_true_i = [NO_LABEL] * num_entities_i_squared

                for arc in x.arcs:
                    assert arc.head_index is not None
                    assert arc.dep_index is not None
                    y_true_i[num_entities_i * arc.head_index + arc.dep_index] = arc.rel
                y_true += y_true_i

                labels_pred_i = labels_pred[i, :num_entities_i, :num_entities_i]
                assert labels_pred_i.shape[0] == num_entities_i, f"{labels_pred_i.shape[0]} != {num_entities_i}"
                assert labels_pred_i.shape[1] == num_entities_i, f"{labels_pred_i.shape[1]} != {num_entities_i}"

                y_pred_i = [NO_LABEL] * num_entities_i_squared
                for head_index, dep_index in zip(*np.where(labels_pred_i != NO_LABEL_ID)):
                    id_label = labels_pred_i[head_index, dep_index]
                    y_pred_i[num_entities_i * head_index + dep_index] = self.inv_re_enc[id_label]
                y_pred += y_pred_i

        loss /= loss_denominator
        re_metrics = classification_report(y_true=y_true, y_pred=y_pred)

        # total
        performance_info = {
            "loss": loss,
            "metrics": re_metrics,
            "score": re_metrics["micro"]["f1"]
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
            re_labels_pred = self.sess.run(self.labels_pred, feed_dict=feed_dict)  # [N, E, E]

            for i in range(len(batch)):
                chunk = batch[i]
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
                            arc = Arc(id=id_arc, head=head.id, dep=dep.id, rel=rel)
                            parent.arcs.append(arc)


class BertForNerAsDependencyParsingAndRelationExtraction:
    """
    требуется найти и сущности, и отношения между ними.

    векторизация сущностей: [start, end, attn, label_emb]
    """
    pass
