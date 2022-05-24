from typing import List, Tuple, Set
from collections import namedtuple
import tempfile
import os
import json
from src.utils import log, LoggerMixin, parse_conll_metrics, classification_report_to_string
from src.data.io import Example, to_conll
from src.data.base import NO_LABEL, Entity
from src.data.datasets import assign_chain_ids
from src.metrics import (
    get_coreferense_resolution_metrics,
    classification_report,
    classification_report_ner,
    classification_report_set
)

Metric = namedtuple("Metric", ["value", "string"])


class BaseEvaluator(LoggerMixin):
    def __init__(self, allow_examples_mismatch: bool = False, logger_parent_name: str = None):
        """
        allow_examples_mismatch - могут ли различаться множества примеров gold и pred
        """
        super().__init__(logger_parent_name=logger_parent_name)
        self.allow_examples_mismatch = allow_examples_mismatch

    def __call__(self, examples_gold: List[Example], examples_pred: List[Example]) -> Metric:
        """
        принимает на вход истинные примеры и предикты, выдаёт словарь с метриками
        """

    def _check_examples_number(self, examples_gold: List[Example], examples_pred: List[Example]):
        n = len(examples_gold)
        m = len(examples_pred)
        if not self.allow_examples_mismatch:
            assert n == m, f'{n} != {m}'
        else:
            diff = n - m
            if diff != 0:
                gold_ids = {x.id for x in examples_gold}
                pred_ids = {x.id for x in examples_pred}
                self.logger.warning(
                    f"number of gold and pred examples mismatch: "
                    f"num_gold: {n}, num_pred: {m}, num_gold - num_pred: {diff}. "
                    f"Number of common examples: {len(gold_ids & pred_ids)}"
                )

    def _get_examples_to_compare(
            self, examples_gold: List[Example], examples_pred: List[Example]
    ) -> Tuple[List[Example], List[Example]]:
        """
        ensure the same set and order
        """
        id2example = {x.id: x for x in examples_pred}
        examples_gold_sort = []
        examples_pred_sort = []
        for x in examples_gold:
            if x.id in id2example.keys():
                examples_gold_sort.append(x)
                examples_pred_sort.append(id2example[x.id])
            elif not self.allow_examples_mismatch:
                raise Exception(f'no predictions for example {x.id}')
        return examples_gold_sort, examples_pred_sort


class CoreferenceResolutionEvaluator(BaseEvaluator):
    def __init__(self, scorer_path: str, allow_examples_mismatch: bool = False, logger_parent_name: str = None):
        super().__init__(allow_examples_mismatch=allow_examples_mismatch, logger_parent_name=logger_parent_name)
        self.scorer_path = scorer_path

    @log
    def __call__(self, examples_gold: List[Example], examples_pred: List[Example]) -> Metric:
        self._check_examples_number(examples_gold=examples_gold, examples_pred=examples_pred)
        examples_gold_sort, examples_pred_sort = self._get_examples_to_compare(
            examples_gold=examples_gold, examples_pred=examples_pred
        )

        self.logger.info("assigning chain ids...")
        for x, y in zip(examples_gold_sort, examples_pred_sort):
            assert x.id == y.id, f'{x.id} != {y.id}'
            assign_chain_ids(x)
            assign_chain_ids(y)

        path_gold = tempfile.mktemp()
        self.logger.info(f"saving gold data to {path_gold}")
        to_conll(examples=examples_gold_sort, path=path_gold)

        path_pred = tempfile.mktemp()
        self.logger.info(f"saving pred data to {path_pred}")
        to_conll(examples=examples_pred_sort, path=path_pred)

        self.logger.info("compute metrics...")
        metrics = {}
        for metric in ["muc", "bcub", "ceafm", "ceafe", "blanc"]:
            stdout = get_coreferense_resolution_metrics(
                path_true=path_gold,
                path_pred=path_pred,
                scorer_path=self.scorer_path,
                metric=metric
            )
            is_blanc = metric == "blanc"
            metrics[metric] = parse_conll_metrics(stdout=stdout, is_blanc=is_blanc)
        metrics["score"] = (metrics["muc"]["f1"] + metrics["bcub"]["f1"] + metrics["ceafm"]["f1"] + metrics["ceafe"][
            "f1"]) / 4.0

        self.logger.info("removing temp files")
        os.remove(path_gold)
        os.remove(path_pred)
        return Metric(value=metrics, string=json.dumps(metrics, indent=2))


# TODO: allow_examples_mismatch
class DependencyParsingEvaluator(BaseEvaluator):
    def __init__(self, allow_examples_mismatch: bool = False, logger_parent_name: str = None):
        super().__init__(allow_examples_mismatch=allow_examples_mismatch, logger_parent_name=logger_parent_name)

    @log
    def __call__(self, examples_gold: List[Example], examples_pred: List[Example]) -> Metric:
        """
        Так как документы уже разделены на предложения, то нужны дополнительные проверки того, есть матчинг 1 к 1
        """
        self._check_examples_number(examples_gold=examples_gold, examples_pred=examples_pred)
        examples_gold_sort, examples_pred_sort = self._get_examples_to_compare(
            examples_gold=examples_gold, examples_pred=examples_pred
        )
        metrics = {
            "las": 0.0,
            "uas": 0.0,
            "support": 0
        }
        for x, y in zip(examples_gold_sort, examples_pred_sort):
            assert x.id == y.id, f'{x.id} != {y.id}'
            for chunk_gold, chunk_pred in zip(x.chunks, y.chunks):
                assert chunk_gold.id == chunk_pred.id, f'{chunk_gold.id} != {chunk_pred.id}'
                assert len(chunk_pred.tokens) == len(chunk_gold.tokens), \
                    f'{len(chunk_pred.tokens)} != {len(chunk_gold.tokens)}'
                for t_pred, t_gold in zip(chunk_pred.tokens, chunk_gold.tokens):
                    assert t_pred.text == t_gold.text, f'{t_pred.text} != {t_gold.text}'
                    metrics["support"] += 1
                    if t_pred.id_head == t_gold.id_head:
                        metrics["uas"] += 1
                        if t_pred.rel == t_gold.rel:
                            metrics["las"] += 1
        metrics["las"] /= metrics["support"]
        metrics["uas"] /= metrics["support"]
        return Metric(value=metrics, string=json.dumps(metrics, indent=2))


class NerEvaluator(BaseEvaluator):
    def __init__(self, allow_examples_mismatch: bool = False, logger_parent_name: str = None):
        super().__init__(allow_examples_mismatch=allow_examples_mismatch, logger_parent_name=logger_parent_name)

    @log
    def __call__(self, examples_gold: List[Example], examples_pred: List[Example]) -> Metric:
        self._check_examples_number(examples_gold=examples_gold, examples_pred=examples_pred)
        examples_gold_sort, examples_pred_sort = self._get_examples_to_compare(
            examples_gold=examples_gold, examples_pred=examples_pred
        )

        for x, y in zip(examples_gold_sort, examples_pred_sort):
            assert x.id == y.id, f'{x.id} != {y.id}'
            assert len(x.tokens) == len(y.tokens), f'[{x.id}] {len(x.tokens)} != {len(y.tokens)}'
            x.assign_labels_to_tokens()
            y.assign_labels_to_tokens()

        y_true = self._get_labels(examples_gold_sort)
        y_pred = self._get_labels(examples_pred_sort)

        y_true_flat = [y for x in y_true for y in x]
        y_pred_flat = [y for x in y_pred for y in x]

        res = {
            "entity_level": classification_report_ner(y_true=y_true, y_pred=y_pred),
            "token_level": classification_report(y_true=y_true_flat, y_pred=y_pred_flat)
        }
        s = ''
        s += '\n'
        s += 'entity level:'
        s += '\n\n'
        s += classification_report_to_string(res["entity_level"])
        s += '\n'
        s += '=' * 80
        s += '\n'
        s += 'token level:'
        s += '\n\n'
        s += classification_report_to_string(res["token_level"])
        return Metric(value=res, string=s)

    @staticmethod
    def _get_labels(examples: List[Example]) -> List[List[str]]:
        labels = []
        for x in examples:
            labels_i = []
            for t in x.tokens:
                assert isinstance(t.label, str), \
                    f'expected token label to be string, but got {t.label} of type {type(t.label)}'
                labels_i.append(t.label)
            labels.append(labels_i)
        return labels


class RelationExtractionEvaluator(BaseEvaluator):
    def __init__(self, allow_examples_mismatch: bool = False, logger_parent_name: str = None):
        super().__init__(allow_examples_mismatch=allow_examples_mismatch, logger_parent_name=logger_parent_name)

    @log
    def __call__(self, examples_gold: List[Example], examples_pred: List[Example]) -> Metric:
        """
        сущности не нужно было предсказывать -> они совпадают с истинными с точностью до идентификаторов
        но лучше не завязываться на идекнтификаторы, а рассматривать сущность как спан
        """
        self._check_examples_number(examples_gold=examples_gold, examples_pred=examples_pred)
        examples_gold_sort, examples_pred_sort = self._get_examples_to_compare(
            examples_gold=examples_gold, examples_pred=examples_pred
        )
        y_true = []
        y_pred = []
        span2index = {}

        def get_entity_span(e):
            return e.tokens[0].span_abs.start, e.tokens[-1].span_abs.end

        def get_flat_labels(x):
            m = len(span2index)
            labels_flat = [NO_LABEL] * m ** 2
            id2entity = {e.id: e for e in x.entities}
            for a in x.arcs:
                i = span2index[get_entity_span(id2entity[a.head])]
                j = span2index[get_entity_span(id2entity[a.dep])]
                labels_flat[i * m + j] = a.rel
            return labels_flat

        for x, y in zip(examples_gold_sort, examples_pred_sort):
            assert x.id == y.id, f'{x.id} != {y.id}'
            entities_gold = {get_entity_span(e) for e in x.entities}
            entities_pred = {get_entity_span(e) for e in y.entities}
            # в случае brat-формата при подгрузке примеров могут поехать спаны сущностей из-за
            # удаления плохих символов из текста (см. src.data.io.parse_example)
            # TODO: как-то обрабатывать этот кейс
            assert entities_gold == entities_pred, f'{entities_gold} != {entities_pred}'
            for i, e in enumerate(x.entities):
                span2index[get_entity_span(e)] = i
            y_true += get_flat_labels(x)
            y_pred += get_flat_labels(y)
            span2index.clear()
        res = classification_report(y_true=y_true, y_pred=y_pred)
        s = "\n" + classification_report_to_string(res)
        return Metric(value=res, string=s)


class NerAndRelationExtractionEvaluator(NerEvaluator):
    def __init__(self, allow_examples_mismatch: bool = False, logger_parent_name: str = None):
        super().__init__(allow_examples_mismatch=allow_examples_mismatch, logger_parent_name=logger_parent_name)

    @log
    def __call__(self, examples_gold: List[Example], examples_pred: List[Example]) -> Metric:
        self._check_examples_number(examples_gold=examples_gold, examples_pred=examples_pred)
        examples_gold_sort, examples_pred_sort = self._get_examples_to_compare(
            examples_gold=examples_gold, examples_pred=examples_pred
        )

        gold_triples = set()  # (head, dep, rel). {head, dep} = (start, end, label)
        pred_triples = set()
        for x, y in zip(examples_gold_sort, examples_pred_sort):
            assert x.id == y.id, f'{x.id} != {y.id}'
            assert len(x.tokens) == len(y.tokens), f'[{x.id}] {len(x.tokens)} != {len(y.tokens)}'
            x.assign_labels_to_tokens()
            y.assign_labels_to_tokens()
            for t in self._get_triples(x):
                gold_triples.add((x.id, t))
            for t in self._get_triples(y):
                pred_triples.add((x.id, t))

        y_true = self._get_labels(examples_gold_sort)
        y_pred = self._get_labels(examples_pred_sort)

        y_true_flat = [y for x in y_true for y in x]
        y_pred_flat = [y for x in y_pred for y in x]

        # ner
        ner_metrics = {
            "entity_level": classification_report_ner(y_true=y_true, y_pred=y_pred),
            "token_level": classification_report(y_true=y_true_flat, y_pred=y_pred_flat)
        }

        # e2e
        e2e_metrics = classification_report_set(y_true=gold_triples, y_pred=pred_triples)

        # result
        res = {
            "ner": ner_metrics,
            "e2e": e2e_metrics
        }

        # string
        s = ''
        s += '\n'
        s += 'entity level:'
        s += '\n\n'
        s += classification_report_to_string(res["ner"]["entity_level"])
        s += '\n'
        s += '=' * 80
        s += '\n'
        s += 'token level:'
        s += '\n\n'
        s += classification_report_to_string(res["ner"]["token_level"])
        s += '\n'
        s += '=' * 80
        s += '\n'
        s += 'end-to-end:'
        s += '\n\n'
        s += classification_report_to_string({"e2e": res["e2e"]})
        return Metric(value=res, string=s)

    def _get_triples(self, x: Example) -> Set[Tuple[Tuple, Tuple, str]]:
        res = set()
        id2entity = {e.id: e for e in x.entities}
        for arc in x.arcs:
            head = self._entity2key(id2entity[arc.head])
            dep = self._entity2key(id2entity[arc.dep])
            item = head, dep, arc.rel
            res.add(item)
        return res

    @staticmethod
    def _entity2key(entity: Entity) -> Tuple[int, int, str]:
        return entity.tokens[0].span_abs.start, entity.tokens[-1].span_abs.end, entity.label
