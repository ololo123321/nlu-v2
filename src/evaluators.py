from typing import List, Dict
import tempfile
import os
from src.utils import log, LoggerMixin, parse_conll_metrics
from src.data.io import Example, to_conll
from src.data.datasets import assign_chain_ids
from src.metrics import get_coreferense_resolution_metrics, classification_report, classification_report_ner


class BaseEvaluator(LoggerMixin):
    def __init__(self, logger_parent_name: str = None):
        super().__init__(logger_parent_name=logger_parent_name)

    def __call__(self, examples_gold: List[Example], examples_pred: List[Example]) -> Dict:
        """
        принимает на вход истинные примеры и предикты, выдаёт словарь с метриками
        """


class CoreferenceResolutionEvaluator(BaseEvaluator):
    def __init__(self, scorer_path: str, logger_parent_name: str = None):
        super().__init__(logger_parent_name=logger_parent_name)
        self.scorer_path = scorer_path

    @log
    def __call__(self, examples_gold: List[Example], examples_pred: List[Example]) -> Dict:
        self.logger.info("check examples match")
        assert len(examples_gold) == len(examples_pred)
        for x, y in zip(examples_gold, examples_pred):
            assert x.id == y.id

        self.logger.info("assigning chain ids...")
        for x in examples_gold:
            assign_chain_ids(x)
        for x in examples_pred:
            assign_chain_ids(x)

        path_gold = tempfile.mktemp()
        self.logger.info(f"saving gold data to {path_gold}")
        to_conll(examples=examples_gold, path=path_gold)

        path_pred = tempfile.mktemp()
        self.logger.info(f"saving pred data to {path_pred}")
        to_conll(examples=examples_pred, path=path_pred)

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
        metrics["score"] = (metrics["muc"]["f1"] + metrics["bcub"]["f1"] + metrics["ceafm"]["f1"] + metrics["ceafe"]["f1"]) / 4.0

        self.logger.info("removing temp files")
        os.remove(path_gold)
        os.remove(path_pred)
        return metrics


class DependencyParsingEvaluator(BaseEvaluator):
    def __init__(self, logger_parent_name: str = None):
        super().__init__(logger_parent_name=logger_parent_name)

    @log
    def __call__(self, examples_gold: List[Example], examples_pred: List[Example]) -> Dict:
        """
        Так как документы уже разделены на предложения, то нужны дополнительные проверки того, есть матчинг 1 к 1
        """
        id2gold = {}
        gold_ids = set()
        for x in examples_gold:
            for chunk in x.chunks:
                assert chunk.id not in id2gold.keys(), f'duplicated id: {chunk.id}'
                id2gold[chunk.id] = chunk
                gold_ids.add(chunk.id)
        metrics = {
            "las": 0.0,
            "uas": 0.0,
            "support": 0
        }
        for x in examples_pred:
            for chunk_pred in x.chunks:
                assert chunk_pred.id in gold_ids, f'unknown chunk: {chunk_pred.id}'
                chunk_gold = id2gold[chunk_pred.id]
                assert len(chunk_pred.tokens) == len(chunk_gold.tokens), \
                    f'{len(chunk_pred.tokens)} != {len(chunk_gold.tokens)}'
                for t_pred, t_gold in zip(chunk_pred.tokens, chunk_gold.tokens):
                    assert t_pred.text == t_gold.text, f'{t_pred.text} != {t_gold.text}'
                    metrics["support"] += 1
                    if t_pred.id_head == t_gold.id_head:
                        metrics["uas"] += 1
                        if t_pred.rel == t_gold.rel:
                            metrics["las"] += 1
                gold_ids.remove(chunk_pred.id)
        metrics["las"] /= metrics["support"]
        metrics["uas"] /= metrics["support"]
        if len(gold_ids) > 0:
            self.logger.warning(f'No predictions for {len(gold_ids)} sentences (top-10): {sorted(gold_ids)[:10]}')
        return metrics


class SequenceLabelingEvaluator(BaseEvaluator):
    def __init__(self, logger_parent_name: str = None):
        super().__init__(logger_parent_name=logger_parent_name)

    @log
    def __call__(self, examples_gold: List[Example], examples_pred: List[Example]) -> Dict:
        """
        лейблы должны быть присвоены всем токенам
        """
        y_true = self._get_labels(examples_gold)
        y_pred = self._get_labels(examples_pred)

        y_true_flat = [y for x in y_true for y in x]
        y_pred_flat = [y for x in y_pred for y in x]

        res = {
            "entity_level": classification_report_ner(y_true=y_true, y_pred=y_pred),
            "token_level": classification_report(y_true=y_true_flat, y_pred=y_pred_flat)
        }
        return res

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
