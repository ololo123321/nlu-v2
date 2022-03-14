from typing import List, Dict
import tempfile
import os
from src.utils import log, LoggerMixin, parse_conll_metrics
from src.data.io import Example, to_conll
from src.metrics import get_coreferense_resolution_metrics


class BaseEvaluator(LoggerMixin):
    def __init__(self, logger_parent_name: str = None):
        super().__init__(logger_parent_name=logger_parent_name)

    @log
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
        path_gold = tempfile.mktemp()
        path_pred = tempfile.mktemp()
        to_conll(examples=examples_gold, path=path_gold)
        to_conll(examples=examples_pred, path=path_pred)

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

        os.remove(path_gold)
        os.remove(path_pred)
        return metrics
