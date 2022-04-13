from typing import List, Dict
import tempfile
import os
from src.utils import log, LoggerMixin, parse_conll_metrics
from src.data.io import Example, to_conll
from src.data.datasets import assign_chain_ids
from src.metrics import get_coreferense_resolution_metrics


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
        pass
