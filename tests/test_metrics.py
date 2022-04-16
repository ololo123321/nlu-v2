import pytest
from src.metrics import f1_precision_recall_support, classification_report, classification_report_ner


@pytest.mark.parametrize("tp, fp, fn, expected", [
    pytest.param(0, 0, 0, {"f1": 0.0, "precision": 0.0, "recall": 0.0, "support": 0}),
    pytest.param(1, 0, 0, {"f1": 1.0, "precision": 1.0, "recall": 1.0, "support": 1}),
    pytest.param(1, 1, 0, {"f1": 2/3, "precision": 0.5, "recall": 1.0, "support": 1}),
    pytest.param(1, 1, 1, {"f1": 0.5, "precision": 0.5, "recall": 0.5, "support": 2})
])
def test_f1_precision_recall_support(tp, fp, fn, expected):
    actual = f1_precision_recall_support(tp=tp, fp=fp, fn=fn)
    assert actual == expected


@pytest.mark.parametrize("y_true, y_pred, expected", [
    pytest.param(
        [], [],
        {
            "micro": {"f1": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0, "support": 0},
        }
    ),
    pytest.param(
        [["B-ORG"], ["B-LOC", "I-LOC"]],
        [["B-ORG"], ["B-LOC", "O"]],
        {
            "ORG": {"f1": 1.0, "precision": 1.0, "recall": 1.0, "tp": 1, "fp": 0, "fn": 0, "support": 1},
            "LOC": {"f1": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 1, "fn": 1, "support": 1},
            "micro": {"f1": 0.5, "precision": 0.5, "recall": 0.5, "tp": 1, "fp": 1, "fn": 1, "support": 2},
        }
    )
])
def test_classification_report_ner(y_true, y_pred, expected):
    actual = classification_report_ner(y_true, y_pred)
    assert actual == expected


@pytest.mark.parametrize("y_true, y_pred, expected", [
    pytest.param(
        [], [],
        {
            "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0, "tp": 0, "fp": 0, "fn": 0}
        }
    ),
    pytest.param(
        [0], [0],
        {
            "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0, "tp": 0, "fp": 0, "fn": 0}
        }
    ),
    pytest.param(
        [1], [1],
        {
            1: {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1, "tp": 1, "fp": 0, "fn": 0},
            "micro": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1, "tp": 1, "fp": 0, "fn": 0}
        }
    ),
    pytest.param(
        [0, 1, 2], [0, 1, 1],
        {
            1: {"precision": 0.5, "recall": 1.0, "f1": 2. / 3., "support": 1, "tp": 1, "fp": 1, "fn": 0},
            2: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1, "tp": 0, "fp": 0, "fn": 1},
            "micro": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2, "tp": 1, "fp": 1, "fn": 1}
        }
    ),
])
def test_classification_report(y_true, y_pred, expected):
    actual = classification_report(y_true=y_true, y_pred=y_pred)
    # print(actual)
    assert actual == expected
