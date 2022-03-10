import copy
import tensorflow as tf

from src.model.ner import BertForNerAsSequenceLabeling, BertForNerAsDependencyParsing
from src.model.coreference_resolution import (
    BertForCoreferenceResolutionMentionPair,
    BertForCoreferenceResolutionMentionRanking,
    BertForCoreferenceResolutionMentionRankingNewInference
)
from src.model.dependency_parsing import BertForDependencyParsing
from src.model.relation_extraction import BertForRelationExtraction
from src.data.base import Example, Entity, Token, Span, Arc


def build_examples():
    tokens = [
        Token(text="мама", label="B_FOO", token_ids=[3], index_abs=0, index_rel=0, id_sent=0,
              span_abs=Span(start=0, end=4), span_rel=Span(start=0, end=4), id_head=0, rel="root"),
        Token(text="мыла", label="I_FOO", token_ids=[4, 5], index_abs=1, index_rel=1, id_sent=0,
              span_abs=Span(start=5, end=9), span_rel=Span(start=5, end=9), id_head=0, rel="foo"),
        Token(text="раму", label="B_BAR", token_ids=[6], index_abs=2, index_rel=2, id_sent=0,
              span_abs=Span(start=10, end=14), span_rel=Span(start=10, end=14), id_head=1, rel="bar")
    ]
    entities = [
        Entity(id="T0", text="мама мыла", tokens=tokens[:2], label="FOO", id_chain=0, index=0),
        Entity(id="T1", text="раму", tokens=tokens[2:3], label="BAR", id_chain=1, index=1)
    ]
    arcs = [
        Arc(id="R0", head="T0", dep="T1", rel="BAZ", head_index=0, dep_index=1)
    ]
    text = "мама мыла раму"
    _examples = [
        Example(id="0", filename="0", text=text, tokens=tokens, entities=entities, arcs=arcs, chunks=[
            Example(id="chunk_0", tokens=tokens, entities=entities, arcs=arcs, parent="0")
        ]),
        Example(id="1", filename="1", text=text, tokens=tokens, entities=entities, arcs=arcs, chunks=[
            Example(id="chunk_1", tokens=tokens, entities=entities, arcs=arcs, parent="1")
        ]),
        Example(id="2", filename="2", text=text, tokens=tokens, entities=entities, arcs=arcs, chunks=[
            Example(id="chunk_2", tokens=tokens, entities=entities, arcs=arcs, parent="2")
        ])
    ]
    return _examples


examples = build_examples()

common_config = {
    "model": {
        "bert": {
            "test_mode": True,
            "dir": None,
            "dropout": 0.2,
            "scope": "bert",
            "pad_token_id": 0,
            "cls_token_id": 1,
            "sep_token_id": 2,
            "params": {
                "hidden_size": 16
            }
        },
        "birnn": {
            "use": False,
            "params": {}
        }
    },
    "training": {
        "num_epochs": 1,
        "batch_size": 16,
        "maxlen": 128,
        "max_epochs_wo_improvement": 1,
        "num_train_samples": 100,
    },
    "optimizer": {
        "init_lr": 2e-5,
        "warmup_proportion": 0.1,
    },
    "inference": {
        "max_tokens_per_batch": 100,
        "maxlen": 128,
        "window": 1
    },
    "validation": {}  # чтоб пайчарм не подчёркивал ниже
}

folds = [
    (["0", "1"], ["2"]),
    (["0", "2"], ["1"]),
    (["1", "2"], ["0"])
]


def _test_model(model_cls, config, drop_entities: bool, **kwargs):
    tf.reset_default_graph()
    model = model_cls(sess=None, config=config, **kwargs)
    model.build()

    with tf.Session() as sess:
        model.sess = sess
        model.reset_weights()

        model.train(
            examples_train=examples,
            examples_valid=examples,
            train_op_name="train_op",
            model_dir=None,
            scope_to_save=None,
            verbose=True,
            verbose_fn=None
        )

        examples_test = copy.deepcopy(examples)
        for x in examples_test:
            if drop_entities:
                x.entities = []
            x.arcs = []
            for t in x.tokens:
                t.reset()
        model.predict(examples=examples_test)

    model.sess = None
    model.cross_validate(
        examples=examples,
        folds=folds,
        valid_frac=0.5,
        model_dir=None,
        verbose=False,
        verbose_fn=None
    )


def test_bert_for_ner_as_sequence_labeling():
    ner_enc = {
        "O": 0,
        "B_FOO": 1,
        "I_FOO": 2,
        "B_BAR": 3,
        "I_BAR": 4
    }
    config = common_config.copy()
    config["model"]["ner"] = {
        "use_crf": True,
        "num_labels": len(ner_enc),
        "no_entity_id": 0,
        "prefix_joiner": "-",
        "use_birnn": False,
        "rnn": {
            "num_layers": 1,
            "cell_dim": 8,
            "dropout": 0.5,
            "recurrent_dropout": 0.0
        }
    }
    _test_model(BertForNerAsSequenceLabeling, config=config, ner_enc=ner_enc, drop_entities=True)


def test_bert_for_ner_as_dependency_parsing():
    ner_enc = {
        "O": 0,
        "FOO": 1,
        "BAR": 2
    }
    config = common_config.copy()
    config["model"]["ner"] = {
        "no_entity_id": 0,
        "use_birnn": False,
        "rnn": {
            "num_layers": 1,
            "cell_dim": 8,
            "dropout": 0.5,
            "recurrent_dropout": 0.0
        },
        "biaffine": {
            "num_mlp_layers": 1,
            "activation": "relu",
            "head_dim": 8,
            "dep_dim": 8,
            "dropout": 0.33,
            "num_labels": len(ner_enc),
        }
    }
    _test_model(BertForNerAsDependencyParsing, config=config, ner_enc=ner_enc, drop_entities=True)


def test_bert_for_cr_mention_pair():
    config = common_config.copy()
    config["model"]["coref"] = {
        "use_birnn": False,
        "rnn": {
            "num_layers": 1,
            "cell_dim": 8,
            "dropout": 0.5,
            "recurrent_dropout": 0.0
        },
        "use_attn": True,
        "attn": {
            "hidden_dim": 8,
            "dropout": 0.3,
            "activation": "relu"
        },
        "hoi": {
            "order": 2,
            "w_dropout": 0.5,
            "w_dropout_policy": 0  # 0 - one mask; 1 - different mask
        },
        "biaffine": {
            "num_mlp_layers": 1,
            "activation": "relu",
            "head_dim": 8,
            "dep_dim": 8,
            "dropout": 0.33,
            "num_labels": 1,
            "use_dep_prior": False
        }
    }
    config["validation"] = {
        "path_true": "/tmp/gold.conll",
        "path_pred": "/tmp/pred.conll",
        "scorer_path": "/home/vitaly/reference-coreference-scorers/scorer.pl"
    }
    _test_model(BertForCoreferenceResolutionMentionPair, config=config, drop_entities=False)


def test_bert_for_cr_mention_ranking():
    config = common_config.copy()
    config["model"]["coref"] = {
        "use_birnn": False,
        "rnn": {
            "num_layers": 1,
            "cell_dim": 8,
            "dropout": 0.5,
            "recurrent_dropout": 0.0
        },
        "use_attn": True,
        "attn": {
            "hidden_dim": 8,
            "dropout": 0.3,
            "activation": "relu"
        },
        "hoi": {
            "order": 2,
            "w_dropout": 0.5,
            "w_dropout_policy": 0  # 0 - one mask; 1 - different mask
        },
        "biaffine": {
            "num_mlp_layers": 1,
            "activation": "relu",
            "head_dim": 8,
            "dep_dim": 8,
            "dropout": 0.33,
            "num_labels": 1,
            "use_dep_prior": False
        }
    }
    config["validation"] = {
        "path_true": "/tmp/gold.conll",
        "path_pred": "/tmp/pred.conll",
        "scorer_path": "/home/vitaly/reference-coreference-scorers/scorer.pl"
    }
    _test_model(BertForCoreferenceResolutionMentionRanking, config=config, drop_entities=False)


def test_bert_for_cr_mention_ranking_new_inference():
    config = common_config.copy()
    config["model"]["coref"] = {
        "use_birnn": False,
        "rnn": {
            "num_layers": 1,
            "cell_dim": 8,
            "dropout": 0.5,
            "recurrent_dropout": 0.0
        },
        "use_attn": True,
        "attn": {
            "hidden_dim": 8,
            "dropout": 0.3,
            "activation": "relu"
        },
        "hoi": {
            "order": 2,
            "w_dropout": 0.5,
            "w_dropout_policy": 0  # 0 - one mask; 1 - different mask
        },
        "biaffine": {
            "num_mlp_layers": 1,
            "activation": "relu",
            "head_dim": 8,
            "dep_dim": 8,
            "dropout": 0.33,
            "num_labels": 1,
            "use_dep_prior": False
        }
    }
    config["validation"] = {
        "path_true": "/tmp/gold.conll",
        "path_pred": "/tmp/pred.conll",
        "scorer_path": "/home/vitaly/reference-coreference-scorers/scorer.pl"
    }
    _test_model(BertForCoreferenceResolutionMentionRankingNewInference, config=config, drop_entities=False)


def test_bert_for_dependency_parsing():
    config = common_config.copy()
    config["model"]["bert"]["root_token_id"] = 10
    config["model"]["parser"] = {
        "use_birnn": False,
        "rnn": {
            "num_layers": 1,
            "cell_dim": 8,
            "dropout": 0.5,
            "recurrent_dropout": 0.0
        },
        "biaffine_arc": {
            "num_mlp_layers": 1,
            "activation": "relu",
            "head_dim": 8,
            "dep_dim": 8,
            "dropout": 0.33,
            "num_labels": 1,
        },
        "biaffine_type": {
            "num_mlp_layers": 1,
            "activation": "relu",
            "head_dim": 8,
            "dep_dim": 8,
            "dropout": 0.33,
            "num_labels": 3,
        }
    }

    rel_enc = {
        "root": 0,
        "foo": 1,
        "bar": 2
    }

    _test_model(BertForDependencyParsing, config=config, rel_enc=rel_enc, drop_entities=True)


def test_bert_for_relation_extraction():
    ner_enc = {
        "O": 0,
        "FOO": 1,
        "BAR": 2
    }
    re_enc = {
        "O": 0,
        "BAZ": 1,
    }
    config = common_config.copy()
    config["model"]["re"] = {
        "no_relation_id": 0,
        "entity_emb": {
            "use": True,
            "params": {
                "dim": 16,
                "num_labels": len(ner_enc),
                "merge_mode": "concat",
                "dropout": 0.3
            }
        },
        "biaffine": {
            "num_mlp_layers": 1,
            "activation": "relu",
            "head_dim": 8,
            "dep_dim": 8,
            "dropout": 0.33,
            "num_labels": len(re_enc),
        }
    }

    _test_model(BertForRelationExtraction, config=config, ner_enc=ner_enc, re_enc=re_enc, drop_entities=False)
