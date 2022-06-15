### Inference


### Best models for each experiment
| experiment                               | dataset                | checkpoint                                                        | metric                                   | value  |
|------------------------------------------|------------------------|-------------------------------------------------------------------|------------------------------------------|--------|
| `coreference_resolution/mention_ranking` | `rucor_v5_fixed_edges` | `bert_for_coreference_resolution_mention_ranking_v2_10_epochs`    | avg(bcub-f1, muc-f1, ceafe-f1, ceafm-f1) | 0.6555 |
| `coreference_resolution/mention_pair`    | `rucor_v5_fixed_edges` |                                                                   | avg(bcub-f1, muc-f1, ceafe-f1, ceafm-f1) |        |
| `dependency_parsing`                     | `syntagrus-v2.4`       | `bert_for_dependency_parsing`                                     | las                                      | 0.9435 |
| `dependency_parsing`                     | `syntagrus-v2.8`       |                                                                   | las                                      |        |
| `ner/sequence_labeling`                  | `collection5`          | `bert_for_ner_collection5`                                        | f1-micro, entity-level                   | 0.9642 |
| `ner/sequence_labeling`                  | `collection3`          |                                                                   | f1-micro, entity-level                   |        |
| `ner/sequence_labeling`                  | `rured`                |                                                                   | f1-micro, entity-level                   |        |
| `ner/sequence_labeling`                  | `rurebus`              |                                                                   | f1-micro, entity-level                   |        |
| `ner/span_prediction`                    | `collection5`          | `bert_for_nested_ner_collection5`                                 | f1-micro, entity-level                   | 0.9605 |
| `relation_extraction/biaffine`           | `rured`                | `bert_for_relation_extraction_rured`                              | f1-micro                                 | 0.6582 |
| `relation_extraction/biaffine`           | `rurebus`              |                                                                   | f1-micro                                 |        |
| `joint/ner_re`                           | `rured`                | `bert_for_ner_as_sequence_labeling_and_relation_extraction_rured` | f1, triplets-level                       | 0.5632 |
| `joint/ner_re`                           | `rurebus`              |                                                                   | f1, triplets-level                       |        |

Validation of coreference resolution models is done with [official implementation](https://github.com/conll/reference-coreference-scorers) of necessary metrics.
If done manually, need to load and unpack [release 8.01](https://github.com/conll/reference-coreference-scorers/archive/v8.01.tar.gz).
If done via my docker image `ololo123321/nlu:cuda10.0-runtime-ubuntu18.04-py3.7`, provide the following value for argument `evaluator.scorer_path`: `/app/reference-coreference-scorers-8.01/scorer.pl`.

### Papers