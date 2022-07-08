Some experiments with classic nlp tasks.  
Backbone for all experiments: [ruBERT from DeepPavlov](http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v2.tar.gz).  
All models implemented in tensorflow-1.15.  
Inference and evaluation of all models can be done in docker: see `bin/predict_docker.sh`, `bin/evaluate_docker.sh`.  
Inference can be done on cpu and gpu.

### Inference guide
1. Create directory with documents to make predictions for. Some tasks require additional information except plain text in `.txt` file:
    * `coreference_resolution` - `.ann` file with mentions
    * `relation_extraction` - `.ann` file with entities.  
    
    The simplest way to run dependency parser on unlabeled data: 
    1. create directory `data_dir` with `.txt` files: one sentence per file.
    2. `python predict.py test_data_path=data_dir ++dataset.from_brat=true ...`
2. Set valid paths in `predict_docker.sh` file.
3. `bash predict_docker.sh`

### Best models for each experiment
| experiment                               | dataset                | checkpoint                                                             | url                                     | metric                                   | value  |
|------------------------------------------|------------------------|------------------------------------------------------------------------|-----------------------------------------|------------------------------------------|--------|
| `coreference_resolution/mention_ranking` | `rucor_v5_fixed_edges` | `bert_for_coreference_resolution_mention_ranking_v2_10_epochs`         | https://disk.yandex.ru/d/dx0AkUQiyOjqPQ | avg(bcub-f1, muc-f1, ceafe-f1, ceafm-f1) | 0.6555 |
| `coreference_resolution/mention_pair`    | `rucor_v5_fixed_edges` | `bert_for_coreference_resolution_mention_pair`                         | https://disk.yandex.ru/d/5AaPyNx0MdL2VA | avg(bcub-f1, muc-f1, ceafe-f1, ceafm-f1) | 0.6397 |
| `dependency_parsing`                     | `syntagrus-v2.4`       | `bert_for_dependency_parsing`                                          | https://disk.yandex.ru/d/kxJ34yyusmsmVA | las                                      | 0.9435 |
| `dependency_parsing`                     | `syntagrus-v2.8`       | `bert_for_dependency_parsing_syntagrus-v2.8`                           | https://disk.yandex.ru/d/mf4_i4Vi9jBJfw | las                                      | 0.9276 |
| `ner/sequence_labeling`                  | `collection5`          | `bert_for_ner_collection5`                                             | https://disk.yandex.ru/d/NwNA0fVXoVnVbw | f1-micro, entity-level                   | 0.9642 |
| `ner/sequence_labeling`                  | `collection3`          | `bert_for_ner_collection3`                                             | https://disk.yandex.ru/d/aJ7Ce7L2h44CyQ | f1-micro, entity-level                   | 0.9762 |
| `ner/sequence_labeling`                  | `rured`                | `bert_for_ner_rured`                                                   | https://disk.yandex.ru/d/oQIAPYTWGIFX3w | f1-micro, entity-level                   | 0.8669 |
| `ner/sequence_labeling`                  | `rurebus`              | `bert_for_ner_rurebus`                                                 | https://disk.yandex.ru/d/miRxzDXVl3b1-w | f1-micro, entity-level                   | 0.5713 |
| `ner/span_prediction`                    | `collection5`          | `bert_for_nested_ner_collection5`                                      | https://disk.yandex.ru/d/AhedezHvgPyODg | f1-micro, entity-level                   | 0.9605 |
| `relation_extraction/biaffine`           | `rured`                | `bert_for_relation_extraction_rured`                                   | https://disk.yandex.ru/d/Uk2dWBeG0kNBzg | f1-micro                                 | 0.6582 |
| `relation_extraction/biaffine`           | `rurebus`              | `bert_for_relation_extraction_rurebus_v2`                              | https://disk.yandex.ru/d/8wI3I8_tZDyzlg | f1-micro                                 | 0.4233 |
| `joint/ner_re`                           | `rured`                | `bert_for_ner_as_sequence_labeling_and_relation_extraction_rured`      | https://disk.yandex.ru/d/Tk4aWNHe-Eny7g | f1, triplets-level                       | 0.5632 |
| `joint/ner_re`                           | `rurebus`              | `bert_for_ner_as_sequence_labeling_and_relation_extraction_rurebus_v2` | https://disk.yandex.ru/d/5TiOLqIG_A9OJg | f1, triplets-level                       | 0.1714 |

Validation of coreference resolution models is done with [official implementation](https://github.com/conll/reference-coreference-scorers) of necessary metrics.
If done manually, need to load and unpack [release 8.01](https://github.com/conll/reference-coreference-scorers/archive/v8.01.tar.gz).
If done via my docker image `ololo123321/nlu:cuda10.0-runtime-ubuntu18.04-py3.7`, provide the following value for argument `evaluator.scorer_path`: `/app/reference-coreference-scorers-8.01/scorer.pl`.

### Papers

| paper                                                                                                                                            | experiment                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| [Higher-order Coreference Resolution with Coarse-to-fine Inference](https://arxiv.org/abs/1804.05392)                                            | `coreference_resolution`                        |
| [Neural Coreference Resolution with Deep Biaffine Attention by Joint Mention Detection and Mention Clustering](https://arxiv.org/abs/1805.04893) | `coreference_resolution`                        |
| [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)                                                        | `dependency_parsing`                            |
| [Named Entity Recognition as Dependency Parsing](https://arxiv.org/abs/2005.07150)                                                               | `ner/span_prediction`                           |
| [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)                                        | `relation_extraction`                           |
| [End-to-end neural relation extraction using deep biaffine attention](https://arxiv.org/abs/1812.11275)                                          | `joint/ner_re`, `relation_extraction`           |