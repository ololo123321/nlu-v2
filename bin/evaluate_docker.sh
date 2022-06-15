experiment="coreference_resolution/mention_ranking"
gold_data_path="/data/rucor_v5_fixed_edges_test"
predictions_path="/data/cr_preds_docker"
scorer_path="/app/reference-coreference-scorers-8.01/scorer.pl"
metrics_path=null
allow_examples_mismatch=true

docker run \
	-it \
	--rm \
	-v /home/vitaly/Desktop/nlu/:/data \
	-v "$(cd ..; pwd)":/nlu \
	ololo123321/nlu:cuda10.0-runtime-ubuntu18.04-py3.7 python /nlu/jobs/evaluate.py \
    +experiment=${experiment} \
    gold_data_path=${gold_data_path} \
    predictions_path=${predictions_path} \
    metrics_path=${metrics_path} \
    ++evaluator.scorer_path=${scorer_path} \
    ++evaluator.allow_examples_mismatch=${allow_examples_mismatch}