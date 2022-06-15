test_data_path="/data/rucor_v5_fixed_edges_test"
model_dir="/data/models/bert_for_coreference_resolution_mention_ranking_v2_10_epochs"
predictions_path="/data/cr_preds_docker"
vocab_path="/nlu/data/vocab.txt"
num_examples_test=3

docker run \
	-it \
	--rm \
	-v /home/vitaly/Desktop/nlu:/data \
	-v "$(cd ..; pwd)":/nlu \
	ololo123321/nlu:cuda10.0-runtime-ubuntu18.04-py3.7 python /nlu/jobs/predict.py \
		test_data_path=${test_data_path} \
  	model_dir=${model_dir} \
  	predictions_path=${predictions_path} \
  	++num_examples_test=${num_examples_test} \
  	+tokenizer.vocab_file=${vocab_path} --cfg job