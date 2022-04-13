experiment="coreference_resolution/mention_ranking"
train_data_path=""
valid_data_path=""
pretrained_dir=""
output_dir=""
scorer_path=""

python ../jobs/train.py \
  +experiment=${experiment} \
  train_data_path=${train_data_path} \
  valid_data_path=${valid_data_path} \
  output_dir=${output_dir} \
  hydra.run.dir=${output_dir} \
  model.pretrained_dir=${pretrained_dir} \
  validation.scorer_path=${scorer_path}