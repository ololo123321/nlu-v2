experiment="coreference_resolution/mention_ranking"
train_data_dir=""
valid_data_dir=""
pretrained_dir=""
output_dir=""
scorer_path=""

python ../jobs/train.py \
  +experiment=${experiment} \
  train_data_dir=${train_data_dir} \
  valid_data_dir=${valid_data_dir} \
  output_dir=${output_dir} \
  hydra.run.dir=${output_dir} \
  model.pretrained_dir=${pretrained_dir} \
  validation.scorer_path=${scorer_path}