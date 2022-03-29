experiment="coreference_resolution/mention_ranking"
train_data_dir=""
valid_data_dir=""
output_dir=""

python ../jobs/train.py \
  +experiment=${experiment} \
  train_data_dir=${train_data_dir} \
  valid_data_dir=${valid_data_dir} \
  output_dir=${output_dir} \
  hydra.run.dir=${output_dir}