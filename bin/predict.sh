test_data_path=""
model_dir=""
output_dir=""

python ../jobs/predict.py \
  test_data_dir=${test_data_path} \
  model_dir=${model_dir} \
  output_dir=${output_dir}