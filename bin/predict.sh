test_data_path=""
model_dir=""
predictions_path=""

python ../jobs/predict.py \
  test_data_path=${test_data_path} \
  model_dir=${model_dir} \
  predictions_path=${predictions_path}