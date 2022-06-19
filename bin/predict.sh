test_data_path=""
model_dir=""
predictions_path=""
ignore_without_annotation=false

python ../jobs/predict.py \
  test_data_path=${test_data_path} \
  model_dir=${model_dir} \
  predictions_path=${predictions_path} \
  ++dataset.ignore_without_annotation=${ignore_without_annotation}