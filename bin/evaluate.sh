experiment="dependency_parsing"
gold_data_path=""
predictions_path=""
metrics_path=null

python ../jobs/evaluate.py \
  +experiment=${experiment} \
  gold_data_path=${gold_data_path} \
  predictions_path=${predictions_path} \
  metrics_path=${metrics_path}