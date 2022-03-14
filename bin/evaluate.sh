experiment="coreference_resolution/evaluate"
gold_data_dir=""
pred_data_dir=""
output_path=null

python ../jobs/evaluate.py \
  +experiment=${experiment} \
  gold_data_dir=${gold_data_dir} \
  pred_data_dir=${pred_data_dir} \
  output_path=${output_path}