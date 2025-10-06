#!/bin/bash

# python ./get_prod.py \
# --model_name Qwen3-4B-Base \
# --checkpoint ./checkpoint/Qwen3-4B-Base \
# --train_path ./data/vihallu-train.csv \
# --private_path ./data/vihallu-private-test.csv \
# --save_dir ./data/prod

# python ./get_prod.py \
# --model_name Qwen2.5-7B-Instruct \
# --checkpoint ./checkpoint/Qwen2.5-7B-Instruct \
# --train_path ./data/vihallu-train.csv \
# --private_path ./data/vihallu-private-test.csv \
# --save_dir ./data/prod

# python ./get_prod.py \
# --model_name Qwen2.5-7B \
# --checkpoint ./checkpoint/Qwen2.5-7B \
# --train_path ./data/vihallu-train.csv \
# --private_path ./data/vihallu-private-test.csv \
# --save_dir ./data/prod

# python ./ensemble_model.py

# python ./get_diff_res.py \
# --csv1 ./data/submission_ensemble_nn.csv \
# --csv2 ./data/qwen2.5-7b.csv

# python ./predict_2class.py --task 01
# python ./predict_2class.py --task 02
# python ./predict_2class.py --task 12

python ./final_submit.py
