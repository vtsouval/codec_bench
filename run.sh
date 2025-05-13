#!/bin/bash

# List of datasets to run
datasets=("esc50" "cremad" "slurp" "timit_dialect" "msdb" "gtzan" "ravdess" "audiomnist" "emovo" "us8k" "vivae")

base_cmd="python main.py --save_dir ./assets --ds_dir ./datasets --model_name Llasa-1B --codec_name xcodec2 --num_epochs 10 --eval_freq 500 --patience 15 --batch_size 2"

# Loop over datasets
for ds in "${datasets[@]}"; do
    echo "🚀 Starting training for dataset: $ds"
    $base_cmd --ds_name "$ds"
    echo "✅ Finished training for dataset: $ds"
    echo "------------------------------------------------------"
done