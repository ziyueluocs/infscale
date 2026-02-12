# !/bin/bash

export CUDA_VISIBLE_DEVICES=7
python per_layer_profiling.py --model_type llama \
                             --seq_lengths 64 256 1024 2048 4096 \
                             --batch_sizes 1 2 4 8 16 \
                             --output_suffix "h200"