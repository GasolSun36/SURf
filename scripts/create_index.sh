#! /bin/bash

python database/create_index.py \
    --model_name ViT-L-14 \
    --pretrained openai \
    --data_file database/sharegpt4v_1246k.json \
    --output_vectors database/sharegpt4v_1246k_file_names.npy \
    --output_index database/sharegpt4v_1246k.index \
    --folder_path data/ \
    --batch_size 4096
