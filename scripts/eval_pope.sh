#! /bin/bash

python eval/pope.py \
    --model-path checkpoints/surf-7b \
    --image-folder data/ \
    --retrieve-image-folder database/data \ 
    --question-file benchmark/POPE/coco/coco_pope_popular.json \
    --answers-file output/coco_pope_popular_results.jsonl \
    --large-data database/sharegpt4v_1246k.json \
    --index-path database/sharegpt4v_1246k_index.json \
    --image-names-path database/sharegpt4v_1246k_file_names.json \
    --clip-model-name ViT-L-14


# After getting the result, you can directly eval the result:

python eval/eval_pope.py \
    --gt_files benchmark/POPE/coco/coco_pope_popular.json \
    --gen_files output/coco_pope_popular_results.jsonl