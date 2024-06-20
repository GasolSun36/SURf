#! /bin/bash


total_chunks=50
for i in $(seq 0 `expr $total_chunks - 1`)
    do
    echo "Running on chunk $i ..."
    log_file="logs/initial/initial_${i}.log"
    nohup srun -p MoE -n1 -N1 --gres=gpu:1 --quotatype=reserved python initial/generate_initial_data.py \
    --model-path llava-v1.5-7b \
    --image-folder database/data \
    --question-file data/data-665k.json \
    --answers-file output/initial/data_${i}.jsonl \
    --num-chunks $total_chunks \
    --chunk-idx $i \
    1>"$log_file" 2>&1 &
    sleep 2
done


# execute after generating the initial datas
output_file = “output/initial_datas.jsonl”
for chunk_idx in $(seq 0 $(($num_chunks - 1)))
do
    cat output/initial/data_${chunk_idx}.jsonl >> "$output_file"
done