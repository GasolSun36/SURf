#! /bin/bash


total_chunks=50
for i in $(seq 0 `expr $total_chunks - 1`)
    do
    echo "Running on chunk $i ..."
    log_file="logs/initial/context_${i}.log"
    nohup srun -p MoE -n1 -N1 --gres=gpu:1 --quotatype=reserved python initial/generate_context_data.py \
    --model-path llava-v1.5-7b \
    --image-folder database/data \
    --question-file initial/initial_datas.json \
    --answers-file output/initial/reanswer_data_${i}.jsonl \
    --topk 10 \
    --num-chunks $total_chunks \
    --chunk-idx $i \
    1>"$log_file" 2>&1 &
    sleep 2
done


# execute after generating the initial datas
output_file = “output/context_datas.jsonl”
for chunk_idx in $(seq 0 $(($num_chunks - 1)))
do
    cat output/initial/reanswer_data_${chunk_idx}.jsonl >> "$output_file"
done