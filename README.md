
## SURf: Teaching Large Vision-Language Models to Selectively Utilize Retrieved Information



<img src="assets/surf.png" alt="SURf" style="width: 10%; vertical-align: middle;"> SURf: Teaching Large Vision-Language Models to Selectively Utilize Retrieved Information
</p>


[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This is a official repository for SURf, which proposes a self-refinement framework designed to teach LVLMs to **S**electively **U**tilize **R**etrieved In**f**ormation (SURf).

## Introduction

Pipeline of SURf:

![image](https://img.picui.cn/free/2024/06/17/666ff36e837ee.png)


## How to Run SURf

#### Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

The main implementation of SURf is based on LLaVA codebase with some changes to implement multi-images training. If you want to implement SURf on other models, you can follow the steps as the follows:

- Add the new prompt template and change in **default_conversation** `conversation.py`..
- change `train.py`
    - `__get_item__` in `LazySupervisedDataset` in  to implement multi-images training.
    - `__call__` in `DataCollatorForSupervisedDataset` 
    - `preprocess` function

### Datasets
We take `POPE` as an example. Datasets are in `SURf/benchmark/POPE/coco`. Before inference, you need to download the images from [COCO val 2014](https://cocodataset.org/#download) into the `SURF/data` folder.


### Prepare Retrieval Database and Index
We use [CLIP](https://github.com/openai/CLIP) as the visual encoder and [sharegpt4v](https://sharegpt4v.github.io/) data collection in this phrase. Then, we use faiss to index the image embedding to index. You can use the following command in `create_index.sh`:
```bash
python database/create_index.py \
    --model_name ViT-L-14 \
    --pretrained openai \
    --data_file database/sharegpt4v_1246k.json \
    --output_vectors database/sharegpt4v_1246k_file_names.npy \
    --output_index database/sharegpt4v_1246k.index \
    --folder_path data/ \
    --batch_size 4096

```

It will take about **2.5** days to embed all the images in the sharegpt4v data collection. We will release the index file in the future.



### Precess LLaVA SFT data
According to the settings of llava, we get the sft data, then we split each data according to the conversation rounds:

```bash
python initial/precess_sft_data.py \
    --input-file data/llava_sft_data.json \
    --output-file data/data-665k.json
```
We finally get 665k of the single-round data in `SURf/data`. We will release the data in the future.


### Collect Initial Wrong Instruction

First, we prompt the frozen parameter LVLMs (llava-1.5-7B is used here) to initialize our results.

```bash
total_chunks=50
for i in $(seq 0 `expr $total_chunks - 1`)
    do
    echo "Running on chunk $i ..."
    log_file="SURf/logs/initial/initial_${i}.log"
    nohup srun -p MoE -n1 -N1 --gres=gpu:1 --quotatype=reserved python initial/generate_initial_data.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder database/data \
    --question-file data/data-665k.json \
    --answers-file output/initial/data_${i}.jsonl \
    --num-chunks $total_chunks \
    --chunk-idx $i \
    1>"$log_file" 2>&1 &
    sleep 2
done
```
We strongly recommend using multi-card batch inference because the amount of data is large. And after that, we can combine the sub-file in the following bash:

```bash
# execute after generating the initial datas
output_file = “output/initial_datas.jsonl”
for chunk_idx in $(seq 0 $(($num_chunks - 1)))
do
    cat output/initial/data_${chunk_idx}.jsonl >> "$output_file"
done
```


We collect all the samples in `initial_datas.json`.



### Construction of Positive and Negative Examples
After obtaining the single-round data above, we can try to introduce the retrieved content when answering again to enhance the context, thereby constructing our positive and negative sample pairs. We use the concatenated retrieved content to prompt the model re-generate the response:


```bash
total_chunks=50
for i in $(seq 0 `expr $total_chunks - 1`)
    do
    echo "Running on chunk $i ..."
    log_file="SURf/logs/initial/context_${i}.log"
    nohup srun -p MoE -n1 -N1 --gres=gpu:1 --quotatype=reserved python initial/generate_context_data.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder database/data \
    --question-file initial/initial_datas.json \
    --answers-file output/initial/reanswer_data_${i}.jsonl \
    --num-chunks $total_chunks \
    --chunk-idx $i \
    1>"$log_file" 2>&1 &
    sleep 2
done
```

Same as above, we strongly recommend using multi-card batch inference because the amount of data is large. And after that, we can combine the sub-file in the following bash:

```bash
# execute after generating the initial datas
output_file = “output/context_datas.jsonl”
for chunk_idx in $(seq 0 $(($num_chunks - 1)))
do
    cat output/initial/reanswer_data_${chunk_idx}.jsonl >> "$output_file"
done
```

Next, we need to use tools to evaluate whether the current spliced ​​search content can help the model answer questions better or worse:

```bash
python initial/tool_evaluate.py \
    --input-file output/context_datas.jsonl \
    --output-file output/training_data.jsonl
```

### Data Filtering

We first need to remove all data that only has positive and negative samples, and then select the positive and negative samples with the largest retrieval scores from all the data as the samples we will eventually train:


The final data will look like this:


```bash
{
        "id": "BNKpPRzoweDRTxvaxgkwVQ",
        "image": "test_image",
        "conversations": [
            {
                "from": "human",
                "value": "<Retrieval>1. <image> Caption1 \n2. <image> Caption 2 \n</Retrieval> <image>\nQuestion"
            },
            {
                "from": "gpt",
                "value": "Answer"
            }
        ],
        "images": [
            "image1",
            "image2"
        ],
        "labels": [
            1,
            0
        ]
    },
```


### RAG Instruction-Tuning

After get the json data, you can use the following command in `training.sh`:

```bash
torchrun --nnodes 1 --nproc_per_node 8 llava/train/train_mem.py \
    --deepspeed config/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version retrieval \
    --data_path data/training_datas.json \
    --image_folder database/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/surf-7b \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
```
We use 8 A100-80g to train for 0.5 hour, and your model will be saved at `SURf/checkpoints`.

## Evaluation

Our evaluation script is the same as llava, and we use pope-MSCOCO as the evaluation example:

```bash
python eval/pope.py \
    --model-path checkpoints/surf-7b \
    --image-folder data/ \
    --retrieve-image-folder database/data\ 
    --question-file benchmark/POPE/coco/coco_pope_popular.json \
    --answers-file output/coco_pope_popular_results.jsonl \
    --large-data database/sharegpt4v_1246k.json \
    --index-path database/sharegpt4v_1246k_index.json \
    --image-names-path database/sharegpt4v_1246k_file_names.json \
    --clip-model-name ViT-L-14
```

After getting the result, you can directly eval the result:

```bash
python eval/eval_pope.py \
    --gt_files benchmark/POPE/coco/coco_pope_popular.json \
    --gen_files output/coco_pope_popular_results.jsonl
```




## Experiments

The following figure shows our main experiment：
![image](https://img.picui.cn/free/2024/06/17/666ff36d84360.png)



## Claims
This project uses the Apache 2.0 protocol. The project assumes no legal responsibility for any of the model's output and will not be held liable for any damages that may result from the use of the resources and output.
