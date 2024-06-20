import json
import argparse
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../llava')
from llava.constants import DEFAULT_IMAGE_TOKEN



def build_prompt(captions, default_image_token):
    prompt = "\n".join([f"{i+1}. {default_image_token} {caption}" for i, caption in enumerate(captions)])
    return prompt


def retrieve_base_filenames(filenames, large_data_dict):
    captions = [large_data_dict[filename] for filename in filenames if filename in large_data_dict]
    assert len(captions) == len(filenames)
    return captions


def filter(input_file, output_file, num_samples, large_data_dict):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = []

    for item in data:
        conversations = item['conversations']
        labels = item.get("labels", [])
        if len(set(labels)) == 1:
            continue

        positive_samples = [(score, idx) for idx, (score, label) in enumerate(zip(item["retrieved_scores"], labels)) if label == 1]
        negative_samples = [(score, idx) for idx, (score, label) in enumerate(zip(item["retrieved_scores"], labels)) if label == 0]

        if positive_samples and negative_samples:
            _, min_pos_idx = min(positive_samples)
            _, min_neg_idx = min(negative_samples)

            captions = retrieve_base_filenames([item["retrieved_images"][min_pos_idx], item["retrieved_images"][min_neg_idx]], large_data_dict)
            captions_prompt = build_prompt(captions, DEFAULT_IMAGE_TOKEN)

            pre_prompt = '<Retrieval> ' + captions_prompt + ' </Retrieval>\n'
            conversations[1]['value'] = pre_prompt + conversations[1]['value']

            filtered_item = {
                "id": item["id"],
                "image": item["image"],
                "conversations": item["conversations"],
                "images": [
                    item["retrieved_images"][min_pos_idx],
                    item["retrieved_images"][min_neg_idx]
                ],
                "labels": [
                    1,
                    0
                ]
            }
            filtered_data.append(filtered_item)

    random.shuffle(filtered_data)
    filtered_data = filtered_data[:num_samples]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_file', type=str, help='')
    parser.add_argument('--output_file', type=str, help='')
    parser.add_argument("--image-folder", type=str, default="", help='')
    parser.add_argument('--samples', type=int, help='', default=2000)
    parser.add_argument('--topk', type=int, default=5, help='')
    parser.add_argument('--index_path', type=str, required=True, help='')
    parser.add_argument('--image_names', type=str, required=True, help='')
    parser.add_argument('--large_data', type=str, required=True, help='')
    args = parser.parse_args()

    with open(args.large_data, "r") as json_file:
        large_data = json.load(json_file)
    large_data_dict = {data['image']: data['caption'] for data in large_data}

    filter(args.input_file, args.output_file, args.samples, large_data_dict)


if __name__ == '__main__':
    main()
