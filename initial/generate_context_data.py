import argparse
import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from tqdm import tqdm
import faiss
import open_clip
import sys
from transformers import set_seed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../llava")

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)


def load_faiss_index(index_path):
    return faiss.read_index(index_path)


def get_chunk(lst, n, k):
    chunk_size = -(-len(lst) // n)
    return lst[k * chunk_size:(k + 1) * chunk_size]


def build_prompt(captions, default_image_token):
    return "".join(f"{i+1}. {default_image_token} {caption}\n" for i, caption in enumerate(captions))


def load_images(image_files, image_folder, retrieve_image_folder):
    images = []
    for i, image_file in enumerate(image_files):
        folder = retrieve_image_folder if i != len(image_files) - 1 else image_folder
        image = Image.open(os.path.join(folder, image_file)).convert("RGB")
        images.append(image)
    return images


def generate_from_image(img, model, preprocess):
    with torch.no_grad(), torch.cuda.amp.autocast():
        image = preprocess(Image.open(img)).unsqueeze(0)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True).detach()
    return image_features.cpu().numpy()


def search_index(query_vector, index, image_names, top_k=5):
    _, indices = index.search(query_vector, top_k)
    filenames = [image_names[i] for i in indices[0]]
    return _, filenames


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    with open(args.question_file, 'r', encoding='utf-8') as file:
        questions = json.load(file)
    
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["id"]
        conversations = line['conversations']
        image_file = line['image']
        question = conversations[0]['value'].replace("<image>\n", "").replace("\n<image>", "")
        cur_prompt = question
        
        query_vector = generate_from_image(os.path.join(args.image_folder, image_file), clip_model, clip_preprocess)
        scores, filenames = search_index(query_vector, index, image_names, args.topk + 1)
        scores = scores[0]

        if float(scores[0]) < 0.0001:
            scores = scores[1:]
            filenames = filenames[1:]
        else:
            scores = scores[:args.topk]
            filenames = filenames[:args.topk]
        
        for i, filename in enumerate(filenames):
            input_prompt = "Here is a image similar to the test image:\n{}\nBased on this image and the test image:\n{}\nAnswer the question: {}".format(
                DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN, question)
            
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], input_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            image_files = [filename] + [image_file]

            images = load_images(image_files, args.image_folder, args.retrieve_image_folder)
            images_tensor = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].half().cuda()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images_tensor,
                    do_sample=True,
                    temperature=0.4,
                    max_new_tokens=100,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            outputs = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            new_line = copy.copy(line)
            new_line.update({
                're_answer': outputs,
                'total_prompt': input_prompt,
                'retrieved_image': filename,
                'retrieved_score': float(scores[i]),
            })
            ans_file.write(json.dumps(new_line) + "\n")
            ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--retrieve-image-folder", type=str, default="database/data")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--large-data", type=str, required=True)
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--image-names-path", type=str, required=True)
    parser.add_argument("--clip-model-name", type=str, default='ViT-L-14')
    args = parser.parse_args()

    set_seed(args.seed)

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        args.clip_model_name, pretrained='openai', force_quick_gelu=True
    )
    clip_tokenizer = open_clip.get_tokenizer(args.clip_model_name)

    index = load_faiss_index(args.index_path)
    image_names = np.load(args.image_names_path)

    with open(args.large_data, 'r') as json_file:
        large_data = json.load(json_file)
    
    eval_model(args, large_data)
