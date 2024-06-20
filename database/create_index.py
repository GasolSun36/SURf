import os
import faiss
import numpy as np
import torch
from PIL import Image
import open_clip
from tqdm import tqdm
import time
import json
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate CLIP vectors and create a FAISS index.')
    parser.add_argument('--model_name', type=str, default='ViT-L-14', help='Model name to use for CLIP.')
    parser.add_argument('--pretrained', type=str, default='openai', help='Pretrained model source.')
    parser.add_argument('--data_file', type=str, default='sharegpt4v_1246k.json', help='JSON file containing image data.')
    parser.add_argument('--output_vectors', type=str, default='sharegpt4v_1246k_file_names.npy', help='Output file for image names.')
    parser.add_argument('--output_index', type=str, default='sharegpt4v_1246k.index', help='Output file for FAISS index.')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing images.')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for processing images.')
    return parser.parse_args()

def load_model_and_preprocess(model_name, pretrained):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, force_quick_gelu=True)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer

def generate_clip_vectors(images, model, preprocess, folder_path, batch_size):
    vectors = []
    batch = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for img in tqdm(images):
            image = preprocess(Image.open(os.path.join(folder_path, img))).unsqueeze(0)
            batch.append(image)

            if len(batch) == batch_size or img == images[-1]:
                batch = torch.cat(batch, dim=0)
                image_features = model.encode_image(batch)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                vectors.append(image_features.cpu())
                batch = []

    return np.vstack(vectors)

def create_faiss_index(vectors):
    d = vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    return index

def main():
    args = parse_arguments()

    model, preprocess, tokenizer = load_model_and_preprocess(args.model_name, args.pretrained)

    with open(args.data_file, 'r') as file:
        sharegpt4v_data = json.load(file)

    images = [data['image'] for data in sharegpt4v_data]
    np.save(args.output_vectors, np.array(images))

    print("Load images complete....")

    start_time = time.time()
    vectors = generate_clip_vectors(images, model, preprocess, args.folder_path, args.batch_size)
    end_time = time.time()

    index = create_faiss_index(vectors)
    faiss.write_index(index, args.output_index)

    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
