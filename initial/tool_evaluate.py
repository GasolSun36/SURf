import json
import copy
import math
from tqdm import tqdm
from bert_score import BERTScorer
import argparse

def exact_match(res, answer):
    res = res.lower().strip()
    answer = answer.lower().strip()
    return res in answer or answer in res

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def main(args):
    scorer = BERTScorer(model_type='bert-base-uncased')

    with open(args.input_file, "r") as json_file:
        total_datas = json.load(json_file)

    datas = []
    for item in tqdm(total_datas):
        question = item['conversations'][0]['value']
        answer = item['conversations'][1]['value']
        responses = item['re_answers']
        original_bert_score = item['bert_score']
        labels = []

        if 'provide a short description for' in question or 'one-sentence caption' in question:
            _, _, original_bert_score = scorer.score([answer], [item['first_response']])
            for response in responses:
                _, _, f1 = scorer.score([answer], [response])
                new_bert_score = float(f1[0])
                labels.append(1 if new_bert_score > original_bert_score else 0)

        elif 'letter from the given' in question or 'a single word or phrase' in question:
            for response in responses:
                labels.append(1 if exact_match(response, answer) else 0)

        else:
            _, _, original_bert_score = scorer.score([answer], [item['first_response']])
            for response in responses:
                _, _, f1 = scorer.score([answer], [response])
                new_bert_score = float(f1[0])
                labels.append(1 if new_bert_score > original_bert_score else 0)

        new_item = copy.copy(item)
        new_item['labels'] = labels
        datas.append(new_item)


    with open(args.output_file, 'w', encoding='utf-8') as file:
        json.dump(datas, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output JSON file")
    args = parser.parse_args()
    main(args)
