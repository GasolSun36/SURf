import os
import json
import argparse

def load_files(file_path):
    with open(os.path.expanduser(file_path), "r") as f:
        return [json.loads(q) for q in f]

def calculate_metrics(gt_files, gen_files):
    true_pos, true_neg, false_pos, false_neg, unknown, yes_answers = 0, 0, 0, 0, 0, 0
    for index, line in enumerate(gt_files):
        idx = line["question_id"]
        gt_answer = line["label"].lower().strip()
        gen_answer = gen_files[index]["text"].lower().strip()
        if idx != gen_files[index]["question_id"]:
            continue
        if gt_answer == 'yes':
            if 'yes' in gen_answer:
                true_pos += 1
                yes_answers += 1
            else:
                false_neg += 1
        elif gt_answer == 'no':
            if 'no' in gen_answer:
                true_neg += 1
            else:
                false_pos += 1
                yes_answers += 1
        else:
            unknown += 1
    return true_pos, true_neg, false_pos, false_neg, yes_answers, unknown, len(gt_files)

def compute_statistics(true_pos, true_neg, false_pos, false_neg, yes_answers, unknown, total_questions):
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / total_questions
    yes_proportion = yes_answers / total_questions
    unknown_prop = unknown / total_questions
    return precision, recall, f1, accuracy, yes_proportion, unknown_prop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_files", type=str, default="")
    parser.add_argument("--gen_files", type=str, default="")
    args = parser.parse_args()
    
    gt_files = load_files(args.gt_files)
    gen_files = load_files(args.gen_files)
    
    true_pos, true_neg, false_pos, false_neg, yes_answers, unknown, total_questions = calculate_metrics(gt_files, gen_files)
    precision, recall, f1, accuracy, yes_proportion, unknown_prop = compute_statistics(true_pos, true_neg, false_pos, false_neg, yes_answers, unknown, total_questions)
    
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    print(f'yes: {yes_proportion}')
    print(f'unknown: {unknown_prop}')

if __name__ == "__main__":
    main()
