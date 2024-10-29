import argparse
import copy
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--pred_path", required=True, help="path to prediction output file.")
    args = parser.parse_args()
    return args


def clean_data(answer):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(answer)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_answer = ' '.join(lemmatized_tokens)
    cleaned_answer = lemmatized_answer.lower()
    return cleaned_answer


def gqa_acc_compute(pred_path):
    file_name = pred_path.split('.json')[0].split('/')[-1]
    with open(pred_path, 'r') as f:
        data = json.load(f)

    output_data = copy.deepcopy(data)
    num = 0
    right = 0
    direct_answer = 0
    direct_correct_answer = 0
    for question_id, dict_info in data.items():
        ref_answer = dict_info['ref_answer']
        if 'model_answer_direct' in dict_info:
            model_answer = dict_info['model_answer_direct']
            direct_answer += 1
        else:
            model_answer = dict_info['model_answer_third_phase']
        
        cleaned_ref_answer = clean_data(ref_answer)
        cleaned_model_answer = clean_data(model_answer)

        if 'not possible' in cleaned_model_answer or 'impossible' in cleaned_model_answer:
            output_data[question_id]['eval'] = 0
        else:
            if cleaned_ref_answer in cleaned_model_answer:
                right += 1
                output_data[question_id]['eval'] = 1
                if 'model_answer_direct' in dict_info:
                    direct_correct_answer += 1
            else:
                output_data[question_id]['eval'] = 0
        num += 1

    acc = np.round(right / num, 3)
    print(num)
    print(f"Acc: {acc}")
    
    
if __name__ == '__main__':
    args = parse_args()
    gqa_acc_compute(args.pred_path)

    