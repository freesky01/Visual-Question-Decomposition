import argparse
import json
import re
import time
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm
from openai import OpenAI
from nltk import word_tokenize

CHAT_GPT = 'gpt-3.5-turbo'
GPT4 = 'gpt-4'


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--pred_path", required=True, help="path to prediction output file.")
    parser.add_argument("--api_key", required=True, help="path to prediction output file.")
    parser.add_argument("--submission_form_path", required=True, help="path to prediction output file.")
    args = parser.parse_args()
    return args


def generate_response(client, prompt, engine):
    if engine == CHAT_GPT or engine == GPT4:
        response = client.chat.completions.create(
                messages=[{'role': 'user', 'content': prompt}],
                model=engine,
                temperature=0.7,
                top_p=1.0,
                max_tokens=600,
            )
    else:
        response = client.completions.create(
                model=engine,
                prompt=prompt,
                temperature=0.7,
                top_p=1.0,
                max_tokens=600,
            )

    return response


def GPT4_eval(client, question, model_answer, prompt_path):
    engine = GPT4
    prompt_template = open(prompt_path, 'r', encoding='utf-8').read().strip()
    prompt_text = prompt_template.replace('{question}', question).replace('{model_answer}', model_answer)
        
    response = generate_response(client, prompt_text, engine)
    time.sleep(4)
    
    if engine == CHAT_GPT or engine == GPT4:
        feedback = response.choices[0].message
        feedback = json.loads(feedback.json())['content']
    else:
        feedback = response.choices[0]['text']

    return feedback 


def str_lower(input_str):   # lower of input str, except for the index
    words = input_str.split()

    transformed_words = []
    if len(input_str) <= 2:
        if input_str.endswith(')') and input_str[0].isalpha() and input_str[0].islower():
            transformed_str = input_str.upper()
        elif len(input_str) == 1 and input_str[0].isalpha() and input_str[0].islower():
            transformed_str = input_str.upper()
        else:
            transformed_str = input_str
        return transformed_str
    else:
        for word in words:
            if word.endswith(')'):                      # do not change the upper case of index
                transformed_words.append(word)
            else:
                transformed_words.append(word.lower())

        # reconstruct the string
        transformed_str = ' '.join(transformed_words)
        return transformed_str


def check_answer(question_id, ori_answer):
    lower_answer = str_lower(ori_answer)
    options = ['A', 'B', 'C', 'D']
    found_options = []
    for option in options:
        if option in lower_answer:
            found_options.append(option)
    if len(found_options) == 0 or len(found_options) > 1:
        print(f'Error! ori_answer: {ori_answer}, lower_answer: {lower_answer}, found_options: {found_options}. Question ID: {question_id}')
        model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')
        device = 'cuda'
        model.to(device)
        with open('../data/inference/VQA/aokvqa_test_set.json', 'r') as f:
            aokvqa_data = json.load(f)
        for sample in aokvqa_data:
            if question_id == sample['question_id']:
                choices = sample['choices']
                break
        choice_embeddings = model.encode([lower_answer] + choices, convert_to_tensor=True)
        a_idx = cos_sim(choice_embeddings[0], choice_embeddings[1:]).argmax().item()
        found_option = chr(65+a_idx)
        return found_option
    else:
        return found_options[0]


def get_answer(client, question_id, ori_answer):
    with open('../data/inference/VQA/aokvqa_test_set.json', 'r') as f:
        aokvqa_data = json.load(f)
        
    answer_no_index = re.sub(r'[a-zA-Z]+\) ', '', ori_answer)
    for sample in aokvqa_data:
        if question_id == sample['question_id']: 
            choices = sample['choices']
            if len(word_tokenize(ori_answer)) > 20: # if the answer is too long, SentenceTransformer is not so accurate to map to the correct option, then use GPT4 to map
                choices_with_options = ['' for _ in range(len(choices))]
                for i in range(len(choices)):
                    index_char = chr(65+i)
                    choices_with_options[i] = f'{index_char}) {choices[i]}'
                choices_str = ', '.join(choices_with_options)
                options_str = f'Choose one option from {choices_str}'
                ori_question = sample['question']
                question = f'{ori_question} {options_str}'
                print('Original answer is too long, use GPT4 to map')
                print(question)
                ori_answer = GPT4_eval(client, question, ori_answer, 'prompt_template_map.txt')
                
            
            for option in choices:
                if answer_no_index == option:
                    remapped_answer = answer_no_index
                    return remapped_answer
            
            model_option = check_answer(question_id, ori_answer)
            index_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            index = index_map[model_option]
            remapped_answer = choices[index]
            break
    return remapped_answer


if __name__ == '__main__':
    args = parse_args()
    with open(args.pred_path, 'r') as f:
        data = json.load(f)
    
    client = OpenAI(api_key=args.api_key)
    output_data = {}

    for question_id, dict_info in tqdm(data.items()):
        if 'model_answer_direct' in dict_info:
            model_answer = dict_info['model_answer_direct']
        else:
            model_answer = dict_info['model_answer_third_phase']
        
        
        output_data[question_id] = {}
        output_data[question_id]['multiple_choice'] = get_answer(client, question_id, model_answer)
        output_data[question_id]['direct_answer'] = "answer"

    json.dump(output_data, open(args.submission_form_path, 'w'), indent=4)