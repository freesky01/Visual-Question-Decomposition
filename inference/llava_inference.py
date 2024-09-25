import argparse
import json
import torch
import requests
import os
import zipfile
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from tqdm import tqdm
import numpy as np
import random

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def sample_data(data_dict, sample_num, seed=42):
    random.seed(seed)
    sample_dict = {}
    
    question_ids = list(data_dict.keys())
    sample_ids = random.sample(question_ids, sample_num)
    for question_id, image_info in data_dict.items():
        if question_id in sample_ids:
            sample_dict[question_id] = image_info
    return sample_dict


def get_data():
    if args.dataset == 'aokvqa_train':
        with open('../aokvqa/aokvqa_v1p0_train.json', 'r') as f:
            data = json.load(f)
    elif args.dataset == 'aokvqa_val':
        with open('../aokvqa/aokvqa_v1p0_val.json', 'r') as f:
            data = json.load(f)
    elif args.dataset == 'aokvqa_test':
        with open('../aokvqa/aokvqa_v1p0_test.json', 'r') as f:
            data = json.load(f)

    output_dict = {}
    for dict_info in data:
        image_id = dict_info['image_id']
        question_id = dict_info['question_id']
        ori_question = dict_info['question']
        choices = dict_info['choices']
        
        for i in range(len(choices)):
            index_char = chr(65+i)
            choices[i] = f'{index_char}) {choices[i]}'
        choices_str = ', '.join(choices)
        question = f'{ori_question} Choose one option from {choices_str}'

        output_dict[question_id] = {}
        output_dict[question_id]['ori_question'] = ori_question
        output_dict[question_id]['question'] = question
        output_dict[question_id]['image_id'] = image_id
        
        if args.dataset != 'aokvqa_test':
            correct_choice_idx = dict_info['correct_choice_idx']
            ref_answer = choices[correct_choice_idx]
            output_dict[question_id]['ref_answer'] = ref_answer

    return output_dict


def get_data_gqa():
    if args.dataset == 'gqa_train':
        with open('../gqa/balanced_train_data.json', 'r') as f:
            data = json.load(f)
    elif args.dataset == 'gqa_val':
        with open('../gqa/balanced_val_data.json', 'r') as f:
            data = json.load(f)
    elif args.dataset == 'gqa_test':
        with open('../gqa/balanced_testdev_data.json', 'r') as f:
            data = json.load(f)

    output_dict = {}
    for dict_info in data['questions']:
        image_id = dict_info['imageId']
        question_id = dict_info['questionId']
        question = dict_info['question']
        ref_answer = dict_info['answer']

        output_dict[question_id] = {}
        output_dict[question_id]['question'] = question
        output_dict[question_id]['ref_answer'] = ref_answer
        output_dict[question_id]['image_id'] = image_id

    return output_dict

"""
def get_data_vqaintrospect(sample=True):
    if args.dataset == 'vqaintrospect_val':
        with open('../VQA-Introspect/VQAIntrospect_valv1.0.json', 'r') as f:
            data = json.load(f)
    if sample:
        data = sample_data(data, 3000)
    output_dict = {}
    for question_id, dict_info in data.items():
        question = dict_info['reasoning_question']
        ref_answer = dict_info['reasoning_answer_most_common']
        image_id = dict_info['image_id']

        output_dict[question_id] = {}
        output_dict[question_id]['question'] = question
        output_dict[question_id]['ref_answer'] = ref_answer
        output_dict[question_id]['image_id'] = image_id

    return output_dict
"""
def get_data_vqaintrospect(sample=True):
    if args.dataset == 'vqaintrospect_val':
        with open('/nfs/data3/zhen/haowei/VQA-Introspect/VQAIntrospect_valv1.0.json', 'r') as f:
            data = json.load(f)
    if sample:
        with open('/nfs/data3/zhen/haowei/MiniGPT-4_v2/output/test/selective/minigptv2_selective_vqaintrospect_val_multi_loss_19_epochs_baseline3_greedy.json', 'r') as f:
            sampled_key_data = json.load(f)
        sampled_keys = sampled_key_data.keys()
        
    output_dict = {}
    for question_id, dict_info in data.items():
        if question_id in sampled_keys:
            question = dict_info['reasoning_question']
            ref_answer = dict_info['reasoning_answer_most_common']
            image_id = dict_info['image_id']

            output_dict[question_id] = {}
            output_dict[question_id]['question'] = question
            output_dict[question_id]['ref_answer'] = ref_answer
            output_dict[question_id]['image_id'] = image_id

    return output_dict

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def compute_seq_prob(output):
    seq_prob = torch.exp(output.sequences_scores[0])
    seq_prob = np.round(float(seq_prob.cpu().numpy()), 3)
    print(f'Sequence prob: {seq_prob}')
    return seq_prob


def ppl_compute(model, input_output, image_tensor, input_len, device):
    trg_len = input_len
    input_ids = input_output.to(device)
    target_ids = input_ids.clone()
    target_ids[:, :trg_len] = -100
    
    with torch.no_grad():
        compute_output = model(input_ids=input_ids, images=image_tensor, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = compute_output.loss

    ppl = torch.exp(neg_log_likelihood)
    
    ppl = np.round(ppl.item(), 3)
    print(f"Perplexity: {ppl}")
    return ppl


def run_llava(args):
    if args.seed is not None:
        setup_seed(args.seed)
        print(f"Seed: {args.seed}")
        
    if 'aokvqa' in args.dataset:
        data = get_data()
    elif 'gqa' in args.dataset:
        data = get_data_gqa()
        """
        if args.dataset == 'gqa_train':
            data = sample_images(data, 2000)
        elif args.dataset == 'gqa_test':
            data = sample_images(data, 1500)
        """
    elif 'vqaintrospect' in args.dataset:
        data = get_data_vqaintrospect()
    else:
        raise ValueError('Dataset not supported!')
    output_dict = {}
    num = 0
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    for question_id, image_info in tqdm(data.items()):
        image_id = image_info['image_id']
        zero_str = '0' * (12 - len(str(image_id)))
        if 'aokvqa' in args.dataset or 'vqaintrospect' in args.dataset:
            image_path = os.path.join(args.image_path, f'{zero_str}{image_id}.jpg')
            if not os.path.exists(image_path):
                image_path = os.path.join('../val2017', f'{zero_str}{image_id}.jpg')
        elif 'gqa' in args.dataset:
            image_path = os.path.join(args.image_path, f'{image_id}.jpg')
        upload_question = image_info['question']
        outputs_list = []

        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        image = load_image(image_path)
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        # TODO load question
        """
        if args.mode == 'direct':
            if 'aokvqa' in args.dataset:
                inp_list = [
                    'Please answer the following question, your answer should mention both the option letter and the word:\n' + upload_question
                ]
            elif 'gqa' in args.dataset or 'vqaintrospect' in args.dataset:
                inp_list = [
                    'Please answer the following question:\n' + upload_question
                ]
        elif args.mode == 'decompose':
            if 'aokvqa' in args.dataset:
                inp_list = [
                    f"Question: {upload_question}\nPlease firstly decompose the given question into several image-relevant sub-questions to help you answer the given question. Please avoid giving repeated subquestions or generating an excessive number. Feel free to suggest an appropriate quantity based on your judgment.",
                    "Please answer each of the above sub-questions raised by yourself.",
                    f"With the help of the already answered sub-questions, please answer the original question, you should both mention the option letter and the word:\n{upload_question}"
                ]
            elif 'gqa' in args.dataset or 'vqaintrospect' in args.dataset:
                inp_list = [
                    f"Question: {upload_question}\nPlease firstly decompose the given question into several image-relevant sub-questions to help you answer the given question. Please avoid giving repeated subquestions or generating an excessive number. Feel free to suggest an appropriate quantity based on your judgment.",
                    "Please answer each of the above sub-questions raised by yourself.",
                    f"With the help of the already answered sub-questions, please answer the original question:\n{upload_question}"
                ]
        """
        zero_inp = f'Question: {upload_question}\nDecide if the question can be directly answered, or the question should be decomposed into sub-questions for easier answering. If the question can be directly answered, please answer \"Yes.\" If the question should be decomposed for easier answering, please answer \"No.\"'
        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                zero_inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + zero_inp
            else:
                zero_inp = DEFAULT_IMAGE_TOKEN + '\n' + zero_inp
            conv.append_message(conv.roles[0], zero_inp)
            image = None
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        ppl_input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device).input_ids
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        #streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                min_length=1,
                length_penalty=1,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                output_scores=True,
                return_dict_in_generate=True
            )
            
        output_ids = outputs.sequences
        output_str_zero_phase = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = output_str_zero_phase
        
        direct_sym = True
        if 'yes' in output_str_zero_phase.lower():
            if 'aokvqa' in args.dataset:
                inp_list = [f'Please answer the following question, your answer should mention both the option letter and the word:\n{upload_question}']
            else:
                inp_list = [f'Please answer the following question:\n{upload_question}']
        else:
            direct_sym = False
            if 'aokvqa' in args.dataset:
                user_message_third = f'With the help of the already answered sub-questions, please answer the original question, your should mention both the option letter and the word:\n{upload_question}'
            else:
                user_message_third = f"With the help of the already answered sub-questions, please answer the original question:\n{upload_question}"
            inp_list = [f"Question: {upload_question}\nPlease firstly decompose the given question into several image-relevant sub-questions to help you answer the given question. Please avoid giving repeated subquestions or generating an excessive number. Feel free to suggest an appropriate quantity based on your judgment.",
                        'Please answer each of the sub-questions raised by yourself in the previous step.',
                        user_message_third
                ]

        
        for inp in inp_list:
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            ppl_input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device).input_ids
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            #streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    min_length=1,
                    length_penalty=1,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    output_scores=True,
                    return_dict_in_generate=True
                )
            output_ids = outputs.sequences
            output_str = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = output_str
            outputs_list.append(output_str)
            
            """
            input_len = ppl_input_ids.shape[1]

            ppl_tokens = torch.cat([ppl_input_ids, output_ids[:, input_ids.shape[1]:]], dim=1)
            ppl = ppl_compute(model, ppl_tokens, image_tensor, input_len, model.device)
            """
            
        output_dict[question_id] = {}
        output_dict[question_id]['question'] = upload_question
        #output_dict[question_id]['correct_answer'] = image_info['correct_answer']
        if args.dataset != 'aokvqa_test':
            output_dict[question_id]['ref_answer'] = image_info['ref_answer']
        output_dict[question_id]['model_answer_zero_phase'] = output_str_zero_phase.replace('</s>', '')
        if direct_sym:
            output_dict[question_id]['model_answer_direct'] = outputs_list[0].replace('</s>', '')
        else:
            output_dict[question_id]['model_answer_first_phase'] = outputs_list[0].replace('</s>', '')
            output_dict[question_id]['model_answer_second_phase'] = outputs_list[1].replace('</s>', '')
            output_dict[question_id]['model_answer_third_phase'] = outputs_list[2].replace('</s>', '')
        num += 1
    json.dump(output_dict, open(args.output_path, 'w'), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--mode", required=True, type=str, default=None, help="inference mode, can be chosen from ['direct', 'decompose_one_phase', 'decompose_two_phase'].")
    parser.add_argument("--dataset", required=True, type=str, default=None, help="can be chosen from ['train', 'val']")
    parser.add_argument("--image_path", required=True, type=str, help="image path")
    parser.add_argument("--output_path", required=True, type=str, default=None, help="output path")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    args = parser.parse_args()
    main(args)
