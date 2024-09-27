import argparse
import json
import torch
import requests
import os

from PIL import Image
from io import BytesIO
from tqdm import tqdm
from typing import Optional

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from ..utils.prompt import PROMPT_DICT


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def run_llava(model_path, data, dataset_name, pred_path, seed: Optional[int] = None):    
    output_dict = {}
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    
    model_base = None
    if model_path != "liuhaotian/llava-v1.5-13b":
        model_base = "liuhaotian/llava-v1.5-13b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    for question_id, sample_info in tqdm(data.items()):
        image_path = sample_info['image_path']
        question = sample_info['question']
        outputs_list = []

        conv = conv_templates[conv_mode].copy()
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

        zero_inp = PROMPT_DICT['whether_decompose'].format(question=question)
        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                zero_inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + zero_inp
            else:
                zero_inp = DEFAULT_IMAGE_TOKEN + '\n' + zero_inp
            conv.append_message(conv.roles[0], zero_inp)
            image = None
        conv.append_message(conv.roles[1], None)
        zero_phase_prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(zero_phase_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=512,
                num_beams=1,
                min_length=1,
                length_penalty=1,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                output_scores=True,
                return_dict_in_generate=True
            )
            
        output_ids = outputs.sequences
        zero_phase_response = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = zero_phase_response
        
        output_dict[question_id] = {}
        output_dict[question_id]['question'] = question
        output_dict[question_id]['image_path'] = image_path
        if 'ref_answer' in sample_info:
            output_dict[question_id]['ref_answer'] = sample_info['ref_answer']
        output_dict[question_id]['model_answer_zero_phase'] = zero_phase_response
        
        if dataset_name == 'whether2deco':
            continue
        else:
            # 1st to 3rd dialogue turns - direct answering or three-phase VQD
            DIRECT_FLAG = True
            if 'yes' in zero_phase_response.lower():
                if dataset_name == 'aokvqa':
                    inp_list = [PROMPT_DICT['direct_aokvqa'].format(question=question)]
                else:
                    inp_list = [PROMPT_DICT['direct_general'].format(question=question)]
            else:
                DIRECT_FLAG = False
                if dataset_name == 'aokvqa':
                    user_message_third = PROMPT_DICT['decompose_third_aokvqa'].format(question=question)
                else:
                    user_message_third = PROMPT_DICT['decompose_third_general'].format(question=question)
                inp_list = [
                        PROMPT_DICT['decompose_first'].format(question=question),
                        PROMPT_DICT['decompose_second'],
                        user_message_third
                    ]

        for inp in inp_list:
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=1.0,
                    num_beams=1,
                    max_new_tokens=512,
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
            
        if DIRECT_FLAG:
            output_dict[question_id]['model_answer_direct'] = outputs_list[0].replace('</s>', '')
        else:
            output_dict[question_id]['model_answer_first_phase'] = outputs_list[0].replace('</s>', '')
            output_dict[question_id]['model_answer_second_phase'] = outputs_list[1].replace('</s>', '')
            output_dict[question_id]['model_answer_third_phase'] = outputs_list[2].replace('</s>', '')
    json.dump(output_dict, open(pred_path, 'w'), indent=4)