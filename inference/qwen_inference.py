import json
import os

from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from typing import Optional

from ..utils.prompt import PROMPT_DICT

def run_qwen(model_path, data, dataset_name, pred_path, seed: Optional[int] = None):
    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    # use bf16
    model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, bf16=True).eval()

    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True, temperature=1.0, num_beams=1)

    output_dict = {}
    
    #if 'if_decompose' not in args.dataset:
    for question_id, sample_info in tqdm(data.items()):
        image_path = sample_info['image_path']
        question = sample_info['question']
        outputs_list = []
    
        # zero stage - selective stage
        zero_inp = PROMPT_DICT['whether_decompose'].format(question=question)
        query = tokenizer.from_list_format([
        {'image': image_path}, # Either a local path or an url
        {'text': zero_inp},
        ])
        zero_phase_response, history = model.chat(tokenizer, query=query, history=None)
        
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
                response, history = model.chat(tokenizer, query=inp, history=history)
                outputs_list.append(response)
    
            if DIRECT_FLAG:
                output_dict[question_id]['model_answer_direct'] = outputs_list[0]
            else:
                output_dict[question_id]['model_answer_first_phase'] = outputs_list[0]
                output_dict[question_id]['model_answer_second_phase'] = outputs_list[1]
                output_dict[question_id]['model_answer_third_phase'] = outputs_list[2]
   
    json.dump(output_dict, open(pred_path, 'w'), indent=4)