import argparse
import json
import numpy as np
import random
import re
import torch

from PIL import Image
from tqdm import tqdm
from typing import Optional

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from ..utils.prompt import PROMPT_DICT
from ..utils.setup_seed import setup_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", default='minigptv2_eval.yaml', help="path to configuration file.")
    args = parser.parse_args()
    return args


def extract_substrings(string):
    # first check if there is no-finished bracket
    index = string.rfind('}')
    if index != -1:
        string = string[:index + 1]

    pattern = r'<p>(.*?)\}(?!<)'
    matches = re.findall(pattern, string)
    substrings = [match for match in matches]

    return substrings


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou


def save_tmp_img(visual_img):
    file_name = "".join([str(random.randint(0, 9)) for _ in range(5)]) + ".jpg"
    file_path = "/tmp/gradio" + file_name
    visual_img.save(file_path)
    return file_path


def mask2bbox(mask):
    if mask is None:
        return ''
    mask = mask.resize([100, 100], resample=Image.NEAREST)
    mask = np.array(mask)[:, :, 0]

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.sum():
        # Get the top, bottom, left, and right boundaries
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = '{{<{}><{}><{}><{}>}}'.format(cmin, rmin, cmax, rmax)
    else:
        bbox = ''

    return bbox


def escape_markdown(text):
    # List of Markdown special characters that need to be escaped
    md_chars = ['<', '>']

    # Escape each special character
    for char in md_chars:
        text = text.replace(char, '\\' + char)

    return text


def reverse_escape(text):
    md_chars = ['\\<', '\\>']

    for char in md_chars:
        text = text.replace(char, char[1:])

    return text


def run_minigptv2(model_path, data, dataset_name, pred_path, seed: Optional[int] = None):
    if seed is not None:
        setup_seed(seed)
        print(f"Seed: {seed}")
    
    args = parse_args()
    cfg = Config(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_config = cfg.model_cfg
    model_config.ckpt = model_path
    model_config.device_8bit = 1
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    model = model.eval()

    CONV_VISION = Conversation(
        system="",
        roles=(r"<s>[INST] ", r" [/INST]"),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="",
    )
    chat = Chat(model, vis_processor, device=device)
    
    num_beams = 1
    temperature = 1.0
    chat_state = CONV_VISION.copy()
    img_list = []
    
    output_dict = {}
    for question_id, sample_info in tqdm(data.items()):
        image_path = sample_info['image_path']
        raw_image = Image.open(image_path).convert('RGB')
        upload_image = chat.upload_img(raw_image, chat_state, img_list)
        encode_img = chat.encode_img(img_list)
        question = sample_info['question']
        outputs_list = []
        
        zero_inp = PROMPT_DICT['whether_decompose'].format(question=question)
        chat.ask(zero_inp, chat_state)

        zero_phase_response = chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    num_beams=num_beams,
                                    temperature=temperature,
                                    max_new_tokens=300
                                    )[0]
        output_dict[question_id] = {}
        output_dict[question_id]['question'] = question
        output_dict[question_id]['image_path'] = image_path
        if 'ref_answer' in sample_info:
            output_dict[question_id]['ref_answer'] = sample_info['ref_answer']
        output_dict[question_id]['model_answer_zero_phase'] = zero_phase_response

        if dataset_name == 'whether2deco':
            chat_state.messages = []
            img_list = []
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
                chat.ask(inp, chat_state)
                response = chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    num_beams=num_beams,
                                    temperature=temperature,
                                    max_new_tokens=300
                                    )[0]
                outputs_list.append(response)
            
            if DIRECT_FLAG:
                output_dict[question_id]['model_answer_direct'] = outputs_list[0]
            else:
                output_dict[question_id]['model_answer_first_phase'] = outputs_list[0]
                output_dict[question_id]['model_answer_second_phase'] = outputs_list[1]
                output_dict[question_id]['model_answer_third_phase'] = outputs_list[2]    
            chat_state.messages = []
            img_list = []
        
    json.dump(output_dict, open(pred_path, 'w'), indent=4)