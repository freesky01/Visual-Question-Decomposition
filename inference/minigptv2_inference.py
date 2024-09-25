import json
import os
import random
from collections import defaultdict
from tqdm import tqdm

import cv2
import random
import re

import numpy as np
from PIL import Image
import torch

import torchvision.transforms as T
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from minigpt4.common.config import Config

from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


device = 'cuda:{}'.format(args.gpu_id)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)

model = model_cls.from_config(model_config).to(device)

bounding_box_size = 100

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


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (210, 210, 0),
    (255, 0, 255),
    (0, 255, 255),
    (114, 128, 250),
    (0, 165, 255),
    (0, 128, 0),
    (144, 238, 144),
    (238, 238, 175),
    (255, 191, 0),
    (0, 128, 0),
    (226, 43, 138),
    (255, 0, 255),
    (0, 215, 255),
]

color_map = {
    f"{color_id}": f"#{hex(color[2])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[0])[2:].zfill(2)}" for
    color_id, color in enumerate(colors)
}

used_colors = colors


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


chat = Chat(model, vis_processor, device=device)
#chat = Chat(model, vis_processor, device='cuda')

title = """<h1 align="center">MiniGPT-v2 Demo</h1>"""
description = 'Welcome to Our MiniGPT-v2 Chatbot Demo!'
# article = """<p><a href='https://minigpt-v2.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p><a href='https://github.com/Vision-CAIR/MiniGPT-4/blob/main/MiniGPTv2.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p><p><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/GitHub-Repo-blue'></a></p><p><a href='https://www.youtube.com/watch?v=atFCwV2hSY4'><img src='https://img.shields.io/badge/YouTube-Video-red'></a></p>"""
article = """<p><a href='https://minigpt-v2.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p>"""

introduction = '''
For Abilities Involving Visual Grounding:
1. Grounding: CLICK **Send** to generate a grounded image description.
2. Refer: Input a referring object and CLICK **Send**.
3. Detection: Write a caption or phrase, and CLICK **Send**.
4. Identify: Draw the bounding box on the uploaded image window and CLICK **Send** to generate the bounding box. (CLICK "clear" button before re-drawing next time).
5. VQA: Input a visual question and CLICK **Send**.
6. No Tag: Input whatever you want and CLICK **Send** without any tagging

You can also simply chat in free form!
'''


def run_minigptv2():
    if args.seed is not None:
        setup_seed(args.seed)
        print(f"Seed: {args.seed}")
        
    if 'aokvqa' in args.dataset:
        data = get_data()
    elif 'gqa' in args.dataset:
        data = get_data_gqa()
    elif 'vqaintrospect' in args.dataset:
        data = get_data_vqaintrospect(sample=True)
    elif 'vqa' in args.dataset:
        data = get_data_vqa()
    else:
        raise ValueError('Dataset not supported!')
    mode = args.mode
    num_beams = 1
    temperature = 1.0
    chat_state = CONV_VISION.copy()
    img_list = []
    
    num = 0
    output_dict = {}
    for question_id, image_info in tqdm(data.items()):
        image_id = image_info['image_id']
        zero_str = '0' * (12 - len(str(image_id)))
        if 'aokvqa' in args.dataset or 'vqaintrospect' in args.dataset or 'vqa' in args.dataset:
            image_path = os.path.join(args.data_path, f'{zero_str}{image_id}.jpg')
            if not os.path.exists(image_path):
                image_path = os.path.join('../val2017', f'{zero_str}{image_id}.jpg')
        else:
            image_path = os.path.join(args.data_path, f'{image_id}.jpg')
        print(image_path)
        raw_image = Image.open(image_path).convert('RGB')

        """
        transform = transforms.Compose([
                transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                ])
        image = transform(raw_image).unsqueeze(0).to(model.device)
        """
        upload_image = chat.upload_img(raw_image, chat_state, img_list)
        encode_img = chat.encode_img(img_list)
        
        #ori_question = image_info['ori_question']
        upload_question = image_info['question']
        if mode == 'direct':
            #user_message = 'Please answer the following question:\n' + upload_question
            if 'aokvqa' in args.dataset:
                user_message = f'Please answer the following question, your answer should mention both the option letter and the word:\n{upload_question}'
            else:
                user_message = f'Please answer the following question:\n {upload_question}'
            print(user_message)
        elif mode == 'decompose_one_phase':
            user_message = f'Question: {upload_question}\nPlease firstly decompose the given question into several image-relevant sub-questions, and then answer the subquestions. Finally give me the answer of the original question.'
        elif mode == 'decompose_two_phase':
            user_message = f'Question: {upload_question}\nPlease firstly decompose the given question into several image-relevant sub-questions, and then answer the sub-questions.'
        elif mode == 'decompose_three_phase':
            user_message = f"Question: {upload_question}\nPlease firstly decompose the given question into several image-relevant sub-questions to help you answer the given question. Please avoid giving repeated subquestions or generating an excessive number. Feel free to suggest an appropriate quantity based on your judgment."
        elif mode == 'revise':
            with open(args.direct_path, 'r') as f:
                direct_data = json.load(f)
            model_direct_answer = direct_data[question_id]['model_answer']
            user_message = f"""
            You will be provided with a question and the answer of a student. Your task is to judge the student answer to the given question. If you think the student has answered the question correctly, please type "true". If you think the student has answered the question incorrectly, please type "false", then give your answer to the given question. I will give you the response format, please follow the format when answering the question.
            Response Format:
            Judgement: true/false.
            Your answer:
            
            Question: {upload_question}
            Student Answer: {model_direct_answer}
            """
            #user_message = f"Question: {upload_question}\nPlease firstly decompose the given question into several (at most 5) image-relevant sub-questions to help you answer the given question."
        elif mode == 'selective' or mode == 'ICL':
            user_message = f'Question: {upload_question}\nDecide if the question can be directly answered, or the question should be decomposed into sub-questions for easier answering. If the question can be directly answered, please answer \"Yes.\" If the question should be decomposed for easier answering, please answer \"No.\"'
        
        
        chat.ask(user_message, chat_state)

        llm_message = chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    num_beams=num_beams,
                                    temperature=temperature,
                                    max_new_tokens=300
                                    )[0]
        
        if mode == 'selective' or mode == 'ICL':
            direct_sym = True
            if 'yes' in llm_message.lower():
                if 'aokvqa' in args.dataset:
                    user_message_direct = f'Please answer the following question, your answer should mention both the option letter and the word:\n{upload_question}'
                else:
                    user_message_direct = f'Please answer the following question:\n{upload_question}'
                chat.ask(user_message_direct, chat_state)
                llm_message_direct = chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    num_beams=num_beams,
                                    temperature=temperature,
                                    max_new_tokens=300
                                    )[0]
            else:
                direct_sym = False
                if mode == 'selective':
                    user_message_first = f"Question: {upload_question}\nPlease firstly decompose the given question into several image-relevant sub-questions to help you answer the given question. Please avoid giving repeated subquestions or generating an excessive number. Feel free to suggest an appropriate quantity based on your judgment."
                else:
                    with open('/nfs/data3/zhen/haowei/MiniGPT-4_v2/prompts/ICL.txt', 'r') as f:
                        prompt_template = f.read()
                    user_message_first = prompt_template.replace('{question}', upload_question)
                    #print(user_message_first)
                chat.ask(user_message_first, chat_state)
                llm_message_first = chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    num_beams=num_beams,
                                    temperature=temperature,
                                    max_new_tokens=300
                                    )[0]
                
                user_message_second = f'Please answer each of the sub-questions raised by yourself in the previous step.'
                chat.ask(user_message_second, chat_state)
                llm_message_second = chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    num_beams=num_beams,
                                    temperature=temperature,
                                    max_new_tokens=300
                                    )[0]
                if 'aokvqa' in args.dataset:
                    user_message_third = f'With the help of the already answered sub-questions, please answer the original question, your should mention both the option letter and the word:\n{upload_question}'
                else:
                    user_message_third = f"With the help of the already answered sub-questions, please answer the original question:\n{upload_question}"
                chat.ask(user_message_third, chat_state)
                llm_message_third = chat.answer(conv=chat_state,
                                        img_list=img_list,
                                        num_beams=num_beams,
                                        temperature=temperature,
                                        max_new_tokens=300,
                                        )[0]
        
        
        if mode == 'decompose_two_phase':
            user_message = f'With the help of the already answered sub-questions, please answer the original question: {upload_question}'
            chat.ask(user_message, chat_state)
            llm_message_two_phase = chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    num_beams=num_beams,
                                    temperature=temperature,
                                    max_new_tokens=300,
                                    )[0]
        elif mode == 'decompose_three_phase':
            user_message = f'Please answer each of the sub-questions raised by yourself in the previous step.'
            chat.ask(user_message, chat_state)
            llm_message_two_phase = chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    num_beams=num_beams,
                                    temperature=temperature,
                                    max_new_tokens=300,
                                    )[0]
            
            if 'aokvqa' in args.dataset:
                user_message = f'With the help of the already answered sub-questions, please answer the original question, your should mention both the option letter and the word:\n{upload_question}'
            else:
                user_message = f"With the help of the already answered sub-questions, please answer the original question:\n{upload_question}"
            chat.ask(user_message, chat_state)
            llm_message_three_phase = chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    num_beams=num_beams,
                                    temperature=temperature,
                                    max_new_tokens=300,
                                    )[0]
            
        output_dict[question_id] = {}
        output_dict[question_id]['question'] = upload_question
        if args.dataset != 'aokvqa_test':
            output_dict[question_id]['ref_answer'] = image_info['ref_answer']
        if mode == 'direct' or mode == 'decompose_one_phase':
            output_dict[question_id]['model_answer'] = llm_message.replace('</s>', '')
            #output_dict[question_id]['max_prob_confidence'] = max_prob_confidence
        elif mode == 'revise':
            output_dict[question_id]['model_answer'] = model_direct_answer
            output_dict[question_id]['judgement'] = llm_message.replace('</s>', '')
            #output_dict[question_id]['confidence'] = llm_message_confident
            #output_dict[question_id]['max_prob_confidence'] = max_prob_confidence
        elif mode == 'decompose_two_phase':
            output_dict[question_id]['model_answer_first_phase'] = llm_message.replace('</s>', '')
            output_dict[question_id]['model_answer_second_phase'] = llm_message_two_phase.replace('</s>', '')
        elif mode == 'decompose_three_phase':
            output_dict[question_id]['model_answer_first_phase'] = llm_message.replace('</s>', '')
            output_dict[question_id]['model_answer_second_phase'] = llm_message_two_phase.replace('</s>', '')
            output_dict[question_id]['model_answer_third_phase'] = llm_message_three_phase.replace('</s>', '')
        elif mode == 'selective' or mode == 'ICL':
            output_dict[question_id]['model_answer_zero_phase'] = llm_message.replace('</s>', '')
            if direct_sym:
                output_dict[question_id]['model_answer_direct'] = llm_message_direct.replace('</s>', '')
            else:
                output_dict[question_id]['model_answer_first_phase'] = llm_message_first.replace('</s>', '')
                output_dict[question_id]['model_answer_second_phase'] = llm_message_second.replace('</s>', '')
                output_dict[question_id]['model_answer_third_phase'] = llm_message_third.replace('</s>', '')
                
        chat_state.messages = []
        img_list = []
        num += 1
        
    json.dump(output_dict, open(args.output_path, 'w'), indent=4)


if __name__ == "__main__":
    run_minigpt4_v2()