import torch
import torchvision.transforms as T
import json
import torch

from typing import Optional
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from ..utils.prompt import PROMPT_DICT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def run_internvl(model_path, data, dataset_name, pred_path, seed: Optional[int] = None):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto').eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    generation_config = dict(
        num_beams=1,
        max_new_tokens=512,
        temperature=1.0,
    )

    output_dict = {}
    
    for question_id, sample_info in tqdm(data.items()):
        image_path = sample_info['image_path']
        question = sample_info['question']
        outputs_list = []
    
        pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).to(model.device)
        
        if dataset_name != 'subquestrater':
            # zero stage - selective stage
            zero_inp = PROMPT_DICT['whether_decompose'].format(question=question)
            zero_phase_response, history = model.chat(tokenizer, pixel_values, zero_inp, generation_config, history=None, return_history=True)
            
            output_dict[question_id] = {}
            output_dict[question_id]['question'] = question
            output_dict[question_id]['image_path'] = image_path
            if 'ref_answer' in sample_info:
                output_dict[question_id]['ref_answer'] = sample_info['ref_answer']
            output_dict[question_id]['model_answer_zero_phase'] = zero_phase_response

            if dataset_name == 'whether2deco':
                continue
            else:
                # 1st to 3rd answering stage with decomposition
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
                    response, history = model.chat(tokenizer, pixel_values, inp, generation_config, history=history, return_history=True)
                    outputs_list.append(response)
                
                if DIRECT_FLAG:
                    output_dict[question_id]['model_answer_direct'] = outputs_list[0]
                else:
                    output_dict[question_id]['model_answer_first_phase'] = outputs_list[0]
                    output_dict[question_id]['model_answer_second_phase'] = outputs_list[1]
                    output_dict[question_id]['model_answer_third_phase'] = outputs_list[2]
        else:
            prompt = PROMPT_DICT['decompose_first'].format(question=question)
            subquestions, history = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
            
            output_dict[question_id] = {}
            output_dict[question_id]['question'] = question
            output_dict[question_id]['image_path'] = image_path
            output_dict[question_id]['model_subquestions'] = subquestions
            
    json.dump(output_dict, open(pred_path, 'w'), indent=4)