import torch
import torchvision.transforms as T
import argparse
import json
import random
import torch
import numpy as np
import os

from typing import Optional
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from transformers.generation import GenerationConfig
from PIL import Image
from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


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


def run_internvl(model_path, data, pred_path, seed: Optional[int] = None):
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

	if seed is not None:
		setup_seed(seed)
		print(f"Seed: {seed}")
		
	if 'aokvqa' in args.dataset:
		data = get_data()
	elif 'gqa' in args.dataset:
		data = get_data_gqa()
	elif 'vqaintrospect' in args.dataset:
		data = get_data_vqaintrospect()
	elif 'if_decompose' in args.dataset:
		data = get_data_if_decompose()
	else:
		raise ValueError('Dataset not supported!')
	
	output_dict = {}
	num = 0
	
	if 'if_decompose' not in args.dataset:
		for question_id, image_info in tqdm(data.items()):
			image_id = image_info['image_id']
			zero_str = '0' * (12 - len(str(image_id)))
			if 'aokvqa' in args.dataset or 'vqaintrospect' in args.dataset:
				image_path = os.path.join(args.image_path, f'{zero_str}{image_id}.jpg')
				if not os.path.exists(image_path):
					image_path = os.path.join(args.image_path.replace('train', 'val'), f'{zero_str}{image_id}.jpg')
			elif 'gqa' in args.dataset:
				image_path = os.path.join(args.image_path, f'{image_id}.jpg')
			upload_question = image_info['question']
			outputs_list = []
		
			pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).to(model.device)
			# zero stage - selective stage
			zero_inp = f'Question: {upload_question}\nDecide if the question can be directly answered, or the question should be decomposed into sub-questions for easier answering. If the question can be directly answered, please answer \"Yes.\" If the question should be decomposed for easier answering, please answer \"No.\"'
			zero_phase_response, history = model.chat(tokenizer, pixel_values, zero_inp, generation_config, history=None, return_history=True)
			
			# 1st to 3rd dialogue turns - direct answering or three-phase VQD
			direct_sym = True
			if 'yes' in zero_phase_response.lower():
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
				inp_list = [
						f"Question: {upload_question}\nPlease firstly decompose the given question into several image-relevant sub-questions to help you answer the given question. Please avoid giving repeated subquestions or generating an excessive number. Feel free to suggest an appropriate quantity based on your judgment.",
						'Please answer each of the sub-questions raised by yourself in the previous step.',
						user_message_third
					]
			
			for inp in inp_list:
				response, history = model.chat(tokenizer, pixel_values, inp, generation_config, history=history, return_history=True)
				outputs_list.append(response)

			
			output_dict[question_id] = {}
			output_dict[question_id]['question'] = upload_question
			#output_dict[question_id]['correct_answer'] = image_info['correct_answer']
			if args.dataset != 'aokvqa_test':
				output_dict[question_id]['ref_answer'] = image_info['ref_answer']
			output_dict[question_id]['model_answer_zero_phase'] = zero_phase_response
			if direct_sym:
				output_dict[question_id]['model_answer_direct'] = outputs_list[0]
			else:
				output_dict[question_id]['model_answer_first_phase'] = outputs_list[0]
				output_dict[question_id]['model_answer_second_phase'] = outputs_list[1]
				output_dict[question_id]['model_answer_third_phase'] = outputs_list[2]
			num += 1
		json.dump(output_dict, open(args.output_path, 'w'), indent=4)

	else:
		for image_info in tqdm(data):
			image_path = image_info['image']
			if not os.path.exists(image_path):
				image_path = image_path.replace('train2017', 'val2017')
			question = image_info['question']

			pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).to(model.device)
			
			response, history = model.chat(tokenizer, pixel_values, zero_inp, generation_config, history=None, return_history=True)
			
			image_info['model_answer'] = response
			num += 1
		
		json.dump(data, open(args.output_path, 'w'), indent=4)