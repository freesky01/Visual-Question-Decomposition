import argparse
import json
import random
import torch
import numpy as np
import os

from PIL import Image
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def run_qwen():
    # Note: The default behavior now has injection attack prevention off.
	tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

	# use bf16
	model = AutoPeftModelForCausalLM.from_pretrained(args.model_path, device_map="auto", trust_remote_code=True, bf16=True).eval()

	# Specify hyperparameters for generation
	model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True, temperature=1.0, num_beams=1)
	
	if args.seed is not None:
		setup_seed(args.seed)
		print(f"Seed: {args.seed}")
		
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
					image_path = os.path.join('../val2017', f'{zero_str}{image_id}.jpg')
			elif 'gqa' in args.dataset:
				image_path = os.path.join(args.image_path, f'{image_id}.jpg')
			upload_question = image_info['question']
			outputs_list = []
		
			# zero stage - selective stage
			zero_inp = f'Question: {upload_question}\nDecide if the question can be directly answered, or the question should be decomposed into sub-questions for easier answering. If the question can be directly answered, please answer \"Yes.\" If the question should be decomposed for easier answering, please answer \"No.\"'
			query = tokenizer.from_list_format([
			{'image': image_path}, # Either a local path or an url
			{'text': zero_inp},
			])
			zero_phase_response, history = model.chat(tokenizer, query=query, history=None)
			
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
				response, history = model.chat(tokenizer, query=inp, history=history)
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

			query = tokenizer.from_list_format([
			{'image': image_path}, # Either a local path or an url
			{'text': question},
			])
			
			response, history = model.chat(tokenizer, query=query, history=None)
			
			image_info['model_answer'] = response
			num += 1
			
		json.dump(data, open(args.output_path, 'w'), indent=4)
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--model_path", required=True, type=str, default=None, help="model path")
	parser.add_argument("--dataset", required=True, type=str, default=None, help="can be chosen from ['train', 'val']")
	parser.add_argument("--image_path", required=True, type=str, help="image path")
	parser.add_argument("--output_path", required=True, type=str, default=None, help="output path")
	parser.add_argument("--seed", type=int, default=None, help="seed")
	args = parser.parse_args()
    
	run_qwen(args)