import json

#TODO: image_id -> image_path

def get_data_aokvqa(data_path):
	with open(data_path, 'r') as f:
		data = json.load(f)

	output_dict = {}
	for sample_info in data:
		image_id = sample_info['image_id']
		question_id = sample_info['question_id']
		orig_question = sample_info['question']     # original question
		choices = sample_info['choices']
		
		for i in range(len(choices)):
			index_char = chr(65+i)
			choices[i] = f'{index_char}) {choices[i]}'
		choices_str = ', '.join(choices)
		question = f'{orig_question} Choose one option from {choices_str}'

		output_dict[question_id] = {}
		output_dict[question_id]['question'] = question
		output_dict[question_id]['image_id'] = image_id

	return output_dict


def get_data_gqa(data_path):
	with open(data_path, 'r') as f:
		data = json.load(f)

	output_dict = {}
	for sample_info in data['questions']:
		question_id = sample_info['questionId']
		question = sample_info['question']
		image_id = sample_info['imageId']
		ref_answer = sample_info['answer']

		output_dict[question_id] = {}
		output_dict[question_id]['question'] = question
		output_dict[question_id]['image_id'] = image_id
		output_dict[question_id]['ref_answer'] = ref_answer

	return output_dict


def get_data_vqaintrospect(data_path):
	with open(data_path, 'r') as f:
		data = json.load(f)
				
	output_dict = {}
	for question_id, sample_info in data.items():
		question = sample_info['reasoning_question']
		image_id = sample_info['image_id']
		ref_answer = sample_info['reasoning_answer_most_common']

		output_dict[question_id] = {}
		output_dict[question_id]['question'] = question
		output_dict[question_id]['image_id'] = image_id
		output_dict[question_id]['ref_answer'] = ref_answer

	return output_dict


def get_data_whether2deco(data_path):
	with open(data_path, 'r') as f:
		data = json.load(f)
				
	output_dict = {}
	for sample_info in data():
		question_id = sample_info['id']
		question = sample_info['question']
		image_id = sample_info['image']
		ref_answer = sample_info['ref_answer']

		output_dict[question_id] = {}
		output_dict[question_id]['question'] = question
		output_dict[question_id]['image_id'] = image_id
		output_dict[question_id]['ref_answer'] = ref_answer

	return output_dict


def get_data(dataset_name, data_path):
	if dataset_name == 'aokvqa':
		return get_data_aokvqa(data_path)
	elif dataset_name == 'gqa':
		return get_data_gqa(data_path)
	elif dataset_name == 'vqaintrospect':
		return get_data_vqaintrospect(data_path)
	elif dataset_name == 'whether2deco':
		return get_data_whether2deco(data_path)
	else:
		raise ValueError(f'Unknown dataset_name: {dataset_name}')