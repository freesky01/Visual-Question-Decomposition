import json
import os

COCO_IMAGE_TRAINING_SET_PATH = 'Visual-Question-Decomposition/data/images/COCO_images/train2017'
COCO_IMAGE_VAL_SET_PATH = 'Visual-Question-Decomposition/data/images/COCO_images/val2017'
COCO_IMAGE_TEST_SET_PATH = 'Visual-Question-Decomposition/data/images/COCO_images/test2017'
GQA_IMAGE_PATH = 'Visual-Question-Decomposition/data/images/GQA_images/'


def get_data_aokvqa(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    output_dict = {}
    for sample_info in data:
        image_id = sample_info['image_id']
        zero_str = '0' * (12 - len(str(image_id)))
        image_path = os.path.join(COCO_IMAGE_TEST_SET_PATH, f'{zero_str}{image_id}.jpg')
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
        output_dict[question_id]['image_path'] = image_path

    return output_dict


def get_data_gqa(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    output_dict = {}
    for sample_info in data['questions']:
        question_id = sample_info['questionId']
        question = sample_info['question']
        image_id = sample_info['imageId']
        image_path = os.path.join(GQA_IMAGE_PATH, f'{image_id}.jpg')
        ref_answer = sample_info['answer']

        output_dict[question_id] = {}
        output_dict[question_id]['question'] = question
        output_dict[question_id]['image_path'] = image_path
        output_dict[question_id]['ref_answer'] = ref_answer

    return output_dict


def get_data_vqaintrospect(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
                
    output_dict = {}
    for question_id, sample_info in data.items():
        question = sample_info['reasoning_question']
        image_id = sample_info['image_id']
        zero_str = '0' * (12 - len(str(image_id)))
        image_path = os.path.join(COCO_IMAGE_TRAINING_SET_PATH, f'{zero_str}{image_id}.jpg')
        if not os.path.exists(image_path):
            image_path = os.path.join(COCO_IMAGE_VAL_SET_PATH, f'{zero_str}{image_id}.jpg')
        ref_answer = sample_info['reasoning_answer_most_common']

        output_dict[question_id] = {}
        output_dict[question_id]['question'] = question
        output_dict[question_id]['image_path'] = image_path
        output_dict[question_id]['ref_answer'] = ref_answer

    return output_dict


def get_data_whether2deco(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
                
    output_dict = {}
    for sample_info in data():
        question_id = sample_info['id']
        question = sample_info['question']
        image_file = sample_info['image'].split('/')[-1]
        image_path = os.path.join(COCO_IMAGE_TRAINING_SET_PATH, image_file)
        ref_answer = sample_info['ref_answer']

        output_dict[question_id] = {}
        output_dict[question_id]['question'] = question
        output_dict[question_id]['image_path'] = image_path
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