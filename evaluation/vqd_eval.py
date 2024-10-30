import argparse
import os
import base64
import requests
import json
from tqdm import tqdm  # Import tqdm for the progress bar

from ..utils.data_utils import COCO_IMAGE_TRAINING_SET_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--pred_path", required=True, help="path to prediction output file.")
    parser.add_argument("--api_key", required=True, help="path to prediction output file.")
    parser.add_argument("--gpt_eval_path", required=True, help="path to evaluation file by GPT.")
    args = parser.parse_args()
    return args


# Function to encode the image to Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


if __name__ == "__main__":
    args = parse_args()
    # Load the JSON file
    with open(args.pred_path, 'r') as file:
        data = json.load(file)

    # Extract the required information for each selected key
    requests_data = []
    for key in data:
        item = data[key]
        image_id_formatted = str(item['image_id']).zfill(12)
        image_path = os.path.join(COCO_IMAGE_TRAINING_SET_PATH, image_id_formatted + ".jpg")
        base64_image = encode_image(image_path)
        prompt_text = f"You have 3 tasks:\nEvaluate the following texts based on the given image.\nIf a sub-question is irrelevant to the main question and does not help in answering it at all (e.g., the main question is asking about relationship between two person sitting at the table, but the sub-questions are meaninglessly asking about colors of their clothes, or shapes of the table), classify it B. Otherwise, classify it G.\nIf a sub-question is a repetition of the main question or any existing sub-questions (repetition means the sub-question repeats exactly the same content or discusses the same topic in a different form), classify it R. Otherwise, classify it U.\nIf the answer to a sub-question can be derived from the image through direct observation, basic knowledge, logical inference, or reasonable assumptions, classify it Y. Otherwise, if the sub-question requires information that is not available in the image, classify it N.\n In conclusion, I want 3 classes for each subquestion, G/B, Y/N, U/R. Attention:1. Do not repeat the subquestion or give explanation, just give me the 3 classes.\n2. If you can not find any subquesion under the answer of a main question, cases can be either the answers are not presented in a subquestion form or the subquestion is incomplete, classify it E.\nHere is the main question:{item['question']}\nHere is the subquestion:{item['model_answer']}"

        requests_data.append({
            'question_id': key,
            'image_base64': base64_image,
            'prompt_text': prompt_text
        })

    # Headers for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}"
    }

    # Send requests and collect responses
    responses = []
    for request_data in tqdm(requests_data, desc="Processing requests"):
        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": request_data['prompt_text']
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{request_data['image_base64']}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        responses.append(response.json())

    # Save the responses to a JSON file
    with open(args.gpt_eval_path, 'w') as outfile:
        json.dump(responses, outfile, indent=4)

    print("Responses saved")