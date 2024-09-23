# Visual-Question-Decomposition
Code for EMNLP 2024 Findings paper "Visual Question Decomposition on Multimodal Large Language Models"

<p align="center">
<img src="figures/subquestion_quality_comparison.png" width="100%">
</p>

## Installation
We do finetuning and inference with four open-source Multimodal Large Language Models (MLLMs). You can easily download the source code and finish the conda environment setup accordingly in the GitHub repository links as follows:
| Model | GitHub Link |
|-----------|---------|
| MiniGPT-v2 | [repo link](https://github.com/Vision-CAIR/MiniGPT-4) |
| LLaVA-1.5  | [repo link](https://github.com/haotian-liu/LLaVA) |
| Qwen-VL-Chat | [repo link](https://github.com/QwenLM/Qwen-VL) |
| InternVL-Chat-V1-5 | [repo link](https://github.com/OpenGVLab/InternVL) |

## Data Preparation
Both finetuning and inference require [COCO images](https://cocodataset.org/#download), we use the 2017 version. Unzip all the training, validation and test set under [data/images/COCO_images/](data/images/COCO_images/).

If you would like to do inference on GQA dataset, [GQA images](https://cs.stanford.edu/people/dorarad/gqa/download.html) are required additionally. Unzip images.zip under [data/images/GQA_images/](data/images/GQA_images/).


## Fintuning

Our proposed dataset [DecoVQA](data/finetuning/DecoVQA.json) and its upgraded version [DecoVQA+](data/finetuning/DecoVQA+.json) are used for finetuning, which contain high-quality human-annotated sub-questions. Run the finetuning script in the official repo of each MLLM accordingly with DecoVQA(+) as finetuning data.

DecoVQA(+) is provided in json format and each sample is a dictionary consisting of an id, an image path and a list of multi-round conversation. Here is an example sample from DecoVQA+:

```
{
    "id": "000000505343",
    "image": "finetuning_COCO_images/000000505343.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nGiven an image and a question: When is it safe to proceed going forward in a vehicle? Choose one option from A) 3 minutes, B) now, C) never, D) 10 minutes.\nDecide if the question can be directly answered, or the question should be decomposed into sub-questions for easier answering. If the question can be directly answered, please answer \"Yes.\" If the question should be decomposed for easier answering, please answer \"No.\""
        },
        {
            "from": "gpt",
            "value": "No."
        },
        {
            "from": "human",
            "value": "Question: When is it safe to proceed going forward in a vehicle? Choose one option from A) 3 minutes, B) now, C) never, D) 10 minutes.\nPlease firstly decompose the given question into several image-relevant sub-questions to help you answer the given question. Please avoid giving repeated subquestions or generating an excessive number. Feel free to suggest an appropriate quantity based on your judgment."
        },
        {
            "from": "gpt",
            "value": "1. What signs can we refer to in the picture to decide when to proceed?\n2. What does the color of the traffic light indicate?"
        },
        {
            "from": "human",
            "value": "Please answer each of the sub-questions raised by yourself in the previous step."
        },
        {
            "from": "gpt",
            "value": "1. We can refer to the traffic lights to decide when to proceed.\n2. It is safe to proceed immediately when the traffic light turns green."
        },
        {
            "from": "human",
            "value": "With the help of the already answered sub-questions, please answer the original question, you should both mention the option letter and the word:\nWhen is it safe to proceed going forward in a vehicle? Choose one option from A) 3 minutes, B) now, C) never, D) 10 minutes."
        },
        {
            "from": "gpt",
            "value": "B) now"
        }
    ]
}
```

If you would like to finetune models with SelectiveVQDLoss, please follow [instruction of finetuning with SelectiveVQDLoss](finetuning/finetuning_instruction_SelectiveVQDLoss.md).

You can also directly download our finetuned checkpoints of these four MLLMs in [Finetuned Model Zoo](finetuning/Finetuned_Model_Zoo.md).


## Inference
### 1. VQD Task

### 2. VQA Task

### 3. Whether2Deco Task



## Evaluation
### 1. Quality of Sub-questions: SubQuestRater Evaluation Framework

### 2. VQA Accuracy

### 3. Whether2Deco Accuracy
```
python ../evaluation/Whether2Deco/Whether2Deco_acc.py \
    --pred_path ../output/Whether2Deco/MiniGPT-v2/minigptv2_whether2deco_output.json
```

## Contact
For any issue or question, kindly contact us per E-Mail: Haowei Zhang (haowei.zhang@tum.de).

## Acknowledgement
The code is partly based on the following repos, thank so much to all the contributors!

- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
- [InternVL](https://github.com/OpenGVLab/InternVL)

## Citation

If you find this work useful, please consider giving this repository a star and citing our paper:

```
example paper
```
