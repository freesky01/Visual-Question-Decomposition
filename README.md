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

## Fintuning
If you would like to finetune models on your own, please follow [the finetuning instruction](finetuning/finetuning_instruction.md).
You can also directly download our finetuned checkpoint for these four MLLMs.

### MiniGPT-v2
| MiniGPT-v2 (finetuned by DecoVQA) | MiniGPT-v2 (finetuned by DecoVQA+) | MiniGPT-v2 (finetuned by DecoVQA+ with SelectiveVQDLoss)| 
|------------------------------|------------------------------|------------------------------|
| [Download](https://drive.google.com/file/d/1d3Q_czHF5L5jQ2zKGYjgNt-iyGpcU1Ea/view?usp=drive_link) |[Download](https://drive.google.com/file/d/1NGGtHG8NcfopmPjvlEoFrQyaScfBbvNP/view?usp=drive_link) | [Download](https://drive.google.com/file/d/1D2zHfpSzpn5F4i9DYTjH3Urc-8mAafPm/view?usp=drive_link) |

### LLaVA-1.5
| LLaVA-1.5 (finetuned by DecoVQA) | LLaVA-1.5 (finetuned by DecoVQA+) | LLaVA-1.5 (finetuned by DecoVQA+ with SelectiveVQDLoss)| 
|------------------------------|------------------------------|------------------------------|
| [Download](https://drive.google.com/drive/folders/1f2huCzA8IAq-v6fAvOnWWjLZEF7uxidO?usp=drive_link) |[Download](https://drive.google.com/drive/folders/1jPL48N9kEjZNXy1cZ2GZr5CvplZTf27D?usp=drive_link) | [Download](https://drive.google.com/drive/folders/1A5ZPe6dUwMTghExwwwxwzREtIBm28L7A?usp=drive_link) |

### Qwen-VL-Chat
| Qwen-VL-Chat (finetuned by DecoVQA) | Qwen-VL-Chat (finetuned by DecoVQA+) | Qwen-VL-Chat (finetuned by DecoVQA+ with SelectiveVQDLoss)| 
|------------------------------|------------------------------|------------------------------|
| [Download](https://drive.google.com/drive/folders/1z7fug8NU_yV9hyMcWfzQpXtka73x6cCC?usp=drive_link) |[Download](https://drive.google.com/drive/folders/1X47IpulDUvsZPYKD1aAxkQ79VXBPPUcn?usp=drive_link) | [Download](https://drive.google.com/drive/folders/1Dz2o-HgePR-o2-W67TtzSVdsPOI188a6?usp=drive_link) |

### InternVL-Chat-V1-5
| InternVL-Chat-V1-5 (finetuned by DecoVQA) | InternVL-Chat-V1-5 (finetuned by DecoVQA+) | InternVL-Chat-V1-5 (finetuned by DecoVQA+ with SelectiveVQDLoss)| 
|------------------------------|------------------------------|------------------------------|
| [Download](https://huggingface.co/freesky/InternVL-Chat-V1-5_ft_by_DecoVQA) |[Download](https://huggingface.co/freesky/InternVL-Chat-V1-5_ft_by_DecoVQAplus) | [Download](https://huggingface.co/freesky/InternVL-Chat-V1-5_ft_by_DecoVQAplus_SelectiveLoss) |

## Inference
### 1. VQD Task

### 2. VQA Task

### 3. Whether2Deco Task

## Evaluation
### 1. Quality of Sub-questions: SubQuestRater Evaluation Framework

### 2. VQA Accuracy

### 3. Whether2Deco Accuracy


## Contact
For any issue or question, kindly contact us per E-Mail: Haowei Zhang (haowei.zhang@tum.de)

## Citation

If you find this work useful, please consider giving this repository a star and citing our paper:

```
example paper
```
