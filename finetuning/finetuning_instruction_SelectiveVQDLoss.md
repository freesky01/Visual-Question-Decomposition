# Finetuning Instruction with SelectiveVQDLoss

To finetune MLLMs with our proposed SelectiveVQDLoss instead of standard Next-token Prediction (NTP) Loss, you should replace the original finetuning scripts with our provided scripts.

## MiniGPT-v2
Replace [minigpt_base.py](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/models/minigpt_base.py) with [minigpt_base_SelectiveVQDLoss.py](minigpt_base_SelectiveVQDLoss.py).

## LLaVA-1.5
Replace [llava_trainer.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py) with [llava_finetune_SelectiveVQDLoss.py](llava_finetune_SelectiveVQDLoss.py).

## Qwen-VL-Chat
Replace [finetune.py](https://github.com/QwenLM/Qwen-VL/blob/master/finetune.py) with [qwen_vl_finetune_SelectiveVQDLoss.py](qwen_vl_finetune_SelectiveVQDLoss.py).

## InternVL-Chat-V1-5
Replace [internvl_chat_finetune.py](https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/train/internvl_chat_finetune.py) with [internvl_finetune_SelectiveVQDLoss.py](internvl_finetune_SelectiveVQDLoss.py).