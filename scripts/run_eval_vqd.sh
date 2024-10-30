python ../evaluation/vqd_eval.py \
    --pred_path example_subquestions.json \
    --api_key xxx \
    --gpt_eval_path example_gpt_eval.json

python ../evaluation/vqd_vis.py \
    --gpt_eval_path ../output/VQD/MiniGPT-v2/minigptv2_vqd_output.json \
    --eval_output_path example_eval_output.pdf