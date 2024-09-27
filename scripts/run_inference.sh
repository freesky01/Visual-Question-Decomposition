export CUDA_VISIBLE_DEVICES=0
python ../inference_master.py \
    --model_path example_model.pt \
    --dataset aokvqa \
    --pred_path example_output.json \
    --seed 42