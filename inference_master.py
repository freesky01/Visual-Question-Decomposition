import argparse
from data_utils import get_data
from inference import run_minigptv2, run_llava, run_qwen, run_internvl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True, type=str, default=None, help="model path")
    parser.add_argument("--dataset", required=True, type=str, default=None, choices=['aokvqa', 'gqa', 'vqaintrospect', 'whether2deco'], help="can be chosen from ['aokvqa', 'gqa', 'vqaintrospect', 'whether2deco']")
    parser.add_argument("--pred_path", required=True, type=str, default=None, help="pred path")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    args = parser.parse_args()
    
    
    data = get_data(args.dataset)
    print(f'Dataset: {args.dataset}')
    print(f'Number of samples: {len(data.keys())}')
    print(f'Prediction path: {args.pred_path}')
    model_path = args.model_path.lower()
    
    if 'minigpt' in model_path:
        print('Model: MiniGPT-v2')
        run_minigptv2(args.model_path, data, args.dataset, args.pred_path, args.seed)
        
    elif 'llava' in model_path:
        print('Model: Llava-1.5')
        run_llava(args.model_path, data, args.dataset, args.pred_path, args.seed)
        
    elif 'qwen' in model_path:
        print('Model: Qwen-VL-Chat')
        run_qwen(args.model_path, data, args.dataset, args.pred_path, args.seed)
        
    elif 'internvl' in model_path:
        print('Model: InternVL-Chat-V1-5')
        run_internvl(args.model_path, data, args.dataset, args.pred_path, args.seed)
    else:
        raise ValueError(f'Unknown model name: {model_path}')