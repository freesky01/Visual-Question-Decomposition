import argparse
from data_utils import get_data
from inference import run_minigptv2, run_llava, run_qwen, run_internvl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True, type=str, default=None, help="model path")
    parser.add_argument("--dataset", required=True, type=str, default=None, help="can be chosen from ['aokvqa', 'gqa', 'vqaintrospect', 'whether2deco']")
    parser.add_argument("--pred_path", required=True, type=str, default=None, help="pred path")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    args = parser.parse_args()
    
    
    data = get_data(args.dataset)
    model_path = args.model_path.lower()
    if 'minigpt' in model_path:
        run_minigptv2(args.model_path, data, args.pred_path, args.seed)
    elif 'llava' in model_path:
        run_llava(args.model_path, data, args.pred_path, args.seed)
    elif 'qwen' in model_path:
        run_qwen(args.model_path, data, args.pred_path, args.seed)
    elif 'internvl' in model_path:
        run_internvl(args.model_path, data, args.pred_path, args.seed)
    else:
        raise ValueError(f'Unknown model name: {model_path}')