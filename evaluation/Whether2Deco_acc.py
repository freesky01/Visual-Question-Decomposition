import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--pred_path", required=True, help="path to prediction output file.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    pred_file_name = args.pred_path.split('/')[-1]
    with open(args.pred_path, 'r') as f:
        data = json.load(f)

    correct = 0
    for sample in data:
        ref_answer = sample['ref_answer']
        model_answer = sample['model_answer']
        
        if ref_answer == 'Yes.':
            if 'yes' in model_answer.lower():
                correct += 1
        else:
            if 'no' in model_answer.lower():
                correct += 1
    print(f'Prediction File: {pred_file_name}')
    print(f'Whether2Deco Accuracy: {correct/len(data)}')