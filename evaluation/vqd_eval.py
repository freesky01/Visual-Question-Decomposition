import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from matplotlib.backends.backend_pdf import PdfPages  # Import PdfPages

# Download the punkt data package
nltk.download('punkt')


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--pred_path", required=True, help="path to prediction output file.")
    parser.add_argument("--eval_output_path", required=True, help="path to evaluation output file.")
    args = parser.parse_args()
    return args


def compute_scores(data):
    results = []
    for sample in data:
        if 'choices' not in sample or not sample['choices']:
            results.append({"G_score": 0, "Y_score": 0, "U_score": 0})
            continue

        try:
            content = sample["choices"][0]["message"]["content"]
        except KeyError:
            results.append({"G_score": 0, "Y_score": 0, "U_score": 0})
            continue

        if 'E' in content:
            results.append({"G_score": 0, "Y_score": 0, "U_score": 0})
            continue

        B_count = content.count('B')
        G_count = content.count('G')
        Y_count = content.count('Y')
        N_count = content.count('N')
        U_count = content.count('U')
        R_count = content.count('R')

        G_score = (G_count / (B_count + G_count)) * 100 if (B_count + G_count) > 0 else 0
        Y_score = (Y_count / (Y_count + N_count)) * 100 if (Y_count + N_count) > 0 else 0
        U_score = (U_count / (U_count + R_count)) * 100 if (U_count + R_count) > 0 else 0

        results.append({"G_score": G_score, "Y_score": Y_score, "U_score": U_score})

    return results


def process_files(file_paths):
    all_results = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            if not isinstance(data, list):
                print(f"Unexpected data format in {file_path}, expected a list of samples.")
                continue

            scores = compute_scores(data)
            df = pd.DataFrame(scores)
            average_scores = df.mean().to_dict()
            all_results.append(average_scores)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if len(all_results) != len(file_paths):
        print(f"Warning: Expected {len(file_paths)} results, but got {len(all_results)}.")

    return all_results


def plot_comparison(results, metric, metric_name, pdf_pages):
    labels = ['MiniGPT-v2', 'LLaVA-1.5', 'Qwen-VL-Chat', 'InternVL\n-Chat-V1-5']
    width = 0.35

    if len(results) != 8:
        print(f"Error: Expected 8 results, but got {len(results)}. Cannot plot comparison.")
        return

    data1 = [results[i][metric] for i in range(0, len(results), 2)]
    data2 = [results[i][metric] for i in range(1, len(results), 2)]

    x = np.arange(len(labels))

    fig, ax = plt.subplots()

    # Define RGB colors (range 0-255)
    color1_rgb = (31,119,180)  # cornflowerblue
    color2_rgb = (255,127,14)  # salmon

    # Convert RGB colors to 0-1 range
    color1 = tuple(c / 255 for c in color1_rgb)
    color2 = tuple(c / 255 for c in color2_rgb)

    bar1 = ax.bar(x - width / 2, data1, width, label='Original Model', color=color1)
    bar2 = ax.bar(x + width / 2, data2, width, label='Finetuned Model', color=color2)

    ax.set_ylabel('Scores')
    ax.set_title(f'Comparison on criterion {metric_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 120)  # Adjust maximum value as needed
    ax.set_yticks(np.arange(0, 120, 20))

    for bar in bar1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=13)

    for bar in bar2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=13)

    ax.set_ylabel('Scores', fontsize=15)
    ax.set_title(f'Comparison on criterion {metric_name}', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)

    ax.legend(fontsize=12)
    fig.tight_layout()

    # Save the current chart to the PDF file
    pdf_pages.savefig(fig)

    plt.close(fig)  # Close the current figure to avoid display


if __name__ == "__main__":
    args = parse_args()
    results = process_files([args.pred_path])
    metrics = {
        'G_score': 'Relevance',
        'Y_score': 'Groundedness',
        'U_score': 'Non-Repetition'
    }

    # Create a PDF file and save all charts
    with PdfPages(args.eval_output_path) as pdf_pages:
        for metric, metric_name in metrics.items():
            plot_comparison(results, metric, metric_name, pdf_pages)

    print(f"All comparison charts saved to {args.eval_output_path}.")
