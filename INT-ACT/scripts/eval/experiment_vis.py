import argparse
import glob
import math
import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns


def parse_success_rates(log_path):
    task_success = {}
    current_task = None

    with open(log_path, 'r') as f:
        for line in f:
            task_match = re.search(r'Task suite: (.+)', line)
            if task_match:
                current_task = task_match.group(1).strip()

            success_match = re.search(r'Success rate: ([\d.]+)%', line)
            if success_match and current_task:
                task_success[current_task] = float(success_match.group(1))
                current_task = None  # reset after reading success rate

    return task_success

def find_latest_timestamp_dir(path):
    dirs = glob.glob(os.path.join(path, '*'))
    dirs = [d for d in dirs if os.path.isdir(d)]
    dirs.sort(key=lambda p: datetime.strptime(os.path.basename(p), "%Y-%m-%d_%H-%M-%S"), reverse=True)
    return dirs[0] if dirs else None

def collect_data(root_dir, model_names, seed, skip_steps):
    data = defaultdict(lambda: defaultdict(dict))  # {task: {model: {step: success_rate}}}

    for model in model_names:
        model_dir = os.path.join(root_dir, model)

        step_dirs = glob.glob(os.path.join(model_dir, "step_*"))
        for step_dir in step_dirs:
            step_number_match = re.search(r'step_(\d+)', step_dir)
            if step_number_match:
                step_number = int(step_number_match.group(1))

                if step_number in skip_steps:
                    print(f"Skipping step {step_number} for model {model} as specified.")
                    continue

                seed_dir = os.path.join(step_dir, str(seed))
                if not os.path.exists(seed_dir):
                    print(f"Seed directory '{seed_dir}' does not exist, skipping...")
                    continue

                timestamp_dir = find_latest_timestamp_dir(seed_dir)
                if not timestamp_dir:
                    print(f"No timestamp directories found in seed directory '{seed_dir}', skipping...")
                    continue

                log_file = os.path.join(timestamp_dir, "eval_simpler.log")
                if not os.path.isfile(log_file):
                    print(f"No log file '{log_file}' found, skipping...")
                    continue

                task_success_rates = parse_success_rates(log_file)
                for task_name, success_rate in task_success_rates.items():
                    data[task_name][model][step_number] = success_rate

    return data

def plot_data(data, output_dir="plots"):
    sns.set(style="whitegrid", font_scale=1.2)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # First, individual task plots (the same as before)
    for task, model_data in data.items():
        plt.figure(figsize=(8,6))
        for model, steps_dict in model_data.items():
            steps_sorted = sorted(steps_dict.keys())
            rates_sorted = [steps_dict[step] for step in steps_sorted]
            plt.plot(steps_sorted, rates_sorted, marker='o', label=model)

        plt.title(f"Task: {task}")
        plt.xlabel("Step")
        plt.ylabel("Success Rate (%)")
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{task.replace(' ', '_')}_success_rate.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved for task '{task}' at '{output_path}'.")

    # Now the tiled plot for all tasks in a single figure
    num_tasks = len(data)
    cols = min(2, num_tasks)  # 1 or 2 columns
    rows = math.ceil(num_tasks / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))

    if num_tasks == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (task, model_data) in enumerate(data.items()):
        ax = axes[idx]
        for model, steps_dict in model_data.items():
            steps_sorted = sorted(steps_dict.keys())
            rates_sorted = [steps_dict[step] for step in steps_sorted]
            ax.plot(steps_sorted, rates_sorted, marker='o', label=model)

        ax.set_title(f"Task: {task}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Success Rate (%)")
        ax.legend()

    # Remove empty subplots
    for j in range(len(data), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    tiled_output_path = os.path.join(output_dir, "all_tasks_success_rate.png")
    plt.savefig(tiled_output_path)
    plt.close()
    print(f"Tiled plot for all tasks saved at '{tiled_output_path}'.")

def main():
    parser = argparse.ArgumentParser(description="Visualize experimental data.")
    parser.add_argument('--log_dir', type=str, default='log/eval_online/simpler',
                        help='Root directory containing model log folders.')
    parser.add_argument('--models', nargs='+', required=True,
                        help='List of model names to process and plot.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for selecting evaluations.')
    parser.add_argument('--skip_steps', type=int, nargs='*', default=[],
                        help='List of evaluation steps to skip.')
    parser.add_argument('--output', type=str, default='scripts/plots',
                        help='Directory to save plotted graphs.')

    args = parser.parse_args()

    print(f"Collecting data for models: {args.models}")
    if args.skip_steps:
        print(f"Skipping specified steps: {args.skip_steps}")

    data = collect_data(args.log_dir, args.models, args.seed, set(args.skip_steps))

    if data:
        plot_data(data, args.output)
    else:
        print("No valid data found. Please verify your folder structure and paths.")

if __name__ == "__main__":
    main()
