import glob
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import draccus

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from src.agent.configuration_pipeline import TrainPipelineConfig


def parse_metrics(log_path: str) -> Dict[str, Dict[str, float]]:
    """Parses the log file to extract arbitrary metrics for each task,
       excluding 'Number of episodes' and 'Total Task Eval Time'."""
    all_task_metrics: Dict[str, Dict[str, float]] = {}
    current_task: str | None = None
    parsing_summary = False

    excluded = {"Number of episodes", "Total Task Eval Time"}
    metric_re = re.compile(r"([-\d\.]+)\s*(\D*)$")  # capture number + optional unit at end

    try:
        with open(log_path) as f:
            for line in f:
                # detect task name
                m = re.search(r"Task suite:\s*(.+)", line)
                if m:
                    current_task = m.group(1).strip()
                    parsing_summary = False
                    all_task_metrics.setdefault(current_task, {})
                    continue

                # start / end of summary
                if "Evaluation Summary" in line and current_task:
                    parsing_summary = True
                    continue
                if parsing_summary and line.strip().startswith("==="):
                    parsing_summary = False
                    continue

                # inside summary: look for any " - Name: Value[ unit]" line
                if parsing_summary and current_task and "- " in line:
                    # strip off everything up to "- "
                    _, tail = line.split("- ", 1)
                    if ":" not in tail:
                        continue
                    name, val_part = tail.split(":", 1)
                    name = name.strip()
                    if name in excluded:
                        continue

                    vp = val_part.strip()
                    m2 = metric_re.match(vp)
                    if not m2:
                        continue
                    val = float(m2.group(1))
                    unit = m2.group(2).strip()
                    if unit.endswith("%"):
                        val /= 100.0

                    all_task_metrics[current_task][name] = val

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
    except Exception as e:
        print(f"An error occurred while parsing {log_path}: {e}")

    return all_task_metrics
def find_latest_timestamp_dir(path):
    dirs = glob.glob(os.path.join(path, '*'))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        return None
    try:
        # Ensure robustness against directories not matching the expected format
        dirs.sort(key=lambda p: datetime.strptime(os.path.basename(p), "%Y-%m-%d_%H-%M-%S"), reverse=True)
        return dirs[0], dirs
    except ValueError:
        print(f"Warning: Could not parse timestamp directories in {path}. Returning the lexicographically largest.")
        dirs.sort(reverse=True) # Fallback to lexicographical sort
        return dirs[0], dirs


def collect_data(root_dir: Path, cfg: TrainPipelineConfig) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Collects and averages metrics across all seeds for each step/task."""
    # raw accumulators: step -> task -> metric -> list of values
    raw: Dict[int, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    steps = cfg.eval_cfg.pretrained_model_gradient_step_cnt

    for step in steps:
        step_int = int(step)
        step_dir = root_dir / f"step_{step}"
        action_step_dir = step_dir / f"ta_{cfg.eval_cfg.action_step}"
        if not action_step_dir.is_dir():
            print(f"No action_step dir '{action_step_dir}', skipping step {step}")
            continue

        # find every seed subfolder
        seed_dirs = [d for d in action_step_dir.iterdir() if d.is_dir()]
        if not seed_dirs:
            print(f"No seeds under '{action_step_dir}', skipping step {step}")
            continue

        for seed_dir in seed_dirs:
            ts_dir, sorted_dir = find_latest_timestamp_dir(str(seed_dir))
            if not ts_dir:
                print(f"  no timestamp in {seed_dir}, skipping")
                continue

            log_file = os.path.join(ts_dir, "eval.log")
            if not os.path.isfile(log_file):
                print(f"  no eval.log in {ts_dir}, skipping")
                continue

            task_metrics = parse_metrics(log_file)
            for task, metrics in task_metrics.items():
                for name, val in metrics.items():
                    raw[step_int][task][name].append(val)

            # ! scan over non-latest timestamp dirs to get older tasks
            # ! if a task is already recorded, do not update it again
            recorded_tasks_this_seed = set(list(task_metrics.keys()))
            non_latest_dirs = [d for d in sorted_dir if d != ts_dir]
            for d in non_latest_dirs:
                print(f"processing {os.path.basename(d)}")
                log_file = os.path.join(d, "eval.log")
                if not os.path.isfile(log_file):
                    print(f"  no eval.log in {d}, skipping")
                    continue
                task_metrics = parse_metrics(log_file)
                for task, metrics in task_metrics.items():
                    if task in recorded_tasks_this_seed:
                        print(f"** {task} **   already recorded in newer runs, skipping its result in timestamp dir {d}")
                        continue
                    else:
                        recorded_tasks_this_seed.add(task)
                        for name, val in metrics.items():
                            raw[step_int][task][name].append(val)
            print(f"Finish collecting for seed {seed_dir}, step {step}")

    # now compute averages
    averaged: Dict[int, Dict[str, Dict[str, float]]] = {}
    for step_int, tasks in raw.items():
        averaged[step_int] = {}
        for task, metrics in tasks.items():
            averaged[step_int][task] = {
                name: sum(vals) / len(vals) for name, vals in metrics.items()
            }

    return averaged


def push_to_wandb(data, cfg):
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.name + f'_ta{cfg.eval_cfg.action_step}',
        config=asdict(cfg),
        entity=cfg.wandb.entity,
        )
    # ! for models with only one step, hard code the steps to plot aligned horizontal line
    if len(cfg.eval_cfg.pretrained_model_gradient_step_cnt) == 1:
        step = cfg.eval_cfg.pretrained_model_gradient_step_cnt[0]
        wandb_dict = {}
        for task, metrics in data[step].items():
            for name, val in metrics.items():
                wandb_dict[f"{task}_{name}"] = val
        artificial_steps = [1513, 3026, 4539, 7565, 15130, 22695]
        for astep in artificial_steps:
            if astep != int(step):  # Skip the actual step for now
                wandb.log(wandb_dict, step=astep, commit=True)
        # Log the actual step last to ensure it's the "real" data point
        wandb.log(wandb_dict, step=int(step), commit=True)
    else:
        for step in cfg.eval_cfg.pretrained_model_gradient_step_cnt:
            wandb_dict = {}
            for task, metrics in data[step].items():
                for name, val in metrics.items():
                    wandb_dict[f"{task}_{name}"] = val
            wandb.log(wandb_dict, step=int(step), commit=True)

    wandb.finish()

@draccus.wrap()
def main(pipeline_cfg: TrainPipelineConfig):

    model_name = pipeline_cfg.name
    vla_log_dir = os.environ.get("VLA_LOG_DIR")
    if not vla_log_dir:
        print("Error: VLA_LOG_DIR environment variable not set.")
        sys.exit(1)

    root_dir: Path = (
        Path(vla_log_dir)
        / "eval_online"
        / pipeline_cfg.eval_cfg.simulator_name
        / model_name
    )
    print(f"Collecting data from root directory: {root_dir}")
    print(f"Collecting data for models: {model_name}")

    # Ensure steps are integers for dictionary keys
    if pipeline_cfg.eval_cfg.pretrained_model_gradient_step_cnt is not None:
        pipeline_cfg.eval_cfg.pretrained_model_gradient_step_cnt = [
            int(s) for s in pipeline_cfg.eval_cfg.pretrained_model_gradient_step_cnt
        ]
    else:
        # artificially add a step for models with only one step
        pipeline_cfg.eval_cfg.pretrained_model_gradient_step_cnt = [15130]

    data = collect_data(root_dir=root_dir, cfg=pipeline_cfg)

    # print(data)
    if data:
        print(f"Collected data for steps: {sorted(data.keys())}")
        push_to_wandb(data, pipeline_cfg)
    else:
        print("No valid data found. Please verify your folder structure, paths, and log file content.")


if __name__ == "__main__":
    main()
