import os
import json
from pathlib import Path
from libero.libero import benchmark
import sys
sys.path.append("/root/vla-clip/clip_verifier/scripts")
from lang_transform import LangTransform

# Output file
OUTPUT_PATH = "libero_rephrase_out_set_new.json"

# List of task suites to process
TASK_SUITES = [
    "libero_spatial",
    # "libero_object",
    # "libero_goal",
    # "libero_10",
    # "libero_90",
]

BATCH_NUMBER = 25
LANG_TRANSFORM_TYPE = "no_transform"

def main():
    lang_transform = LangTransform()
    benchmark_dict = benchmark.get_benchmark_dict()
    all_rephrases = {}

    for suite_name in TASK_SUITES:
        print(f"Processing suite: {suite_name}")
        task_suite = benchmark_dict[suite_name]()
        n_tasks = task_suite.n_tasks
        suite_rephrases = {}
        for task_id in range(n_tasks):
            task = task_suite.get_task(task_id)
            original_instruction = task.language
            rephrases = lang_transform.transform(original_instruction, LANG_TRANSFORM_TYPE, batch_number=BATCH_NUMBER)
            suite_rephrases[task_id] = {
                "original": original_instruction,
                "rephrases": rephrases,
            }
            print(f"  Task {task_id}: {original_instruction} -> {len(rephrases)} rephrases")
        all_rephrases[suite_name] = suite_rephrases

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_rephrases, f, indent=2, ensure_ascii=False)
    print(f"Saved rephrases to {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 