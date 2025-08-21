import json
import yaml
from tqdm import tqdm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge the partition results")
    parser.add_argument("--result_path", required=True, type=str, help="Path of the merged results")
    parser.add_argument("--output_path", required=True, type=str, help="Output path of the top-k results")
    parser.add_argument("--k", required=True, type=int, help="Top K")
    args = parser.parse_args()
    
    """
    Assume the `result_path` contains a JSON like:
    {
        task: {
            annotation: {
                "success": ...,
                "total": ...,
            }
        }
    }

    Output the YAML file like:
    task1:
    - annotation1
    - annotation2
    task2:
    ...
    ...
    """
    output = {}
    with open(args.result_path, "r") as f:
        results = json.load(f)
        for task, task_info in tqdm(results.items()):
            # [(annotation, {"success": ..., "total": ....}), ...]
            sorted_task_infos = sorted(list(task_info.items()), key=lambda item: item[1]["success"])
            # [annotation_1, annotation_2, ..., annotation_k]
            topK = list(map(lambda x: x[0], sorted_task_infos[:args.k]))
            output[task] = topK
    
    with open(args.output_path, "w") as f:
        yaml.dump(output, f)

