import os
import json
import glob
from collections import defaultdict
import re
from tqdm import tqdm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge the partition results")
    parser.add_argument("--dir", required=True, type=str, help="Directory of partition results")
    parser.add_argument("--output_path", required=True, type=str, help="Output path of the merged results")
    parser.add_argument("--num_partitions", required=True, type=int, help="Number of partitions")
    args = parser.parse_args()
    
    """
    Will produce a JSON file with the following structure
    {
        task: {
            annotation: {
                "success": ..., 
                "total": ...
            },
            ...
        },
        ...
    }
    """
    output = defaultdict( # Task name
        lambda: defaultdict( # Annotation name
            lambda: defaultdict( # success/total dict
                lambda: 0 # initialize as zeros
            )
        )
    )
    
    # Scan over all partitions
    for partition_idx in tqdm(range(args.num_partitions)):
        pattern = r"result-" + str(partition_idx) + r"-of-" + str(args.num_partitions)
        partition_result = defaultdict(dict)

        # Scan over all copies of the results in this partition (there many copies at different time of evalution so we have to aggregate them)
        for result_path in glob.glob(f"{args.dir}/result-{partition_idx}-of-{args.num_partitions}*.txt"):
            with open(result_path, "r") as f:
                result = json.load(f)
            
            for task, task_info in result["null"]["task_info"].items():
                for annotation, annotation_info in task_info.items():
                    # No need to take `max` over the existing ones, since the success-total of a task-annotation pair won't change after it was saved in the file
                    partition_result[task][annotation] = annotation_info

        # Accumulate to the merged output
        for task, task_info in partition_result.items():
            for annotation, annotation_info in task_info.items():
                for k, v in annotation_info.items():
                    output[task][annotation][k] += v
    
    # Output the result
    with open(args.output_path, "w") as f:
        json.dump(output, f)
    print(f"Save the merged results at: {args.output_path}")

