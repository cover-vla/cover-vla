import os
import subprocess
from tqdm import tqdm

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate instructions for red teaming")
    parser.add_argument("--n", required=True, type=int, help="Number of runs")
    parser.add_argument("--k", required=True, type=int, help="Index of iterations")
    args = parser.parse_args()

    output_path = "annotations/calvin/div_vlm/{i}/{k}.yaml"
    example_path = "annotations/calvin/div_vlm/{i}/{k}/top3.yaml"

    # Assuming you're in the same directory with `annotations` and `generate_instructions.py`
    for i in tqdm(range(args.n)):
        cmd_args = [
            f"--output_path={output_path.format(i=i, k=args.k)}",
            f"--task_images=vlm_initial_state_links.json",
            "--use_image",
        ]
        
        if args.k > 0:
            cmd_args.append(f"--examples_path={example_path.format(i=i, k=args.k - 1)}")
            if not os.path.exists(f"{example_path.format(i=i, k=args.k - 1)}"):
                print(f"Example file does't exist: {example_path.format(i=i, k=args.k - 1)}")
                continue
        
        if os.path.exists(f"{output_path.format(i=i, k=args.k)}"):
            print(f"File exists: {output_path.format(i=i, k=args.k)}")
            continue
        
        result = subprocess.run([
            "python",  "generate_instructions.py", *cmd_args], 
            capture_output=True, text=True)
        print(result.stdout)

