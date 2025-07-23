import os
import json
from pathlib import Path
from libero.libero import benchmark
import sys
sys.path.append("/root/vla-clip/clip_verifier/scripts")
from lang_transform import LangTransform

# Output file
OUTPUT_PATH = "libero_rephrase_pos_rephrase_neg_negation.json"

# List of task suites to process
TASK_SUITES = [
    "libero_spatial",
    # "libero_object",
    # "libero_goal",
    # "libero_10",
    # "libero_90",
]

BATCH_NUMBER = 100
LANG_TRANSFORM_TYPE = "rephrase"
NEGATION_TYPE = "negation"
MAX_DUPLICATE_REPLACEMENT_ATTEMPTS = 5  # Maximum attempts to replace duplicates

def check_and_replace_duplicates(lang_transform, original_instruction, instructions, transform_type, max_attempts=MAX_DUPLICATE_REPLACEMENT_ATTEMPTS):
    """
    Check for duplicate instructions and replace them with new ones.
    
    Args:
        lang_transform: LangTransform instance
        original_instruction: The original instruction
        instructions: List of generated instructions
        transform_type: Type of transformation ("rephrase" or "negation")
        max_attempts: Maximum number of attempts to replace duplicates
    
    Returns:
        List of instructions with duplicates replaced
    """
    unique_instructions = []
    seen_instructions = set()
    duplicate_count = 0
    
    for instruction in instructions:
        # Normalize instruction for comparison (lowercase, strip whitespace)
        normalized = instruction.lower().strip()
        
        if normalized in seen_instructions:
            duplicate_count += 1
            print(f"    Found duplicate: '{instruction}'")
            
            # Try to generate a replacement
            replacement_found = False
            for attempt in range(max_attempts):
                try:
                    # Use a larger batch size to get better quality responses
                    new_instructions = lang_transform.transform(original_instruction, transform_type, batch_number=3)
                    if new_instructions:
                        # Filter out invalid short responses and find a unique one
                        for new_instruction in new_instructions:
                            new_normalized = new_instruction.lower().strip()
                            
                            # Skip very short responses (likely errors)
                            if len(new_instruction.strip()) < 10:
                                continue
                                
                            # Check if the new instruction is unique
                            if new_normalized not in seen_instructions and new_normalized != original_instruction.lower().strip():
                                unique_instructions.append(new_instruction)
                                seen_instructions.add(new_normalized)
                                replacement_found = True
                                print(f"    Replaced with: '{new_instruction}' (attempt {attempt + 1})")
                                break
                        
                        if replacement_found:
                            break
                except Exception as e:
                    print(f"    Error generating replacement (attempt {attempt + 1}): {e}")
                    continue
            
            if not replacement_found:
                print(f"    Could not find unique replacement after {max_attempts} attempts, skipping")
        else:
            unique_instructions.append(instruction)
            seen_instructions.add(normalized)
    
    if duplicate_count > 0:
        print(f"    Replaced {duplicate_count} duplicate(s)")
    
    return unique_instructions

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
            print(f"  Task {task_id}: {original_instruction}")
            
            # Generate initial rephrases
            print(f"    Generating {BATCH_NUMBER} rephrases...")
            rephrases = lang_transform.transform(original_instruction, LANG_TRANSFORM_TYPE, batch_number=BATCH_NUMBER)
            
            # Check and replace duplicates in rephrases
            print(f"    Checking for duplicates in rephrases...")
            rephrases = check_and_replace_duplicates(lang_transform, original_instruction, rephrases, LANG_TRANSFORM_TYPE)
            
            # Generate initial negative rephrases
            print(f"    Generating {BATCH_NUMBER} negative rephrases...")
            negative_rephrases = lang_transform.transform(original_instruction, NEGATION_TYPE, batch_number=BATCH_NUMBER)
            
            # Check and replace duplicates in negative rephrases
            print(f"    Checking for duplicates in negative rephrases...")
            negative_rephrases = check_and_replace_duplicates(lang_transform, original_instruction, negative_rephrases, NEGATION_TYPE)
            
            suite_rephrases[task_id] = {
                "original": original_instruction,
                "rephrases": rephrases,
                "negative_rephrases": negative_rephrases,
            }
            print(f"    Final counts: {len(rephrases)} rephrases, {len(negative_rephrases)} negative rephrases")
        all_rephrases[suite_name] = suite_rephrases

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_rephrases, f, indent=2, ensure_ascii=False)
    print(f"Saved rephrases to {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 