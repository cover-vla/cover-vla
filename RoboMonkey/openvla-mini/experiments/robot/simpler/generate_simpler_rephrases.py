import os
import json
from pathlib import Path
import argparse
from typing import List, Set
import sys
sys.path.append("/root/vla-clip/clip_verifier/scripts")
from lang_transform import LangTransform

BATCH_NUMBER = 100
LANG_TRANSFORM_TYPE = "rephrase"
MAX_DUPLICATE_REPLACEMENT_ATTEMPTS = 5  # Maximum attempts to replace duplicates
MAX_GENERATION_ROUNDS = 50  # Safety cap to avoid infinite loops when topping up

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
                    new_instructions = lang_transform.transform(original_instruction, transform_type, batch_number=10)
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

def normalize_text(text: str) -> str:
    return text.lower().strip()

def generate_unique_rephrases(
    lang_transform: LangTransform,
    original_instruction: str,
    existing_normalized: Set[str],
    needed: int,
    transform_type: str = LANG_TRANSFORM_TYPE,
    batch_number: int = BATCH_NUMBER,
) -> List[str]:
    """
    Generate up to `needed` unique rephrases that are not in `existing_normalized`.
    Filters very short responses and avoids duplicates and the original text.
    """
    collected: List[str] = []
    rounds = 0
    while len(collected) < needed and rounds < MAX_GENERATION_ROUNDS:
        rounds += 1
        try:
            new_instructions = lang_transform.transform(
                original_instruction, transform_type, batch_number=batch_number
            )
        except Exception as e:
            print(f"    Error during generation round {rounds}: {e}")
            continue

        if not new_instructions:
            continue

        for candidate in new_instructions:
            if len(candidate.strip()) < 10:
                continue
            norm = normalize_text(candidate)
            if norm == normalize_text(original_instruction):
                continue
            if norm in existing_normalized:
                continue
            collected.append(candidate)
            existing_normalized.add(norm)
            if len(collected) >= needed:
                break

    if len(collected) < needed:
        print(
            f"    Warning: only generated {len(collected)} of {needed} required unique rephrases after {rounds} rounds"
        )
    return collected

def top_up_simpler_rephrases(input_json_path: str, save_in_place: bool = True) -> str:
    """
    Load a simpler_rephrased.json file, and for each instruction ensure the number
    of entries in "rephrases" equals the value in "count" by generating additional
    unique rephrases as needed. Returns the output path written to.
    """
    print(f"Loading simpler rephrases from: {input_json_path}")
    with open(input_json_path, "r") as f:
        data = json.load(f)

    if "instructions" not in data or not isinstance(data["instructions"], dict):
        raise ValueError("Invalid simpler JSON: missing 'instructions' dictionary")

    lang_transform = LangTransform()
    instructions_dict = data["instructions"]

    updated_total = 0
    for key, entry in instructions_dict.items():
        try:
            original = entry.get("original") or key
            rephrases: List[str] = entry.get("rephrases", []) or []
            target_count: int = int(entry.get("count", 0))
        except Exception:
            print(f"  Skipping malformed entry for key: {key}")
            continue

        current_count = len(rephrases)
        if target_count <= 0:
            print(f"  [{key}] target count is {target_count}; skipping")
            continue

        if current_count >= target_count:
            print(f"  [{key}] already has {current_count}/{target_count} rephrases; OK")
            continue

        print(f"  [{key}] topping up {current_count}->{target_count} rephrases")

        # Build normalized seen set from existing rephrases and the original
        seen: Set[str] = {normalize_text(original)}
        for r in rephrases:
            seen.add(normalize_text(r))

        needed = target_count - current_count
        additions = generate_unique_rephrases(
            lang_transform=lang_transform,
            original_instruction=original,
            existing_normalized=seen,
            needed=needed,
            transform_type=LANG_TRANSFORM_TYPE,
            batch_number=min(BATCH_NUMBER, max(needed, 5)),
        )

        if additions:
            entry.setdefault("rephrases", []).extend(additions)
            updated_total += len(additions)
            print(f"    Added {len(additions)} new rephrases")
        else:
            print("    No new rephrases added")

    # Decide output path
    if save_in_place:
        output_path = input_json_path
    else:
        p = Path(input_json_path)
        output_path = str(p.with_name(p.stem + "_topped_up" + p.suffix))

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Completed top-up. Added total of {updated_total} rephrases. Saved to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Top-up Simpler rephrases to match target count")
    parser.add_argument(
        "--simpler-json",
        type=str,
        required=True,
        help="Path to simpler_rephrased.json to top-up 'rephrases' to 'count'",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite input JSON; write to *_topped_up.json instead",
    )

    args = parser.parse_args()

    input_path = args.simpler_json
    save_in_place = not args.no_overwrite
    top_up_simpler_rephrases(input_path, save_in_place=save_in_place)

if __name__ == "__main__":
    main() 