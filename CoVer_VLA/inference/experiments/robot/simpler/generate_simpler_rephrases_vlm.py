import os
import json
from pathlib import Path
import argparse
from typing import List, Set, Optional, Dict
import sys

# Resolve vla-clip root dynamically (go up from this file: simpler -> robot -> experiments -> inference -> CoVer_VLA -> vla-clip)
_SCRIPT_DIR = Path(__file__).resolve().parent
_VLA_CLIP_ROOT = _SCRIPT_DIR.parents[5]
sys.path.insert(0, str(_VLA_CLIP_ROOT))
from bridge_verifier.lang_transform_vlm import LangTransform

BATCH_NUMBER = 100
LANG_TRANSFORM_TYPE = "rephrase"
MAX_DUPLICATE_REPLACEMENT_ATTEMPTS = 5  # Maximum attempts to replace duplicates
MAX_GENERATION_ROUNDS = 50  # Safety cap to avoid infinite loops when topping up
INITIAL_FRAME_DIR = str(_VLA_CLIP_ROOT / "bridge_verifier" / "initial_frame")

def check_and_replace_duplicates(lang_transform, original_instruction, instructions, image_path, max_attempts=MAX_DUPLICATE_REPLACEMENT_ATTEMPTS):
    """
    Check for duplicate instructions and replace them with new ones.
    
    Args:
        lang_transform: LangTransform instance
        original_instruction: The original instruction
        instructions: List of generated instructions
        image_path: Path to the image for this task (can be None)
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
                    new_instructions = lang_transform.transform(original_instruction, batch_number=10, image=image_path)
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

def load_image_mapping(image_dir: str) -> Dict[str, str]:
    """
    Load all images from the initial_frame directory and create a mapping
    from normalized task names to image file paths.
    
    Returns:
        Dictionary mapping normalized task names to image file paths
    """
    image_map = {}
    image_path = Path(image_dir)
    
    if not image_path.exists():
        print(f"Warning: Image directory {image_dir} does not exist")
        return image_map
    
    for img_file in image_path.glob("*.png"):
        # Remove .png extension and normalize
        task_name = img_file.stem.lower().strip()
        image_map[task_name] = str(img_file)
    
    print(f"Loaded {len(image_map)} images from {image_dir}")
    return image_map

def find_matching_image(instruction: str, image_map: Dict[str, str]) -> Optional[str]:
    """
    Find the best matching image for a given instruction.
    Performs exact matching by normalizing spaces/underscores.
    
    Args:
        instruction: The instruction text to match (e.g., "put tennis ball into yellow basket")
        image_map: Dictionary mapping normalized task names to image paths
    
    Returns:
        Path to matching image, or None if no match found
    """
    if not image_map:
        return None
    
    # Normalize the instruction (lowercase, strip)
    normalized_instruction = normalize_text(instruction)
    
    # Convert instruction spaces to underscores for comparison
    # e.g., "put tennis ball into yellow basket" -> "put_tennis_ball_into_yellow_basket"
    instruction_normalized = normalized_instruction.replace(" ", "_")
    
    # Try exact match with normalized instruction (with underscores)
    if instruction_normalized in image_map:
        return image_map[instruction_normalized]
    
    # Also try exact match with original normalized instruction (in case image name uses spaces)
    if normalized_instruction in image_map:
        return image_map[normalized_instruction]
    
    # If no exact match found, this is an error - print detailed error message
    print(f"    ERROR: No exact match found for instruction: '{instruction}'")
    print(f"    Normalized instruction: '{normalized_instruction}'")
    print(f"    Instruction with underscores: '{instruction_normalized}'")
    print(f"    Available image names: {sorted(image_map.keys())}")
    
    return None

def generate_unique_rephrases(
    lang_transform: LangTransform,
    original_instruction: str,
    existing_normalized: Set[str],
    needed: int,
    image_path: Optional[str] = None,
    batch_number: int = BATCH_NUMBER,
) -> List[str]:
    """
    Generate up to `needed` unique rephrases that are not in `existing_normalized`.
    Filters very short responses and avoids duplicates and the original text.
    
    Args:
        lang_transform: LangTransform instance
        original_instruction: The original instruction to rephrase
        existing_normalized: Set of already seen normalized instructions
        needed: Number of unique rephrases needed
        image_path: Optional path to image for this task
        batch_number: Number of rephrases to generate in one batch
    """
    collected: List[str] = []
    
    # Generate a larger batch to account for potential duplicates
    # Request 1.5x what we need to have better chances of getting enough unique ones
    request_batch_size = max(needed, 10)  # At least 10 to get good variety
    
    try:
        new_instructions = lang_transform.transform(
            original_instruction, batch_number=request_batch_size, image=image_path
        )
    except Exception as e:
        print(f"    Error during generation: {e}")
        return collected

    if not new_instructions:
        return collected

    # Filter and collect unique rephrases
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
            f"    Warning: only generated {len(collected)} of {needed} required unique rephrases"
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

    # Load image mapping
    image_map = load_image_mapping(INITIAL_FRAME_DIR)
    
    lang_transform = LangTransform()
    instructions_dict = data["instructions"]

    updated_total = 0
    for key, entry in instructions_dict.items():
        try:
            original = entry.get("original") or key
            rephrases: List[str] = entry.get("ert_rephrases", []) or []
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

        # Find matching image for this task
        # Use the key (task name) for matching, not the "original" field
        # e.g., key="put tennis ball into yellow basket" should match "put_tennis_ball_into_yellow_basket.png"
        image_path = find_matching_image(key, image_map)
        if image_path:
            print(f"    Using image: {Path(image_path).name}")
        else:
            print(f"    ERROR: No matching image found for task key: '{key}'")
            print(f"    This is an error - image matching must be exact!")

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
            image_path=image_path,
        )

        if additions:
            entry.setdefault("ert_rephrases", []).extend(additions)
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