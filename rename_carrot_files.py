#!/usr/bin/env python3
import os
import glob

def rename_carrot_files():
    """
    Rename all files containing 'place_the_carrot_on_the_dish_' 
    to use 'put_carrot_on_plate' instead.
    """
    # Directory containing the files
    directory = "/root/vla-clip/RoboMonkey/openvla-mini/experiments/robot/simpler/rollouts_clip/robomonkey"
    
    # Pattern to match carrot files
    pattern = "*place_the_spoon_on_the_towel_*"
    
    # Find all matching files
    matching_files = glob.glob(os.path.join(directory, pattern))
    
    if not matching_files:
        print("No files found matching the pattern.")
        return
    
    print(f"Found {len(matching_files)} files to rename:")
    
    # Rename each file
    for old_path in matching_files:
        # Get the directory and filename
        dir_path = os.path.dirname(old_path)
        filename = os.path.basename(old_path)
        
        # Create new filename by replacing the task description
        new_filename = filename.replace("place_the_spoon_on_the_towel_", "put_the_spoon_on_the_towel")
        new_path = os.path.join(dir_path, new_filename)
        
        # Rename the file
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
        except Exception as e:
            print(f"Error renaming {filename}: {e}")
    
    print(f"\nSuccessfully renamed {len(matching_files)} files.")

if __name__ == "__main__":
    rename_carrot_files()
