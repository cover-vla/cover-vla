#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <directory> <old_phrase> <new_phrase>"
    exit 1
fi

directory=$1
old_phrase=$2
new_phrase=$3

# Check if directory exists
if [ ! -d "$directory" ]; then
    echo "Error: Directory '$directory' does not exist."
    exit 1
fi

# Iterate through files in the directory
for file in "$directory"/*; do
    if [[ -f "$file" ]]; then
        base_name=$(basename "$file")
        if [[ "$base_name" == *"$old_phrase"* ]]; then
            new_base_name="${base_name//$old_phrase/$new_phrase}"
            new_name="$directory/$new_base_name"
            mv "$file" "$new_name"
            echo "Renamed: '$file' -> '$new_name'"
        fi
    fi
done

echo "Renaming completed."