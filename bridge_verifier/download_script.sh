#!/bin/bash

# Target S3 directory
S3_PATH="s3://bridge-data-bucket/vla-clip/final_run/"

# Create local directory if needed
mkdir -p ./downloads

# Loop through matching .pt files in S3
for file in $(aws s3 ls "$S3_PATH" | awk '{print $4}' | grep '^bridge_4096_6e5_64_epoch_16.*\.pt$'); do
    echo "Downloading $S3_PATH$file ..."
    aws s3 cp "$S3_PATH$file" "./downloads/$file"
done

echo "✅ All bridge_4096_6e5_64_epoch_*.pt files downloaded successfully to ./downloads/"
