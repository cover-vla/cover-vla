#!/bin/bash
set -e

LOG_DIR="logs"

printf "%-40s %-20s %-20s\n" "Checkpoint" "Overall Accuracy" "Mean Rank"
printf "%-40s %-20s %-20s\n" "----------------------------------------" "--------------------" "--------------------"

for log_file in $LOG_DIR/*.log; do
    ckpt=$(basename "$log_file" | sed 's/^eval_//' | sed 's/\.log$//')

    # Extract "Overall accuracy" line
    acc_line=$(grep -Ei "overall accuracy" "$log_file" | tail -n 1)
    # Extract "mean rank" line
    rank_line=$(grep -Ei "mean rank" "$log_file" | tail -n 1)

    if [ -n "$acc_line" ]; then
        acc_value=$(echo "$acc_line" | grep -oE "[0-9]+\.[0-9]+")
    else
        acc_value="N/A"
    fi

    if [ -n "$rank_line" ]; then
        rank_value=$(echo "$rank_line" | grep -oE "[0-9]+\.[0-9]+")
    else
        rank_value="N/A"
    fi

    printf "%-40s %-20s %-20s\n" "$ckpt" "$acc_value" "$rank_value"
done
