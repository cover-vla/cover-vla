#!/bin/bash

set -e

REPO_URL="https://github.com/monkey-vla/RoboMonkey.git"
REPO_NAME="RoboMonkey"

# Retry logic with max attempts
retry_with_limit() {
    local cmd="$1"
    local msg="$2"
    local max_attempts=3
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        echo "[$msg] Attempt $attempt of $max_attempts..."
        eval "$cmd"
        if [ $? -eq 0 ]; then
            echo "[$msg] succeeded."
            return 0
        else
            echo "[$msg] failed."
            ((attempt++))
            sleep 5
        fi
    done

    echo "❌ [$msg] failed after $max_attempts attempts."
    echo "❗ Please check the logs or fix any issues, then rerun the script."
    exit 1
}

# Step 1: Run env.sh
retry_with_limit "bash scripts/env.sh" "env.sh"

# Step 2: Run env_verifier.sh
retry_with_limit "bash scripts/env_verifier.sh" "env_verifier.sh"

# Step 3: Run env_sglang.sh
retry_with_limit "bash scripts/env_sglang.sh" "env_sglang.sh"

# Step 4: Run env_simpler.sh
retry_with_limit "bash scripts/env_simpler.sh" "env_simpler.sh"

echo "✅ All setup steps completed successfully."
