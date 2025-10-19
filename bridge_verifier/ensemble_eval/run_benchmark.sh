#!/bin/bash
# Script to run latency benchmark on merged ensemble

cd /root/vla-clip/bridge_verifier

echo "========================================"
echo "Latency Benchmark - Merged Ensemble"
echo "========================================"
echo ""

python3 ensemble_eval/benchmark_latency.py \
  --merged_checkpoint downloads/ensemble_789_trainable_only.pt \
  --device cuda \
  --warmup_runs 10 \
  --benchmark_runs 50 \
  --batch_sizes 1 5 10 20 50 \
  --output logs/benchmark_results.json

echo ""
echo "========================================"
echo "✅ Benchmark complete!"
echo "Results saved to: logs/benchmark_results.json"
echo "========================================"

