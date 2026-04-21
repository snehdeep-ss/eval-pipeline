#!/bin/bash
set -e

echo "=== Preparing data ==="
python prepare_data.py

echo "=== Baseline ==="
python infer.py --mode baseline --limit 200

echo "=== Optimized (4-shot CoT + self-consistency) ==="
python infer.py --mode optimized --limit 200

echo "=== Done. Results in eval_runner/results/ ==="
