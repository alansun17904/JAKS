#!/usr/bin/env bash

# Circuit Stability Analysis Script
# This script runs proper circuit discovery with graph visualization

echo "🔬 Starting Circuit Stability Analysis..."

# Set up environment
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

# Create output directory
mkdir -p results/circuit_analysis
cd results/circuit_analysis

# Run circuit discovery with custom dataset (now has actual data)
echo "📊 Running circuit discovery on arithmetic dataset..."

python ../../src/experiments/circuit_discovery.py \
  "gpt2" \
  "output.pkl" \
  --batch_size 1 \
  --ndevices 1 \
  --device "cuda" \
  --seed 42 \
  --dataset custom \
  --format zero-shot \
  --extraction last_token \
  --patching_metric kl \
  --ig_steps 5

echo "✅ Circuit analysis complete!"
echo ""
echo "📈 Results generated:"
echo "   - Circuit graphs: *.png files"
echo "   - Circuit data: *.json files" 
echo "   - Performance metrics in terminal output"
echo ""
echo "🔍 View the circuit graphs to see discovered computation paths!"