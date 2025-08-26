#!/bin/bash
# Unified Environment Activation Script
# Activates the single virtual environment for Tree-of-Thought + Circuit Stability

echo "🌲🔬 Activating unified Tree-of-Thought + Circuit Stability environment..."

# Check if unified environment exists
if [[ ! -d "venv" ]]; then
    echo "❌ Unified environment not found! Please run ./setup_unified_environment.sh first"
    return 1 2>/dev/null || exit 1
fi

# Activate environment
source venv/bin/activate

# Set up Python paths
export PYTHONPATH="$(pwd)/tree-of-thought/src:$(pwd)/circuit-stability/code/src:${PYTHONPATH:-}"

echo ""
echo "✅ Unified environment is now active!"
echo ""
echo "🔧 Environment Details:"
echo "   Python version: $(python --version)"
echo "   Virtual env: $(which python)"
echo "   PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "   Device: $(python -c 'import torch; print("MPS" if torch.backends.mps.is_available() else "CPU")')"
echo ""
echo "🚀 Quick Start Commands:"
echo "   Test integration:      cd tree-of-thought && PYTHONPATH=\"\$(pwd)/src:\${PYTHONPATH:-}\" python3 run.py --task game24 --method_evaluate circuits --task_start_index 900 --task_end_index 901"
echo "   Run Tree-of-Thought:   cd tree-of-thought && python run.py --task game24 --task_start_index 0 --task_end_index 1"
echo "   Test circuit analysis: cd circuit-stability/code && python src/experiments/circuit_discovery.py gpt2 test --dataset custom --format zero-shot --device mps"
echo ""
echo "📖 Integration ready for mechanistic interpretability research!"
echo "