#!/bin/bash
# Tree of Thoughts Environment Activation Script
# This script activates the conda environment for ToT experiments

echo "🌲 Activating Tree of Thoughts environment..."

# Add miniconda to PATH if needed
export PATH="$HOME/miniconda3/bin:$PATH"

# Initialize conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Activate environment
conda activate tot_env

echo ""
echo "✅ Tree of Thoughts environment is now active!"
echo ""
echo "🔧 Environment Details:"
echo "   Python version: $(python --version)"
echo "   Environment: $CONDA_DEFAULT_ENV"
echo "   PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "   CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "🚀 Quick Start Commands:"
echo "   Test installation:     python -c 'from tot.tasks import get_task; print(\"✅ ToT ready!\")'  "
echo "   Run Game24 example:    python run.py --task game24 --task_start_index 0 --task_end_index 1"
echo "   View help:             python run.py --help"
echo ""
echo "📖 For mechanistic interpretability research:"
echo "   - Use TransformerLens for model analysis"
echo "   - Modify evaluation methods in src/tot/methods/bfs.py:243"
echo "   - Circuit analysis hooks are ready for integration"
echo ""