#!/bin/bash
# Unified Environment Setup Script for Tree-of-Thought + Circuit Stability
# This script creates a single virtual environment with all dependencies

set -e  # Exit on any error

echo "🚀 Setting up unified Tree-of-Thought + Circuit Stability environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [[ ! -d "tree-of-thought" || ! -d "circuit-stability" ]]; then
    echo -e "${RED}❌ Error: Please run this script from the JAKS root directory${NC}"
    echo "Expected structure: JAKS/tree-of-thought and JAKS/circuit-stability"
    exit 1
fi

echo -e "${BLUE}📍 Working directory: $(pwd)${NC}"

# Remove any existing virtual environments
echo -e "${YELLOW}🧹 Cleaning up existing environments...${NC}"
rm -rf venv tot_env circuit_env .venv

# Create new unified virtual environment
echo -e "${BLUE}🐍 Creating unified virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}⬆️ Upgrading pip...${NC}"
pip install --upgrade pip

# Install unified requirements
echo -e "${BLUE}📦 Installing unified dependencies...${NC}"
pip install -r requirements_unified.txt

# Install Tree-of-Thought package in development mode
echo -e "${BLUE}🌲 Installing Tree-of-Thought package...${NC}"
cd tree-of-thought
pip install -e .
cd ..

# Verify installations
echo -e "${BLUE}✅ Verifying installations...${NC}"

# Test imports
python3 -c "
import sys
import traceback
success = True

# Test core dependencies
tests = [
    ('torch', 'PyTorch'),
    ('transformers', 'Huggingface Transformers'), 
    ('transformer_lens', 'TransformerLens'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('matplotlib', 'Matplotlib'),
]

for module, name in tests:
    try:
        __import__(module)
        print(f'✅ {name}: OK')
    except ImportError as e:
        print(f'❌ {name}: FAILED - {e}')
        success = False

# Test Tree-of-Thought imports
try:
    from tot.tasks import get_task
    from tot.methods.bfs import solve
    print('✅ Tree-of-Thought: OK')
except ImportError as e:
    print(f'❌ Tree-of-Thought: FAILED - {e}')
    success = False

# Test circuit-stability imports  
try:
    sys.path.append('circuit-stability/code/src')
    from eap import Graph
    from cdatasets import DatasetBuilder
    print('✅ Circuit Stability: OK')
except ImportError as e:
    print(f'❌ Circuit Stability: FAILED - {e}')
    success = False

if not success:
    print('❌ Some imports failed!')
    sys.exit(1)
else:
    print('✅ All imports successful!')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}🎉 Unified environment setup complete!${NC}"
    echo ""
    echo -e "${BLUE}📋 Usage Instructions:${NC}"
    echo "1. Activate environment: source venv/bin/activate"
    echo "2. Run Tree-of-Thought with circuit evaluation:"
    echo "   cd tree-of-thought"
    echo "   PYTHONPATH=\"\$(pwd)/src:\${PYTHONPATH:-}\" python3 run.py --task game24 --method_evaluate circuits --task_start_index 900 --task_end_index 901"
    echo ""
    echo -e "${GREEN}✅ Environment ready for integrated Tree-of-Thought + Circuit Stability research!${NC}"
else
    echo -e "${RED}❌ Setup failed during verification. Check error messages above.${NC}"
    exit 1
fi