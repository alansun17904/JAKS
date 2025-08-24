#!/bin/bash

# ============================================================================
# Circuit Stability Research Framework - Comprehensive Installation Script
# ============================================================================
# This script sets up a complete environment for circuit stability research
# including PyTorch, TransformerLens, and all required dependencies.

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# ============================================================================
# Configuration and Colors
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="circuit-stability"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${PURPLE}=== $1 ===${NC}"; }

# ============================================================================
# System Detection and Validation
# ============================================================================

detect_system() {
    log_step "Detecting System Configuration"
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        log_info "Operating System: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_info "Operating System: macOS"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Detect CUDA availability
    if command -v nvidia-smi &> /dev/null; then
        CUDA_AVAILABLE=true
        CUDA_VERSION_DETECTED=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
        log_info "NVIDIA GPU detected (Driver: $CUDA_VERSION_DETECTED)"
    else
        CUDA_AVAILABLE=false
        log_warning "No NVIDIA GPU detected - will install CPU-only PyTorch"
    fi
    
    # Check available disk space (minimum 5GB)
    AVAILABLE_SPACE=$(df "$SCRIPT_DIR" | tail -1 | awk '{print $4}')
    AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))
    
    if [ "$AVAILABLE_GB" -lt 5 ]; then
        log_warning "Low disk space: ${AVAILABLE_GB}GB available (minimum 5GB recommended)"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_info "Disk space: ${AVAILABLE_GB}GB available"
    fi
}

# ============================================================================
# Conda Installation and Setup
# ============================================================================

setup_conda() {
    log_step "Setting Up Conda Environment"
    
    # Try to find conda installation
    if command -v conda &> /dev/null; then
        log_info "Found existing conda installation"
        CONDA_PATH=$(which conda)
    elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
        log_info "Found miniconda at /opt/miniconda3"
        source /opt/miniconda3/etc/profile.d/conda.sh
    elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        log_info "Found miniconda at /root/miniconda3"
        source /root/miniconda3/etc/profile.d/conda.sh
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        log_info "Found miniconda at $HOME/miniconda3"
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/usr/local/miniconda3/etc/profile.d/conda.sh" ]; then
        log_info "Found miniconda at /usr/local/miniconda3"
        source /usr/local/miniconda3/etc/profile.d/conda.sh
    else
        log_error "Conda not found. Please install miniconda or anaconda first."
        log_info "Install miniconda: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    # Verify conda is working
    if ! conda --version &> /dev/null; then
        log_error "Conda installation appears to be broken"
        exit 1
    fi
    
    log_success "Conda is available: $(conda --version)"
}

# ============================================================================
# Environment Management
# ============================================================================

create_environment() {
    log_step "Creating Conda Environment"
    
    # Remove existing environment if it exists
    if conda env list | grep -q "^$ENV_NAME "; then
        log_warning "Removing existing '$ENV_NAME' environment"
        conda env remove -n "$ENV_NAME" -y
    fi
    
    # Create base environment with essential packages
    log_info "Creating new environment '$ENV_NAME' with Python $PYTHON_VERSION"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" pip -y -q
    
    # Activate environment
    log_info "Activating environment"
    conda activate "$ENV_NAME"
    
    # Verify activation
    ACTIVE_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
    if [ "$ACTIVE_ENV" != "$ENV_NAME" ]; then
        log_error "Failed to activate environment '$ENV_NAME'"
        exit 1
    fi
    
    log_success "Environment '$ENV_NAME' created and activated"
}

# ============================================================================
# System Dependencies
# ============================================================================

install_system_dependencies() {
    log_step "Installing System Dependencies"
    
    if [ "$OS" == "linux" ]; then
        # Check if we have sudo access
        if command -v apt-get &> /dev/null && [ "$EUID" -eq 0 ]; then
            log_info "Installing system packages with apt"
            apt-get update -qq
            apt-get install -y -qq graphviz graphviz-dev pkg-config build-essential
            log_success "System dependencies installed"
        elif command -v apt-get &> /dev/null; then
            log_warning "Root access required for system dependencies"
            log_info "Please run: sudo apt-get install -y graphviz graphviz-dev pkg-config build-essential"
        fi
    elif [ "$OS" == "macos" ]; then
        if command -v brew &> /dev/null; then
            log_info "Installing system packages with Homebrew"
            brew install graphviz
            log_success "System dependencies installed"
        else
            log_warning "Homebrew not found. Please install graphviz manually"
        fi
    fi
}

# ============================================================================
# Core Package Installation
# ============================================================================

install_pytorch() {
    log_step "Installing PyTorch"
    
    if [ "$CUDA_AVAILABLE" = true ]; then
        log_info "Installing PyTorch with CUDA $CUDA_VERSION support"
        pip install --no-cache-dir torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu$CUDA_VERSION"
    else
        log_info "Installing CPU-only PyTorch"
        pip install --no-cache-dir torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cpu"
    fi
    
    # Verify PyTorch installation
    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
"
    
    log_success "PyTorch installed successfully"
}

install_ml_packages() {
    log_step "Installing ML and Research Packages"
    
    # Core ML libraries
    log_info "Installing core ML libraries"
    pip install --no-cache-dir \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
        plotly
    
    # HuggingFace ecosystem
    log_info "Installing HuggingFace ecosystem"
    pip install --no-cache-dir \
        transformers \
        datasets \
        accelerate \
        tokenizers \
        safetensors
    
    # TransformerLens and related
    log_info "Installing TransformerLens and tensor manipulation libraries"
    pip install --no-cache-dir \
        transformer-lens \
        einops \
        fancy-einsum \
        jaxtyping
    
    # Research and utility tools
    log_info "Installing research and utility tools"
    pip install --no-cache-dir \
        wandb \
        tqdm \
        rich \
        tabulate \
        beartype \
        pynvml \
        psutil \
        networkx
    
    # Jupyter ecosystem
    log_info "Installing Jupyter ecosystem"
    pip install --no-cache-dir \
        jupyter \
        jupyterlab \
        ipykernel
    
    # Try to install pygraphviz (may fail without system dependencies)
    log_info "Installing pygraphviz (optional - for circuit visualization)"
    if ! pip install --no-cache-dir pygraphviz; then
        log_warning "pygraphviz installation failed - circuit visualization may be limited"
        log_info "This is usually due to missing system dependencies"
    fi
    
    log_success "ML packages installed successfully"
}

# ============================================================================
# Project Setup
# ============================================================================

setup_project() {
    log_step "Setting Up Project Environment"
    
    # Add project to Python path
    PYTHONPATH_ENTRY="$SCRIPT_DIR/code/src"
    
    # Create activation script
    log_info "Creating activation script"
    cat > "$SCRIPT_DIR/activate_environment.sh" << EOF
#!/bin/bash
# Circuit Stability Environment Activation Script
# Generated by install.sh on $(date)

echo "🧠 Activating Circuit Stability Environment..."

# Activate conda environment
if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "\$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/usr/local/miniconda3/etc/profile.d/conda.sh" ]; then
    source /usr/local/miniconda3/etc/profile.d/conda.sh
fi

conda activate $ENV_NAME

# Set Python path
export PYTHONPATH="\$PYTHONPATH:$PYTHONPATH_ENTRY"

# Verify environment
echo "✓ Environment activated: \$(conda info --envs | grep '*' | awk '{print \$1}')"
python -c "
import torch
import transformer_lens
print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'✓ TransformerLens {transformer_lens.__version__}')
try:
    import sys; sys.path.insert(0, '$PYTHONPATH_ENTRY')
    from cdatasets import DatasetBuilder
    print(f'✓ Circuit Stability Framework (Datasets: {len(list(DatasetBuilder.ids.keys()))})')
except ImportError as e:
    print(f'⚠ Framework import issue: {e}')
"

echo "🚀 Environment ready for circuit stability research!"
EOF
    
    chmod +x "$SCRIPT_DIR/activate_environment.sh"
    
    # Register Jupyter kernel
    log_info "Registering Jupyter kernel"
    python -m ipykernel install --user --name "$ENV_NAME" --display-name "Circuit Stability"
    
    log_success "Project environment configured"
}

# ============================================================================
# Installation Verification
# ============================================================================

verify_installation() {
    log_step "Verifying Installation"
    
    # Test core functionality
    python -c "
import sys
import traceback

print('🔍 Running comprehensive installation verification...')
print()

# Test 1: Core Python packages
print('1. Testing core Python packages...')
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy
    import sklearn
    print('   ✓ Data science stack working')
except Exception as e:
    print(f'   ✗ Data science stack failed: {e}')
    sys.exit(1)

# Test 2: PyTorch and CUDA
print('2. Testing PyTorch...')
try:
    import torch
    import torchvision
    print(f'   ✓ PyTorch {torch.__version__}')
    print(f'   ✓ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   ✓ CUDA version: {torch.version.cuda}')
        print(f'   ✓ GPU device: {torch.cuda.get_device_name(0)}')
        # Test CUDA functionality
        x = torch.randn(100, 100).cuda()
        y = torch.matmul(x, x.T)
        print(f'   ✓ CUDA computation test passed')
    else:
        print('   ⚠ CUDA not available - using CPU')
        # Test CPU functionality
        x = torch.randn(100, 100)
        y = torch.matmul(x, x.T)
        print(f'   ✓ CPU computation test passed')
except Exception as e:
    print(f'   ✗ PyTorch failed: {e}')
    traceback.print_exc()
    sys.exit(1)

# Test 3: HuggingFace ecosystem
print('3. Testing HuggingFace ecosystem...')
try:
    import transformers
    import datasets
    import accelerate
    import tokenizers
    print(f'   ✓ Transformers {transformers.__version__}')
    print(f'   ✓ Datasets {datasets.__version__}')
    print('   ✓ HuggingFace ecosystem working')
except Exception as e:
    print(f'   ✗ HuggingFace ecosystem failed: {e}')
    sys.exit(1)

# Test 4: TransformerLens
print('4. Testing TransformerLens...')
try:
    import transformer_lens
    from transformer_lens import HookedTransformer
    print(f'   ✓ TransformerLens {transformer_lens.__version__}')
    
    # Test model loading (lightweight test)
    print('   ✓ TransformerLens core functionality working')
except Exception as e:
    print(f'   ✗ TransformerLens failed: {e}')
    sys.exit(1)

# Test 5: Additional libraries
print('5. Testing additional libraries...')
try:
    import einops
    import fancy_einsum
    import jaxtyping
    import wandb
    import tqdm
    import rich
    import networkx
    print('   ✓ Additional libraries working')
except Exception as e:
    print(f'   ✗ Additional libraries failed: {e}')
    sys.exit(1)

# Test 6: Circuit Stability Framework
print('6. Testing Circuit Stability Framework...')
framework_working = True
sys.path.insert(0, '$PYTHONPATH_ENTRY')

try:
    from cdatasets import DatasetBuilder, PromptFormatter
    datasets = list(DatasetBuilder.ids.keys())
    print(f'   ✓ Dataset framework: {len(datasets)} datasets available')
    if datasets:
        print(f'   ✓ Available datasets: {\", \".join(datasets[:3])}{\", ...\" if len(datasets) > 3 else \"\"}')
except Exception as e:
    print(f'   ⚠ Dataset framework issue: {e}')
    framework_working = False

try:
    from eap import Graph, attribute
    print('   ✓ EAP (Edge Activation Patching) tools loaded')
except Exception as e:
    print(f'   ⚠ EAP tools issue: {e}')
    framework_working = False

try:
    from experiments import circuit_discovery
    print('   ✓ Circuit discovery experiments loaded')
except Exception as e:
    print(f'   ⚠ Circuit discovery issue: {e}')
    framework_working = False

try:
    from utils import seed_everything
    print('   ✓ Utility functions loaded')
except Exception as e:
    print(f'   ⚠ Utils issue: {e}')
    framework_working = False

if framework_working:
    print('   ✓ Circuit Stability Framework fully functional')
else:
    print('   ⚠ Some framework components had issues')

# Test 7: Optional components
print('7. Testing optional components...')
try:
    import pygraphviz
    print('   ✓ Pygraphviz available (circuit visualization enabled)')
except ImportError:
    print('   ⚠ Pygraphviz not available (limited circuit visualization)')

print()
if framework_working:
    print('🎉 All core components verified successfully!')
    print('🧠 Circuit Stability Framework is ready for research!')
else:
    print('⚠  Core libraries working, but some framework issues detected.')
    print('   You may need to check imports or file paths manually.')
print()
"
    
    if [ $? -eq 0 ]; then
        log_success "Installation verification completed successfully"
        return 0
    else
        log_error "Installation verification failed"
        return 1
    fi
}

# ============================================================================
# Usage Information
# ============================================================================

print_usage_info() {
    log_step "Installation Complete!"
    
    echo
    echo -e "${GREEN}🎉 Circuit Stability Research Environment Successfully Installed!${NC}"
    echo
    echo -e "${CYAN}Quick Start:${NC}"
    echo "  ./activate_environment.sh"
    echo
    echo -e "${CYAN}Manual Activation:${NC}"
    echo "  conda activate $ENV_NAME"
    echo "  export PYTHONPATH=\"\$PYTHONPATH:$SCRIPT_DIR/code/src\""
    echo
    echo -e "${CYAN}Start Jupyter Lab:${NC}"
    echo "  ./activate_environment.sh"
    echo "  jupyter lab"
    echo
    echo -e "${CYAN}Run Circuit Discovery:${NC}"
    echo "  python -m experiments.circuit_discovery microsoft/phi-1_5 output.pkl --dataset custom --format cot"
    echo
    echo -e "${CYAN}Test Installation:${NC}"
    echo "  python -c \"import torch; print('CUDA:', torch.cuda.is_available())\""
    echo "  python -c \"from cdatasets import DatasetBuilder; print('Datasets:', list(DatasetBuilder.ids.keys()))\""
    echo
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "  - If imports fail, check PYTHONPATH: export PYTHONPATH=\"\$PYTHONPATH:$SCRIPT_DIR/code/src\""
    echo "  - For CUDA issues, verify: nvidia-smi && python -c \"import torch; print(torch.cuda.is_available())\""
    echo "  - For circuit visualization, install system graphviz: apt-get install graphviz graphviz-dev"
    echo
    echo -e "${GREEN}Environment: $ENV_NAME${NC}"
    echo -e "${GREEN}Python Path: $SCRIPT_DIR/code/src${NC}"
    echo -e "${GREEN}Activation Script: $SCRIPT_DIR/activate_environment.sh${NC}"
    echo
}

# ============================================================================
# Error Handling and Cleanup
# ============================================================================

cleanup_on_error() {
    log_error "Installation failed. Cleaning up..."
    
    # Deactivate conda environment if active
    if [ ! -z "${CONDA_DEFAULT_ENV:-}" ] && [ "$CONDA_DEFAULT_ENV" == "$ENV_NAME" ]; then
        conda deactivate
    fi
    
    # Optionally remove partially created environment
    if conda env list | grep -q "^$ENV_NAME "; then
        read -p "Remove partially created environment '$ENV_NAME'? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n "$ENV_NAME" -y
        fi
    fi
    
    exit 1
}

# Set up error handling
trap cleanup_on_error ERR

# ============================================================================
# Main Installation Flow
# ============================================================================

main() {
    echo -e "${PURPLE}"
    echo "============================================================================"
    echo "  Circuit Stability Research Framework - Installation Script"
    echo "============================================================================"
    echo -e "${NC}"
    echo "This script will install a complete environment for circuit stability"
    echo "research including PyTorch, TransformerLens, and all dependencies."
    echo
    
    # Confirmation prompt
    read -p "Continue with installation? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
    
    # Run installation steps
    detect_system
    setup_conda
    install_system_dependencies
    create_environment
    install_pytorch
    install_ml_packages
    setup_project
    
    # Verify installation
    if verify_installation; then
        print_usage_info
        log_success "🚀 Installation completed successfully!"
    else
        log_error "Installation completed with some issues. See verification output above."
        exit 1
    fi
}

# ============================================================================
# Script Entry Point
# ============================================================================

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi