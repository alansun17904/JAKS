#!/bin/bash

# Tree of Thoughts - Complete Installation Script
# This script installs miniconda, creates a virtual environment, and installs all dependencies

set -e  # Exit on any error

echo "🌲 Tree of Thoughts Installation Script 🌲"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="tot_env"
PYTHON_VERSION="3.11"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is already installed
check_conda() {
    if command -v conda &> /dev/null; then
        print_success "Conda is already installed"
        return 0
    else
        print_status "Conda not found, will install miniconda"
        return 1
    fi
}

# Install miniconda
install_miniconda() {
    print_status "Installing Miniconda..."
    
    # Download miniconda
    if [ ! -f "$MINICONDA_INSTALLER" ]; then
        print_status "Downloading Miniconda installer..."
        wget -q "$MINICONDA_URL" -O "$MINICONDA_INSTALLER"
    else
        print_status "Miniconda installer already exists"
    fi
    
    # Install miniconda
    print_status "Running Miniconda installer..."
    bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
    
    # Add conda to PATH
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    # Initialize conda
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    
    # Accept conda terms of service to avoid interactive prompts
    conda config --set channel_priority strict
    conda config --set auto_activate_base false
    
    print_success "Miniconda installed successfully"
}

# Create conda environment
create_environment() {
    print_status "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
    
    # Accept conda-forge terms of service
    conda config --add channels conda-forge
    conda config --set channel_priority flexible
    
    # Check if environment already exists
    if conda env list | grep -q "^$ENV_NAME "; then
        print_warning "Environment $ENV_NAME already exists. Removing it..."
        conda env remove -n "$ENV_NAME" -y
    fi
    
    # Create new environment using conda-forge to avoid TOS issues
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -c conda-forge -y
    print_success "Environment $ENV_NAME created"
}

# Install PyTorch with CUDA support
install_pytorch() {
    print_status "Installing PyTorch with CUDA support..."
    
    # Activate environment
    conda activate "$ENV_NAME"
    
    # Try installing PyTorch via pip to avoid conda channel issues
    print_status "Installing PyTorch via pip..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    print_success "PyTorch installed"
}

# Install core dependencies via conda
install_conda_deps() {
    print_status "Installing core dependencies via conda..."
    
    conda activate "$ENV_NAME"
    
    # Install scientific computing packages via conda-forge (better compatibility)
    conda install -y -c conda-forge \
        numpy=1.24.3 \
        pandas=2.0.3 \
        sympy=1.13.1 \
        requests=2.32.2 \
        tqdm=4.66.3 \
        certifi=2023.5.7
        
    print_success "Core conda dependencies installed"
}

# Install remaining dependencies via pip
install_pip_deps() {
    print_status "Installing remaining dependencies via pip..."
    
    conda activate "$ENV_NAME"
    
    # Create a clean requirements file without conda-managed packages
    cat > temp_requirements.txt << EOF
aiohttp==3.8.4
aiosignal==1.3.1
async-timeout==4.0.2
attrs==23.1.0
backoff==2.2.1
charset-normalizer==3.1.0
frozenlist==1.3.3
idna==3.4
mpmath==1.3.0
multidict==6.0.4
openai==0.27.7
urllib3==2.0.2
yarl==1.9.2
huggingface_hub==0.30.0
transformers==4.51
python-dotenv==1.0.0
transformer_lens
jaxtyping
einops
EOF
    
    # Install via pip
    pip install -r temp_requirements.txt
    
    # Clean up
    rm temp_requirements.txt
    
    print_success "Pip dependencies installed"
}

# Install the ToT package itself
install_tot_package() {
    print_status "Installing Tree of Thoughts package..."
    
    conda activate "$ENV_NAME"
    
    # Install in development mode
    pip install -e .
    
    print_success "ToT package installed"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    conda activate "$ENV_NAME"
    
    # Test imports
    python -c "
import torch
import numpy as np
import pandas as pd
import transformers
import transformer_lens
from tot.tasks import get_task
from tot.models import gpt
print('✅ All imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'TransformerLens version: {transformer_lens.__version__}')
"
    
    print_success "Installation verification complete!"
}

# Create activation script
create_activation_script() {
    print_status "Creating activation script..."
    
    cat > activate_tot.sh << EOF
#!/bin/bash
# Activation script for Tree of Thoughts environment

# Add miniconda to PATH if needed
export PATH="\$HOME/miniconda3/bin:\$PATH"

# Initialize conda
source "\$HOME/miniconda3/etc/profile.d/conda.sh"

# Activate environment
conda activate $ENV_NAME

echo "🌲 Tree of Thoughts environment activated!"
echo "Python version: \$(python --version)"
echo "Environment: \$CONDA_DEFAULT_ENV"
EOF
    
    chmod +x activate_tot.sh
    print_success "Activation script created: ./activate_tot.sh"
}

# Main installation flow
main() {
    print_status "Starting Tree of Thoughts installation..."
    
    # Install miniconda if needed
    if ! check_conda; then
        install_miniconda
        # Re-source conda after installation
        export PATH="$HOME/miniconda3/bin:$PATH"
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
    
    # Create environment and install dependencies
    create_environment
    install_pytorch
    install_conda_deps
    install_pip_deps
    install_tot_package
    
    # Verify everything works
    verify_installation
    
    # Create activation script
    create_activation_script
    
    print_success "🎉 Installation complete!"
    echo ""
    echo "To activate the environment, run:"
    echo "    source ./activate_tot.sh"
    echo ""
    echo "Or manually:"
    echo "    conda activate $ENV_NAME"
    echo ""
    echo "To test the installation, try:"
    echo "    python -c 'from tot.tasks import get_task; print(\"ToT ready!\")'"
}

# Run main function
main "$@"