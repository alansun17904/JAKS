# Circuit Stability Installation Script for Windows
# This script automates the environment setup and code fixes

Write-Host "Circuit Stability - Windows Installation Script" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Check if Conda is installed
$condaPath = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaPath) {
    Write-Host "❌ Error: Conda not found. Please install Anaconda or Miniconda first." -ForegroundColor Red
    Write-Host "Download from: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Yellow
    exit 1
}

# Check if Git is installed
$gitPath = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitPath) {
    Write-Host "❌ Error: Git not found. Please install Git first." -ForegroundColor Red
    Write-Host "Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Check if Graphviz is installed
$dotPath = Get-Command dot -ErrorAction SilentlyContinue
if (-not $dotPath) {
    Write-Host "⚠️ Warning: Graphviz not found. Some visualizations may not work." -ForegroundColor Yellow
    Write-Host "Download from: https://graphviz.org/download/" -ForegroundColor Yellow
    Write-Host "During installation, make sure to select 'Add Graphviz to system PATH'" -ForegroundColor Yellow
    $continue = Read-Host "Do you want to continue without Graphviz? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}

Write-Host "✅ Prerequisites check passed!" -ForegroundColor Green

# Create and activate the conda environment
Write-Host "`n1. Creating conda environment 'ml' with Python 3.10..." -ForegroundColor Cyan
conda create -n ml python=3.10 -y

# Make the script work with conda activate
$condaPrefix = (& conda info --base)
$activateScript = Join-Path $condaPrefix "Scripts\activate.ps1"
. $activateScript

# Activate the environment
conda activate ml

# Install PyTorch with CUDA
Write-Host "`n2. Installing PyTorch with CUDA support..." -ForegroundColor Cyan
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
Write-Host "`n3. Installing other dependencies..." -ForegroundColor Cyan
python -m pip install transformer-lens transformers huggingface-hub
conda install -c conda-forge pygraphviz -y
python -m pip install typeguard matplotlib jupyter tqdm

# Verify CUDA support
Write-Host "`n4. Verifying CUDA support..." -ForegroundColor Cyan
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"


Write-Host "`n✅ Installation completed successfully!" -ForegroundColor Green
Write-Host "You can now run experiments with: ./run_experiment.ps1" -ForegroundColor Green
Write-Host "`nTo activate the environment in the future, use: conda activate ml" -ForegroundColor Yellow 