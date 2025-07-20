# JAKS Research Repository
This repository contains research on circuit stability in neural networks, focusing on problem decomposition and automated plan creation. The work investigates how stable circuit patterns emerge in transformer models and their applications to systematic problem-solving tasks. 

## Repository Structure

```
├── circuit-stability/           # Circuit stability research (published work)
│   ├── code/
│   │   ├── src/                # Source code
│   │   │   ├── cdatasets/      # Dataset implementations
│   │   │   ├── eap/           # Edge Attribution Patching
│   │   │   ├── experiments/    # Experiment scripts
│   │   │   └── scripts/        # Shell scripts for experiments
│   │   ├── tests/             # Test files
│   │   └── compare_circuits.py # Circuit comparison tools
│   ├── data/                  # Pickled data files
│   ├── documentation/         # Documentation files
│   ├── figures/               # Paper figures and automated outputs
│   ├── notebooks-source/      # Jupyter notebooks and config
│   │   └── config/            # Environment & setup files
│   └── results/               # JSON/pickle output files
├── test.ipynb                 # Test notebook
└── v1_circuit.png, v2_circuit.png  # Circuit visualization files
``` 

## Requirements and Dependencies

The circuit stability codebase relies on a `conda` environment whose dependencies are listed in `circuit-stability/notebooks-source/config/environment.yml`. Before installing this environment, you need to install `Graphviz` (required for circuit visualization). To make things easy, you can directly run 

```bash
git clone <this-repo>
cd JAKS
cd circuit-stability/notebooks-source/config
./install.sh
conda activate ml
```

**Below are instructions for manual installation.**

**macOS:**
```bash
brew install graphviz
```
**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install graphviz
```
**Windows:**
Download and install from [https://graphviz.org/download/](https://graphviz.org/download/)

### Python Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <this-repo>
   cd JAKS
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda env create -f circuit-stability/notebooks-source/config/environment.yml
   conda activate ml
   ```

3. **Verify the installation:**
   ```bash
   python -c "import torch; import transformer_lens; import pygraphviz; print('All dependencies installed successfully!')"
   ```

### Alternative Installation (if conda is not available)

If you don't have conda installed, you can use pip with a virtual environment:

```bash
# Create virtual environment
python -m venv 

# Activate virtual environment
# On macOS/Linux:
source cs-env/bin/activate
# On Windows:
cs-env\Scripts\activate

# Install dependencies (this will take some time)
pip install torch==2.4.1
pip install transformers==4.44.2
pip install transformer-lens==2.11.0
pip install pygraphviz==1.14
pip install -r <(conda env export -f circuit-stability/notebooks-source/config/environment.yml | grep "pip:" -A 1000 | tail -n +2 | sed 's/^      - //')
```

### Computational Requirements

Case Study I and Case Study II can be run on a single NVIDIA A100 80GB GPU. However, Case Study III needs two of the aforementioned GPUs.


## Citation

## Acknowledgments

- [EAP-IG](https://github.com/hannamw/eap-ig) for the base circuit discovery method
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for model analysis capabilities
- The broader mechanistic interpretability research community for foundational work 


