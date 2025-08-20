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
├──v1_circuit.png, v2_circuit.png  # Circuit visualization files
├── tree-of-thought
      ├── MANIFEST.in
      ├── README.md
      ├── pyproject.toml
      ├── requirements.txt
      ├── run.py
      ├── scripts
      │    ├── crosswords
      │    ├── game24
      │    └── text    
      ├── setup.py
      └── src
          ├── tot
          │    ├── __init__.py
          │    ├── data
          │    │     ├── 24
          │    │     ├── crosswords
          │    │     └── text
          │    ├── hf_llm.py
          │    ├── methods
          │    │     └── bfs.py
          │    ├── models.py
          │    ├── prompts
          │    │     ├── crosswords.py
          │    │     ├── game24.py
          │    │     └── text.py
          │    ├── t_lens_generate.py
          │    └── tasks
          │        ├── __init__.py
          │        ├── base.py
          │        ├── crosswords.py
          │        ├── game24.py
          │        └── text.py

``` 

## Requirements and Dependencies

On runpod or mac you can also do.
```bash
git clone <this-repo>
cd JAKS

bash install.sh
```

1. Get hugginface access token here: https://huggingface.co/settings/tokens (for gated models) (https://transformerlensorg.github.io/TransformerLens/content/getting_started.html)

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

## Experimental Procedure

This section outlines the high-level experimental workflow for circuit discovery and stability analysis across thought variations. The procedure follows an iterative approach to generate, analyze, and optimize circuits for complex problem-solving tasks.

### 1. Input Prompt Preparation
- **Define the initial problem statement** that will serve as the foundation for all subsequent steps
- **Establish domain-specific constraints** and context that apply to the problem
- **Set backtrack signals** to identify which steps may need revision during the iterative process

### 2. Generate Step K
- **For k=1**: Prompt the LLM to propose the initial decomposition step of the problem
  - Current state includes partial solution from steps 1 to k-1 (variables, equations, conclusions)
  - Problem context maintains original problem statement and domain constraints
  - Backtrack signal identifies which step to revisit if needed
- **Output**: Natural language instruction for the current subtask

### 3. Generate (n) Variations of Step K
- **Create multiple semantic variations** of the step K output using different terminology and syntax
- **Preserve core problem-decomposition operations** while varying surface-level presentation
- **Generate user-defined number (n) of valid variations** that maintain logical equivalence
- **Input**: Step K output from previous stage
- **Output**: List of n semantically identical but linguistically diverse variations

### 4. Circuit Discovery on Thought Variations
- **Process each variation through circuit discovery pipeline** (`circuit_discovery.py`)
- **Use DataLoader** to feed individual variations into the analysis framework
- **Apply Edge Attribution Patching (EAP-IG)** to attribute nodes throughout the neural network
- **Extract top 200 nodes** ranked by attribution score from each variation
- **Export subgraphs** to JSON/PNG format for further analysis
- **Output**: JSON data containing circuit scores and optional PNG visualizations

### 5. Compute Stability Metrics
- **Analyze activation score distributions** for each prompt variation
  - Plot variations on distribution graphs (x-axis: activation scores, y-axis: node count)
  - Measure skewness using SciPy's skew function for each variation
  - Calculate similarity scores between variations and their previous steps
- **Repeat analysis** until stability metrics are computed for every variation
- **Output**: Skewness and similarity scores for each prompt variation

### 6. Store Circuit Data in Database
- **Create tree-like data structure** to organize all variations (`create_tree.py`)
  - Use Python nested dictionary for simplicity and navigation
  - Recursively store data per thought variation and parse into strings
  - Implement prefix trees with ID structure for easy backtracking and navigation
- **Generate JSON output file** starting with base query, prompt parameters at base level
- **Enable recursive nested subtasks** (k+1 inside subtask k) with unique IDs
- **Schema includes**:
  - `"id"`: Unique prefix for navigation
  - `"current_depth"`: Depth level starting at 1
  - `"text"`: LLM response at this level
  - `"children"`: Sub-task candidates and executed tasks
  - `"circuit_path"`: Path to circuit JSON for variation
  - `"score"`: Computed stability metrics for decision-making

### 7. Compute Optimal Circuit/Variation
- **Retrieve circuit stability metrics** (similarity, skewness) for each prompt variation
- **Apply optimization algorithm** with hyperparameters (alpha, beta)
  - For each thought variation: extract circuit stability metrics from dataset
  - Compute combined metric: `alpha × skewness - beta × similarity`
  - Continue until metrics are computed for all circuits of the variation
- **Output**: Maximum optimization score identifying the best circuit/variation

### 8. Decision Point: Solution Quality Check
- **Evaluate if current solution adequately solves the problem**
- **If solution is satisfactory**: Proceed to Final Answer (Step 9)
- **If solution needs improvement**: 
  - Check if all circuits are considered "poor quality"
  - If so, backtrack P steps and select second-best circuit
  - Otherwise, take another iteration step and return to Step 2
- **Continue iterative process** until circuits exist for all variations

### 9. Final Answer
- **Compile the complete solution** using the optimal circuit/variation identified
- **Present final results** based on the most stable and highest-scoring decomposition path

### Key Features of the Workflow:
- **Iterative refinement**: The procedure includes feedback loops for continuous improvement
- **Quality control**: Built-in checkpoints ensure solution adequacy before proceeding
- **Scalability**: User-defined variation counts (n) allow for experiment customization
- **Traceability**: Tree structure enables full backtracking of decision paths
- **Optimization**: Multi-metric optimization balances circuit stability and performance

## Citation

## Acknowledgments

- [EAP-IG](https://github.com/hannamw/eap-ig) for the base circuit discovery method
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for model analysis capabilities
- The broader mechanistic interpretability research community for foundational work 


