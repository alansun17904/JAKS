# Source Notebooks Documentation

Jupyter notebooks in `experiments/notebooks-source/` that analyze circuit stability - the consistency of neural network internal representations across similar tasks.

## Notebook Documentation

### Core Analysis Notebooks

#### `benchmark.ipynb` - Primary Analysis & Benchmarking Hub
Central analysis for circuit discovery benchmarking across multiple models and tasks.

**Key Experiments**:
- **Gemma-2-2b Addition Performance**: Analyzes accuracy on arithmetic problems (1-8 digits), showing exponential decay as complexity increases
- **Circuit Correlation Analysis**: Computes rank correlations between circuits for different arithmetic problems
- **Connected Components Analysis**: Groups similar circuits based on correlation thresholds
- **Cross-Pareto Analysis**: Studies circuit performance vs. component count
- **Boolean Expression Evaluation**: Benchmarks Phi-1.5 on boolean expressions

**Key Findings**:
- Arithmetic accuracy follows predictable decay (R² = 0.9978)
- Circuit rank correlations reveal distinct problem groupings
- Strong correlation between circuit complexity and task performance

#### `circuit-discovery.ipynb` - Core Circuit Analysis
Establishes foundational methodology for circuit discovery in arithmetic tasks.

**Key Experiments**:
- **Single-Digit Addition**: Studies memorization patterns in simple arithmetic
- **Activation Patching**: Implements logit difference metrics for circuit analysis
- **Token Analysis**: Analyzes token-by-token processing of arithmetic expressions

### Boolean Expression Analysis

#### `bool-no-paren.ipynb` - Boolean Expression Self-Consistency
Analyzes model self-consistency on boolean expressions with and without parentheses.

**Key Experiments**:
- **Self-Consistency Analysis**: Measures consistency across different presentations
- **Parentheses vs No-Parentheses**: Compares performance on different structural complexities
- **Subtask Performance**: Analyzes "Not", "Not+And", and "Not+And+Or" expression types

**Key Findings**:
- Parentheses significantly impact performance and consistency
- "Not+And" expressions show highest consistency (~75%)
- Pure "Not" expressions show lowest consistency (~48%)

#### `precedence.ipynb` - Structural Generalization Analysis
Studies how circuit stability relates to structural generalization in boolean expressions.

**Key Experiments**:
- **Parentheses Impact on Circuits**: Compares circuits for expressions with/without parentheses
- **Subtask Generalization**: Analyzes circuit transfer between task types
- **Cross-Correlation Analysis**: Measures circuit similarity across structural presentations

**Key Findings**:
- Parentheses improve circuit stability (correlation ~0.69 vs ~0.25)
- "Not" expressions show highest within-task but lowest cross-task correlation
- Structural complexity correlates with circuit stability

### Prompting Strategy Analysis

#### `prompting.ipynb` - Prompting Data Processing
Preprocesses and analyzes prompting experiment data with focus on response format standardization.

**Key Experiments**:
- **Response Length Analysis**: Studies distribution across prompting methods
- **Token Cleaning**: Removes special tokens and standardizes formats
- **Format Validation**: Ensures consistent parsing across strategies

#### `prompting-copy.ipynb` - Prompting Strategy Analysis
Analyzes different prompting strategies across multiple reasoning tasks.

**Key Experiments**:
- **Multi-Task Evaluation**: Tests Sports, Movie, Date, Dyck, and Common Sense reasoning
- **Prompting Strategy Comparison**: Compares zero-shot, few-shot (3-shot, 5-shot), and Chain-of-Thought
- **Response Processing**: Develops parsers for extracting answers from different formats

**Key Findings**:
- CoT prompting dramatically outperforms other methods (Sports: 78.8% vs ~10%)
- Performance varies significantly by task type
- Response preprocessing crucial for accurate evaluation

#### `prompting-effectiveness.ipynb` - Circuit Stability in Prompting
Examines relationship between prompting strategies and circuit stability.

**Key Experiments**:
- **Circuit Stability Measurement**: Calculates rank correlations between circuits across prompt instances
- **Cross-Model Analysis**: Compares Llama-3.1-8B and Gemma-2-9B on sports reasoning
- **Prompting Method Impact**: Analyzes how prompting affects circuit consistency

**Key Findings**:
- CoT prompting produces most stable circuits (correlation ~0.96)
- Higher circuit stability correlates with better task performance
- Model architecture influences prompting-stability relationship

### Validation and Visualization

#### `percent_token.ipynb` - Tokenization Validation
Validates tokenization consistency across different arithmetic problems.

**Key Experiments**:
- **Tokenization Pattern Analysis**: Checks consistency across arithmetic expressions
- **Multi-digit Problem Validation**: Tests tokenization for 1-8 digit problems
- **Format Consistency**: Ensures arithmetic problems follow expected patterns

**Key Findings**: 100% tokenization consistency across all tested formats, validating experimental setup.

#### `single-digit-viz.ipynb` - Circuit Visualization
Creates visualizations for circuits discovered in single-digit arithmetic tasks.

**Key Experiments**:
- **Circuit Network Visualization**: Generates graph representations of discovered circuits
- **Cross-Task Correlation**: Computes and visualizes correlations between arithmetic circuits
- **Dimensionality Reduction**: Uses t-SNE to visualize circuit similarity

**Key Findings**:
- Similar arithmetic problems cluster together in embedding space
- Visual analysis reveals clear circuit organization patterns
- Correlation analysis shows distinct arithmetic circuit families

## Research Impact

This research provides:

1. **Theoretical Insights**: How large language models develop consistent internal strategies for related problems
2. **Methodological Tools**: Techniques for analyzing and visualizing neural network internal mechanisms
3. **Empirical Evidence**: Circuit stability is measurable and correlates with task performance
4. **Practical Applications**: Understanding how prompting strategies affect internal representations

## Dependencies

### Required Files

#### `alan.mplstyle` - Matplotlib Style Sheet
Standardizes plot appearance across all notebooks for consistent visualizations.

**Settings**:
- Font size: 12pt
- Tick sizes: Major (4pt), Minor (2pt)
- Axis line width: 1.5
- Grid: Solid black lines with full opacity
- Colors: White background, black axis edges

**Used by**: 7 notebooks (`benchmark.ipynb`, `bool-no-paren.ipynb`, `precedence.ipynb`, `prompting.ipynb`, `prompting-copy.ipynb`, `prompting-effectiveness.ipynb`, `single-digit-viz.ipynb`)

**Why needed**: Ensures consistent plot styling across research. Without it, notebooks revert to matplotlib defaults.

#### `circuit_visual.py` - Circuit Visualization Module
Provides specialized functions for visualizing neural circuit structures and correlations.

**Key Functions**:
- Circuit network graph generation
- Correlation matrix visualization
- t-SNE dimensionality reduction for circuit embeddings
- Custom plotting utilities for circuit analysis

**Used by**: `single-digit-viz.ipynb` (imported as `circuit_visual`)

**Why needed**: Contains domain-specific visualization functions not available in standard libraries. Essential for creating circuit diagrams and correlation plots.

### External Libraries

Key libraries include:
- `matplotlib` and `seaborn` for visualization
- `numpy` and `pandas` for data analysis
- `transformer_lens` for circuit analysis
- Custom utilities from the project's `code/` directory

## Usage

Notebooks should be run in order of dependencies:
1. Start with `circuit-discovery.ipynb` for methodology
2. Use `percent_token.ipynb` to validate setup
3. Run `benchmark.ipynb` for main analysis
4. Explore specific aspects with remaining notebooks
5. Use `single-digit-viz.ipynb` for visualization

Each notebook generates PDF visualizations saved to the parent directory for publication-ready figures.