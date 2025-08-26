# Circuit Stability Figures

`figures/` contains figures that show that similar tasks produce similar neural circuits across different models and conditions.

## Structure

```
figures/
├── notebook-outputs/            # PDFs from Jupyter analysis (51 files, 6 categories)
├── automated-outputs/           # Script-generated PNGs (9 files, 3 categories)
└── circuit-stability-cover.png  # Main research figure
```

## `notebook-outputs/`
PDFs generated from source notebooks, organized into 6 categories:

`benchmarks/` (8 files) Performance heatmaps showing model accuracy across arithmetic and boolean tasks. Establishes baseline performance before circuit analysis.

`connected-components/` (25 files)
Core research evidence: Network visualizations showing how circuits cluster at different correlation thresholds (0.3-0.83). Proves similar tasks produce similar circuits.

`correlation-analysis/` (3 files)
Statistical correlation matrices quantifying circuit similarity. Provides mathematical rigor for circuit equivalence claims.

`generalization-analysis/` (5 files)
Studies how circuits transfer between related tasks. Proves circuits represent generalizable computational strategies, not just memorization.

`prompting-studies/` (5 files)
Comparative analysis of prompting strategies' effects on circuits. Shows that both performance and internal representations change with prompting.

`statistical-analysis/` (5 files)
Regression analyses and significance testing. Ensures discovered patterns are statistically meaningful, not random artifacts.

## `automated-outputs/` 
Script-generated PNGs for quick analysis during development:
- `circuit-analysis/` (3 files): Direct circuit structure visualizations
- `comparisons/` (3 files): Comparative analysis between experimental conditions  
- `distributions/` (3 files): Statistical distribution and ranking visualizations

## `circuit-stability-cover.png` 
Main research figure: 9-panel visualization showing circuit evolution across correlation thresholds. Demonstrates that circuits cluster predictably at different similarity levels.

## Naming Conventions
Files generally use consistent patterns: `{model}-{task}-{analysis}-{parameters}.{format}`
- Models: gemma, phi, llama
- Tasks: bool, add, connected 
- Parameters: correlation thresholds, complexity levels