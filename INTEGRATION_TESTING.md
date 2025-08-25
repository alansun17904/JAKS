# Testing the ToT-Circuit Stability Integration

This document explains how to test the integration between the Tree-of-Thought (ToT) method and circuit stability analysis pipeline. The integration allows ToT to use circuit analysis scores when selecting which reasoning paths to pursue.

## Quick Start

Run the integration test:

```bash
python test_integration.py
```

This simulates the complete pipeline and validates that all components work together.

## How the Integration Works

### Overview

The integration connects two main components:

1. **Tree-of-Thought (ToT)**: Generates multiple reasoning steps and needs to evaluate which ones are most promising
2. **Circuit Stability Pipeline**: Analyzes neural circuits and produces stability metrics for different reasoning variations

### Data Flow

1. ToT generates reasoning variations for a problem (e.g., "1 + 1 = 2", "1 * 4 = 4")
2. These variations get saved as JSON files in `circuit-stability/code/src/cdatasets/data/`
3. The circuit analysis pipeline runs on each variation
4. Circuit scores are saved as `*_score.json` files
5. ToT reads these scores via `get_circuit_scores()` function
6. Scores influence which reasoning paths ToT explores next

### Key Files

- `test_integration.py`: Full integration test that simulates the entire pipeline
- `test_circuit_scores.py`: Tests just the score parsing functionality
- `tree-of-thought/src/tot/methods/bfs.py`: Where ToT calls `get_circuit_scores()` (line 112)
- `circuit-stability/code/src/experiments/circuit_discovery.py`: Generates circuit analysis scores

## Understanding the Metrics

### circuit_loss

`circuit_loss` is a placeholder for metric. Not really sure what we decided on implementing. This is just what LLMs came up with.

**Calculation:**
- Baseline: Full model performance on the task
- Circuit Performance: Performance using only the discovered circuit  
- `circuit_loss = circuit_performance - baseline_performance`

**Interpretation:**
- Lower values = better circuits (closer to full model performance)
- `circuit_loss ≈ 0`: Circuit preserves most model capability
- Higher values: Significant performance loss when using just the circuit

### Score Transformation

Circuit losses get converted to ToT selection scores using this mapping:
- Perfect circuits (`circuit_loss ≤ 0`) → score = 20
- Poor circuits (`circuit_loss ≥ 50000`) → score = 0.001
- Everything else gets linearly interpolated

Note, the ranges (25000-50000) only serve as a proof-of-concept. They are artbitrarily based on observed values and temporary metrics.

## Design Choices

### Why JSON Files for Communication?

We use JSON files to pass data between ToT and circuit analysis because:
- **Decoupling**: Each component can run independently and at different times
- **Debugging**: Easy to inspect intermediate results
- **Flexibility**: Simple to modify data formats as research evolves
- **Robustness**: If one component fails, the other can continue

### Score Range Mapping

ToT expects scores in the range [0.001, 20] to match its existing heuristic scoring. We map circuit metrics to this range to maintain compatibility with the existing ToT codebase.

### File Naming Convention

Score files use the pattern: `{thought_index}_th_thought_{variation_index}_th_variation_score.json`

This allows ToT to easily locate the score for any specific reasoning variation.

## Running Individual Components

### Test Just Score Parsing

```bash
python test_circuit_scores.py
```

### Test Just Circuit Analysis

```bash
cd circuit-stability/code
bash src/scripts/naive_run.sh --data_params data_file=1,1,4,6.json
```

### Test ToT Without Circuit Scores

```bash
cd tree-of-thought
python run.py --task game24 --task_start_index 900 --task_end_index 910 --method_generate propose --method_evaluate value --method_select greedy --n_generate_sample 1 --prompt_sample cot
```

## Expected Outputs

When the integration test runs successfully, you should see:

1. **ToT Data Creation**: Confirmation that sample reasoning variations were created
2. **Circuit Analysis**: Output from running the stability analysis
3. **Score Parsing**: Demonstration of converting circuit metrics to ToT scores
4. **Thought Selection**: Ranking of reasoning variations by circuit quality

## Troubleshooting

### Common Issues

**"No circuit score files found"**: The circuit analysis didn't run or failed. Check the circuit-stability environment setup.

**Environment errors**: Make sure both ToT and circuit-stability environments are properly configured. The integration test expects to run from the main workspace directory.

**Score file parsing errors**: Usually indicates a change in the JSON format. Check that `circuit_discovery.py` is generating the expected score format.

## Future Extensions

This integration is designed to be extensible. As our research progresses, we can:

- Add new circuit quality metrics beyond `circuit_loss`
- Implement more sophisticated score transformation functions
- Support different data exchange formats
- Add real-time communication between components

The modular design makes it easy to swap out individual components or metrics without affecting the overall integration.