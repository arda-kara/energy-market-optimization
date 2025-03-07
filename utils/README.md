# Utilities for Energy Market Optimization

This directory contains utility functions for the Energy Market Optimization project.

## Visualization

The `visualization.py` module provides functions for visualizing training results and agent comparisons:

- `plot_training_curves`: Plots training curves for a single agent, including rewards, losses, and other metrics.
- `plot_comparison`: Plots a comparison of a specific metric across different agents.
- `plot_comprehensive_comparison`: Plots a comprehensive comparison of multiple metrics across different agents.

### Visualization Script

The `visualize_results.py` script can be used to generate visualizations from existing experiment results:

```bash
python -m utils.visualize_results --results-dir path/to/results --output-dir path/to/output
```

Arguments:
- `--results-dir`: Directory containing experiment results (required)
- `--output-dir`: Directory to save visualizations (defaults to results directory)
- `--metrics`: Metrics to compare in individual comparison plots (default: rewards, policy_loss, value_loss)

## Data Processing

The `data_processing.py` module provides functions for processing data, including:

- `create_results_directory`: Creates a directory for storing experiment results.
- `load_data`: Loads data from a file.
- `save_data`: Saves data to a file.

## Debugging

The project includes several debugging utilities to help identify and fix issues:

1. **TensorBoard Integration**: Use the `--use-tensorboard` flag with `run_experiment.py` to enable TensorBoard logging.
2. **Improved Error Handling**: Better error messages for dimension mismatches and other common issues.
3. **Visualization Tools**: Enhanced visualization tools for analyzing training results.

## Common Issues and Solutions

### Empty Plots

If you see empty plots in the visualization:

1. Check that the agent is correctly storing metrics during training.
2. Ensure the metric names are consistent between agents (e.g., 'episode_rewards' vs 'episode_returns').
3. Use the `visualize_results.py` script to regenerate visualizations with the latest improvements.

### Missing Metrics

If certain metrics are missing from the plots:

1. Check the agent implementation to ensure it's calculating and storing those metrics.
2. Verify that the visualization functions are looking for the correct metric names.
3. Use the `plot_comprehensive_comparison` function for a more complete view of all available metrics. 