#!/bin/bash
echo "Installing Energy Market Optimization..."
echo

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo
echo "Installing package in development mode..."
python install_dev.py

echo
echo "Installation complete!"
echo "You can now run experiments with:"
echo "  python experiments/run_experiment.py"
echo "  python experiments/agent_comparison.py"
echo "  python experiments/hyperparameter_tuning.py"
echo 