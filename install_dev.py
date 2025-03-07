#!/usr/bin/env python
"""
Script to install the package in development mode.
This allows you to modify the code and have the changes take effect immediately.
"""

import subprocess
import sys
import os

def main():
    """Install the package in development mode."""
    print("Installing Energy Market Optimization in development mode...")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run pip install -e .
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", current_dir])
        print("\nInstallation successful!")
        print("You can now import modules from the project from anywhere.")
        print("For example: from rl_environment.energy_env import EnergyMarketEnv")
    except subprocess.CalledProcessError as e:
        print(f"\nInstallation failed with error: {e}")
        sys.exit(1)
    
    print("\nTo run experiments, use:")
    print("  python experiments/run_experiment.py")
    print("  python experiments/agent_comparison.py")
    print("  python experiments/hyperparameter_tuning.py")

if __name__ == "__main__":
    main() 