import os
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.visualization import plot_training_curves, plot_comparison, plot_comprehensive_comparison

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize RL experiment results')
    
    parser.add_argument('--results-dir', type=str, required=True, 
                        help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save visualizations (defaults to results directory)')
    parser.add_argument('--metrics', nargs='+', default=['rewards', 'policy_loss', 'value_loss'],
                        help='Metrics to compare (for individual comparisons)')
    
    return parser.parse_args()

def load_agent_metrics(results_dir):
    """Load agent metrics from pickle files."""
    agent_metrics = {}
    
    # Look for PPO metrics
    ppo_path = os.path.join(results_dir, 'ppo_metrics.pkl')
    if os.path.exists(ppo_path):
        with open(ppo_path, 'rb') as f:
            agent_metrics['ppo'] = pickle.load(f)
        print(f"Loaded PPO metrics from {ppo_path}")
    
    # Look for SAC metrics
    sac_path = os.path.join(results_dir, 'sac_metrics.pkl')
    if os.path.exists(sac_path):
        with open(sac_path, 'rb') as f:
            agent_metrics['sac'] = pickle.load(f)
        print(f"Loaded SAC metrics from {sac_path}")
    
    return agent_metrics

def visualize_results(args):
    """Visualize experiment results."""
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load agent metrics
    agent_metrics = load_agent_metrics(args.results_dir)
    
    if not agent_metrics:
        print("No agent metrics found in the specified directory.")
        return
    
    # Plot individual agent training curves
    for agent_name, metrics in agent_metrics.items():
        plot_training_curves(
            metrics,
            os.path.join(output_dir, f"{agent_name}_training_curves.png"),
            title=f"{agent_name.upper()} Training Curves"
        )
        print(f"Generated {agent_name.upper()} training curves")
    
    # Plot agent comparisons if multiple agents
    if len(agent_metrics) > 1:
        # Comprehensive comparison
        plot_comprehensive_comparison(
            agent_metrics,
            os.path.join(output_dir, "agent_comprehensive_comparison.png"),
            title="Comprehensive Agent Comparison"
        )
        print("Generated comprehensive agent comparison")
        
        # Individual metric comparisons
        for metric in args.metrics:
            plot_comparison(
                agent_metrics,
                metric,
                os.path.join(output_dir, f"agent_comparison_{metric}.png"),
                title=f"Agent Comparison - {metric.capitalize()}"
            )
            print(f"Generated agent comparison for {metric}")
    
    print(f"All visualizations saved to {output_dir}")

def main():
    """Main function."""
    args = parse_args()
    visualize_results(args)

if __name__ == "__main__":
    main() 