import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch

from rl_environment.energy_env import EnergyMarketEnv
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from utils.visualization import plot_training_curves, plot_comparison, plot_comprehensive_comparison
from utils.data_processing import create_results_directory

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run RL experiments on Energy Market Environment')
    
    parser.add_argument('--agents', nargs='+', default=['ppo', 'sac'], 
                        choices=['ppo', 'sac'], help='Agents to train')
    parser.add_argument('--timesteps', type=int, default=100000, 
                        help='Total timesteps for training')
    parser.add_argument('--eval-episodes', type=int, default=10, 
                        help='Number of episodes for final evaluation')
    parser.add_argument('--episode-length', type=int, default=24*7, 
                        help='Episode length in hours')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--use-tensorboard', action='store_true',
                        help='Enable TensorBoard logging')
    
    return parser.parse_args()

def run_experiment(args):
    """Run experiment with specified agents."""
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = create_results_directory(f"experiment_{timestamp}")
    
    # Create environment
    env = EnergyMarketEnv(
        episode_length=args.episode_length,
        normalize_state=True
    )
    
    # Print environment information
    print(f"Observation space: {env.observation_space}")
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    
    # Dictionary to store results
    results = {
        "metrics": {},
        "eval_returns": {},
        "config": vars(args)
    }
    
    # Setup TensorBoard if enabled
    tb_writer = None
    if args.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_log_dir = os.path.join("logs", "tensorboard", f"experiment_{timestamp}")
            os.makedirs(tb_log_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_log_dir)
            print(f"TensorBoard logging enabled. Log directory: {tb_log_dir}")
        except ImportError:
            print("Warning: TensorBoard not available. Install with 'pip install tensorboard'")
    
    # Train and evaluate agents
    for agent_name in args.agents:
        if agent_name == "ppo":
            print("\n=== Training PPO Agent ===")
            
            # Create PPO agent
            ppo_agent = PPOAgent(
                env=env,
                hidden_dim=256,
                lr=3e-4,
                gamma=0.99,
                lam=0.95,
                clip_ratio=0.2,
                entropy_coef=0.01,
                value_coef=0.5,
                max_grad_norm=0.5
            )
            
            # Train PPO agent
            ppo_metrics = ppo_agent.train(
                env=env,
                total_timesteps=args.timesteps,
                buffer_size=2048,
                batch_size=64,
                update_epochs=10,
                num_eval_episodes=5,
                eval_freq=args.timesteps // 20,
                save_freq=args.timesteps // 5,
                save_path=os.path.join(output_dir, "ppo_model"),
                tb_writer=tb_writer
            )
            
            # Plot training curves
            plot_training_curves(
                ppo_metrics, 
                os.path.join(output_dir, "ppo_training_curves.png"),
                title="PPO Training Curves"
            )
            
            # Evaluate PPO agent
            print("\nEvaluating PPO agent...")
            ppo_return = ppo_agent.evaluate(env, num_episodes=args.eval_episodes)
            print(f"PPO final evaluation return: {ppo_return:.2f}")
            
            # Store results
            results["metrics"]["ppo"] = ppo_metrics
            results["eval_returns"]["ppo"] = ppo_return
            
        elif agent_name == "sac":
            print("\n=== Training SAC Agent ===")
            
            # Create SAC agent
            sac_agent = SACAgent(
                env=env,
                hidden_dim=256,
                lr=3e-4,
                gamma=0.99,
                tau=0.005,
                alpha=0.2,
                auto_alpha=True
            )
            
            # Train SAC agent
            sac_metrics = sac_agent.train(
                env=env,
                total_timesteps=args.timesteps,
                batch_size=256,
                buffer_size=100000,
                update_after=1000,
                update_every=50,
                num_eval_episodes=5,
                eval_freq=args.timesteps // 20,
                save_freq=args.timesteps // 5,
                save_path=os.path.join(output_dir, "sac_model"),
                tb_writer=tb_writer
            )
            
            # Plot training curves
            plot_training_curves(
                sac_metrics, 
                os.path.join(output_dir, "sac_training_curves.png"),
                title="SAC Training Curves"
            )
            
            # Evaluate SAC agent
            print("\nEvaluating SAC agent...")
            sac_return = sac_agent.evaluate(env, num_episodes=args.eval_episodes)
            print(f"SAC final evaluation return: {sac_return:.2f}")
            
            # Store results
            results["metrics"]["sac"] = sac_metrics
            results["eval_returns"]["sac"] = sac_return
    
    # Compare agents if multiple agents were trained
    if len(args.agents) > 1:
        print("\nComparing agents...")
        
        # Plot comprehensive comparison
        plot_comprehensive_comparison(
            results["metrics"],
            os.path.join(output_dir, "agent_comprehensive_comparison.png"),
            title="Comprehensive Agent Comparison"
        )
        
        # Also plot individual metric comparisons
        metrics_to_compare = ['rewards', 'policy_loss', 'value_loss']
        for metric in metrics_to_compare:
            plot_comparison(
                results["metrics"],
                metric,
                os.path.join(output_dir, f"agent_comparison_{metric}.png"),
                title=f"Agent Comparison - {metric.capitalize()}"
            )
        
        # Save comparison results
        with open(os.path.join(output_dir, "comparison_results.txt"), "w") as f:
            f.write("Agent Comparison Results:\n")
            f.write("========================\n\n")
            
            for agent_name, eval_return in results["eval_returns"].items():
                f.write(f"{agent_name.upper()} final evaluation return: {eval_return:.2f}\n")
            
            if "ppo" in results["eval_returns"] and "sac" in results["eval_returns"]:
                diff = results["eval_returns"]["sac"] - results["eval_returns"]["ppo"]
                f.write(f"\nDifference (SAC - PPO): {diff:.2f}\n")
                
                if diff > 0:
                    f.write("SAC performed better than PPO\n")
                elif diff < 0:
                    f.write("PPO performed better than SAC\n")
                else:
                    f.write("PPO and SAC performed equally\n")
    
    print(f"\nResults saved to {output_dir}")
    return results

def main():
    """Main function."""
    args = parse_args()
    results = run_experiment(args)
    
    # Print final results
    print("\nFinal Results:")
    for agent_name, eval_return in results["eval_returns"].items():
        print(f"{agent_name.upper()} final evaluation return: {eval_return:.2f}")

if __name__ == "__main__":
    main() 