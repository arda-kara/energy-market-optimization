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
from utils.visualization import plot_training_curves, plot_comparison
from utils.data_processing import create_results_directory

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare RL agents on Energy Market Environment')
    
    parser.add_argument('--agents', nargs='+', default=['ppo', 'sac'], 
                        choices=['ppo', 'sac'], help='Agents to compare')
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

def train_and_evaluate_agents(config):
    """
    Train and evaluate multiple agents with the given configuration.
    
    Args:
        config (dict): Configuration dictionary with training parameters
        
    Returns:
        dict: Dictionary with training metrics and evaluation results
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = create_results_directory(f"agent_comparison_{timestamp}")
    
    # Create environment
    print("\n=== Creating Environment ===")
    env = EnergyMarketEnv(
        episode_length=config.get("episode_length", 24*7),  # Default: 1 week
        normalize_state=config.get("normalize_state", True)
    )
    print(f"Environment created with episode length: {config.get('episode_length', 24*7)} hours")
    print(f"State dimension: {env.observation_space.shape[0]}, Action dimension: {env.action_space.shape[0]}")
    
    # Dictionary to store results
    results = {
        "metrics": {},
        "eval_returns": {},
        "config": config
    }
    
    # Setup TensorBoard if enabled
    tb_writer = None
    if config.get("use_tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_log_dir = os.path.join("logs", "tensorboard", f"agent_comparison_{timestamp}")
            os.makedirs(tb_log_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_log_dir)
            print(f"TensorBoard logging enabled. Log directory: {tb_log_dir}")
        except ImportError:
            print("Warning: TensorBoard not available. Install with 'pip install tensorboard'")
    
    # Train and evaluate PPO agent
    if "ppo" in config["agents"]:
        print("\n=== Training PPO Agent ===")
        ppo_config = config["agents"]["ppo"]
        
        print(f"PPO Configuration:")
        for key, value in ppo_config.items():
            print(f"  {key}: {value}")
        
        print("\nInitializing PPO agent...")
        ppo_agent = PPOAgent(
            env=env,
            hidden_dim=ppo_config.get("hidden_dim", 256),
            lr=ppo_config.get("lr", 3e-4),
            gamma=ppo_config.get("gamma", 0.99),
            lam=ppo_config.get("lam", 0.95),
            clip_ratio=ppo_config.get("clip_ratio", 0.2),
            entropy_coef=ppo_config.get("entropy_coef", 0.01),
            value_coef=ppo_config.get("value_coef", 0.5),
            max_grad_norm=ppo_config.get("max_grad_norm", 0.5)
        )
        
        print(f"\nStarting PPO training for {config.get('total_timesteps', 100000)} timesteps...")
        ppo_metrics = ppo_agent.train(
            env=env,
            total_timesteps=config.get("total_timesteps", 100000),
            buffer_size=ppo_config.get("buffer_size", 2048),
            batch_size=ppo_config.get("batch_size", 64),
            update_epochs=ppo_config.get("update_epochs", 10),
            num_eval_episodes=config.get("num_eval_episodes", 5),
            eval_freq=config.get("eval_freq", 5000),
            save_freq=config.get("save_freq", 10000),
            save_path=os.path.join(output_dir, "ppo_model"),
            tb_writer=tb_writer
        )
        
        print("\nPPO training completed")
        print(f"Saving training curves to {os.path.join(output_dir, 'ppo_training_curves.png')}")
        
        # Plot PPO training metrics
        plot_training_curves(ppo_metrics, os.path.join(output_dir, "ppo_training_curves.png"))
        
        # Evaluate PPO agent
        print("\nEvaluating PPO agent...")
        ppo_return = ppo_agent.evaluate(env, num_episodes=config.get("final_eval_episodes", 10))
        print(f"PPO final evaluation return: {ppo_return:.2f}")
        
        # Store results
        results["metrics"]["ppo"] = ppo_metrics
        results["eval_returns"]["ppo"] = ppo_return
    
    # Train and evaluate SAC agent
    if "sac" in config["agents"]:
        print("\n=== Training SAC Agent ===")
        sac_config = config["agents"]["sac"]
        
        print(f"SAC Configuration:")
        for key, value in sac_config.items():
            print(f"  {key}: {value}")
        
        print("\nInitializing SAC agent...")
        sac_agent = SACAgent(
            env=env,
            hidden_dim=sac_config.get("hidden_dim", 256),
            lr=sac_config.get("lr", 3e-4),
            gamma=sac_config.get("gamma", 0.99),
            tau=sac_config.get("tau", 0.005),
            alpha=sac_config.get("alpha", 0.2),
            auto_alpha=sac_config.get("auto_alpha", True)
        )
        
        print(f"\nStarting SAC training for {config.get('total_timesteps', 100000)} timesteps...")
        sac_metrics = sac_agent.train(
            env=env,
            total_timesteps=config.get("total_timesteps", 100000),
            batch_size=sac_config.get("batch_size", 256),
            buffer_size=sac_config.get("buffer_size", 100000),
            update_after=sac_config.get("update_after", 1000),
            update_every=sac_config.get("update_every", 50),
            num_eval_episodes=config.get("num_eval_episodes", 5),
            eval_freq=config.get("eval_freq", 5000),
            save_freq=config.get("save_freq", 10000),
            save_path=os.path.join(output_dir, "sac_model"),
            tb_writer=tb_writer
        )
        
        print("\nSAC training completed")
        print(f"Saving training curves to {os.path.join(output_dir, 'sac_training_curves.png')}")
        
        # Plot SAC training metrics
        plot_training_curves(sac_metrics, os.path.join(output_dir, "sac_training_curves.png"))
        
        # Evaluate SAC agent
        print("\nEvaluating SAC agent...")
        sac_return = sac_agent.evaluate(env, num_episodes=config.get("final_eval_episodes", 10))
        print(f"SAC final evaluation return: {sac_return:.2f}")
        
        # Store results
        results["metrics"]["sac"] = sac_metrics
        results["eval_returns"]["sac"] = sac_return
    
    # Compare agents
    if len(results["eval_returns"]) > 1:
        print("\nComparing agents...")
        
        # Plot comparison
        plot_comparison(
            results["metrics"],
            os.path.join(output_dir, "agent_comparison.png"),
            title="Agent Comparison"
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
    """
    Main function to compare different agents.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Configuration for agent comparison
    config = {
        "agents": {
            "ppo": {
                "hidden_dim": 256,
                "lr": 3e-4,
                "gamma": 0.99,
                "lam": 0.95,
                "clip_ratio": 0.2,
                "entropy_coef": 0.01,
                "value_coef": 0.5,
                "max_grad_norm": 0.5,
                "buffer_size": 2048,
                "batch_size": 64,
                "update_epochs": 10
            },
            "sac": {
                "hidden_dim": 256,
                "lr": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "alpha": 0.2,
                "auto_alpha": True,
                "buffer_size": 100000,
                "batch_size": 256,
                "update_after": 1000,
                "update_every": 50
            }
        },
        "episode_length": args.episode_length,
        "normalize_state": True,
        "total_timesteps": args.timesteps,
        "num_eval_episodes": 5,
        "eval_freq": args.timesteps // 20,
        "save_freq": args.timesteps // 5,
        "final_eval_episodes": args.eval_episodes,
        "agents": args.agents,
        "use_tensorboard": args.use_tensorboard
    }
    
    # Train and evaluate agents
    results = train_and_evaluate_agents(config)
    
    # Print final results
    print("\nFinal Results:")
    for agent_name, eval_return in results["eval_returns"].items():
        print(f"{agent_name.upper()} final evaluation return: {eval_return:.2f}")

if __name__ == "__main__":
    main() 