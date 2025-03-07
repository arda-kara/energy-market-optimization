import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
from datetime import datetime

from market_simulator.demand_generator import DemandGenerator
from market_simulator.supply_simulator import SupplySimulator
from market_simulator.grid_simulator import GridSimulator
from market_simulator.pricing_model import PricingModel
from rl_environment.energy_env import EnergyMarketEnv
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from experiments.baseline import run_baseline_strategies
from utils.visualization import plot_comparison, plot_agent_metrics
from utils.data_processing import (
    calculate_performance_metrics, create_results_directory,
    create_tensorboard_writer, log_metrics_to_tensorboard,
    close_tensorboard_writer
)

def load_agent(agent_type, model_path, env):
    """
    Load a trained agent from a saved model file.
    
    Args:
        agent_type (str): Type of agent ('ppo' or 'sac')
        model_path (str): Path to the saved model
        env (EnergyMarketEnv): Environment to use with the agent
        
    Returns:
        Agent: Loaded agent
    """
    if agent_type.lower() == 'ppo':
        agent = PPOAgent(env)
        agent.load(model_path)
    elif agent_type.lower() == 'sac':
        agent = SACAgent(env)
        agent.load(model_path)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    return agent

def evaluate_agent(agent, env, num_episodes=10, render=False, tb_writer=None):
    """
    Evaluate a trained agent on the environment.
    
    Args:
        agent: Trained agent (PPO or SAC)
        env (EnergyMarketEnv): Environment to evaluate on
        num_episodes (int): Number of episodes to evaluate
        render (bool): Whether to render the environment
        tb_writer: TensorBoard writer for logging
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_costs = []
    episode_emissions = []
    episode_imbalances = []
    episode_renewable_usage = []
    
    for episode in range(num_episodes):
        # Create a new environment for each episode to ensure fresh state
        # This is more reliable than using env.reset() which might not be properly implemented
        demand_generator = DemandGenerator()
        supply_simulator = SupplySimulator()
        grid_simulator = GridSimulator()
        pricing_model = PricingModel()
        
        new_env = EnergyMarketEnv(
            demand_generator=demand_generator,
            supply_simulator=supply_simulator,
            grid_simulator=grid_simulator,
            pricing_model=pricing_model,
            max_steps=env.max_steps
        )
        
        # Update agent's environment
        agent.env = new_env
        
        state = new_env.reset()
        done = False
        total_reward = 0
        episode_cost = 0
        episode_emission = 0
        episode_imbalance = 0
        episode_renewable = 0
        step = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = new_env.step(action)
            
            total_reward += reward
            episode_cost += info.get('cost', 0)
            episode_emission += info.get('emissions', 0)
            episode_imbalance += info.get('imbalance', 0)
            episode_renewable += info.get('renewable_percentage', 0)
            
            # Log step metrics to TensorBoard
            if tb_writer:
                step_metrics = {
                    'reward': reward,
                    'cost': info.get('cost', 0),
                    'emissions': info.get('emissions', 0),
                    'imbalance': info.get('imbalance', 0),
                    'renewable_percentage': info.get('renewable_percentage', 0)
                }
                log_metrics_to_tensorboard(
                    tb_writer, 
                    step_metrics, 
                    episode * new_env.max_steps + step, 
                    prefix='step'
                )
            
            state = next_state
            step += 1
            
            if render:
                new_env.render()
        
        # Average renewable percentage over the episode
        episode_renewable /= new_env.max_steps
        
        episode_rewards.append(total_reward)
        episode_costs.append(episode_cost)
        episode_emissions.append(episode_emission)
        episode_imbalances.append(episode_imbalance)
        episode_renewable_usage.append(episode_renewable)
        
        # Log episode metrics to TensorBoard
        if tb_writer:
            episode_metrics = {
                'reward': total_reward,
                'cost': episode_cost,
                'emissions': episode_emission,
                'imbalance': episode_imbalance,
                'renewable_usage': episode_renewable
            }
            log_metrics_to_tensorboard(
                tb_writer, 
                episode_metrics, 
                episode, 
                prefix='episode'
            )
        
        print(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward:.2f}, Cost: {episode_cost:.2f}, "
              f"Emissions: {episode_emission:.2f}, Imbalance: {episode_imbalance:.2f}, "
              f"Renewable %: {episode_renewable*100:.2f}%")
    
    # Calculate average metrics
    metrics = {
        'avg_reward': np.mean(episode_rewards),
        'avg_cost': np.mean(episode_costs),
        'avg_emissions': np.mean(episode_emissions),
        'avg_imbalance': np.mean(episode_imbalances),
        'avg_renewable_usage': np.mean(episode_renewable_usage),
        'std_reward': np.std(episode_rewards),
        'std_cost': np.std(episode_costs),
        'std_emissions': np.std(episode_emissions),
        'std_imbalance': np.std(episode_imbalances),
        'std_renewable_usage': np.std(episode_renewable_usage),
    }
    
    # Log summary metrics to TensorBoard
    if tb_writer:
        log_metrics_to_tensorboard(
            tb_writer, 
            metrics, 
            0, 
            prefix='summary'
        )
    
    return metrics

def compare_with_baselines(agent_metrics, baseline_metrics):
    """
    Compare agent performance with baseline strategies.
    
    Args:
        agent_metrics (dict): Metrics from agent evaluation
        baseline_metrics (dict): Metrics from baseline strategies
        
    Returns:
        pd.DataFrame: Comparison dataframe
    """
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Metric': ['Cost', 'Emissions', 'Grid Imbalance', 'Renewable Usage (%)'],
        'RL Agent': [
            f"{agent_metrics['avg_cost']:.2f} ± {agent_metrics['std_cost']:.2f}",
            f"{agent_metrics['avg_emissions']:.2f} ± {agent_metrics['std_emissions']:.2f}",
            f"{agent_metrics['avg_imbalance']:.2f} ± {agent_metrics['std_imbalance']:.2f}",
            f"{agent_metrics['avg_renewable_usage']*100:.2f}% ± {agent_metrics['std_renewable_usage']*100:.2f}%"
        ],
        'Merit Order': [
            f"{baseline_metrics['merit_order']['avg_cost']:.2f} ± {baseline_metrics['merit_order']['std_cost']:.2f}",
            f"{baseline_metrics['merit_order']['avg_emissions']:.2f} ± {baseline_metrics['merit_order']['std_emissions']:.2f}",
            f"{baseline_metrics['merit_order']['avg_imbalance']:.2f} ± {baseline_metrics['merit_order']['std_imbalance']:.2f}",
            f"{baseline_metrics['merit_order']['avg_renewable_usage']*100:.2f}% ± {baseline_metrics['merit_order']['std_renewable_usage']*100:.2f}%"
        ],
        'Proportional': [
            f"{baseline_metrics['proportional']['avg_cost']:.2f} ± {baseline_metrics['proportional']['std_cost']:.2f}",
            f"{baseline_metrics['proportional']['avg_emissions']:.2f} ± {baseline_metrics['proportional']['std_emissions']:.2f}",
            f"{baseline_metrics['proportional']['avg_imbalance']:.2f} ± {baseline_metrics['proportional']['std_imbalance']:.2f}",
            f"{baseline_metrics['proportional']['avg_renewable_usage']*100:.2f}% ± {baseline_metrics['proportional']['std_renewable_usage']*100:.2f}%"
        ],
        'Renewable First': [
            f"{baseline_metrics['renewable_first']['avg_cost']:.2f} ± {baseline_metrics['renewable_first']['std_cost']:.2f}",
            f"{baseline_metrics['renewable_first']['avg_emissions']:.2f} ± {baseline_metrics['renewable_first']['std_emissions']:.2f}",
            f"{baseline_metrics['renewable_first']['avg_imbalance']:.2f} ± {baseline_metrics['renewable_first']['std_imbalance']:.2f}",
            f"{baseline_metrics['renewable_first']['avg_renewable_usage']*100:.2f}% ± {baseline_metrics['renewable_first']['std_renewable_usage']*100:.2f}%"
        ]
    })
    
    return comparison

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate trained RL agents')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to evaluate')
    parser.add_argument('--days', type=int, default=7, help='Number of days to simulate')
    parser.add_argument('--render', action='store_true', help='Render the environment during evaluation')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable TensorBoard logging')
    args = parser.parse_args()
    
    # Calculate max steps based on days
    max_steps = 24 * args.days
    
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", "evaluation", f"evaluation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create TensorBoard writer if enabled
    tb_writer = None
    if not args.no_tensorboard:
        tb_writer = create_tensorboard_writer(
            log_dir=os.path.join("logs", "tensorboard", f"evaluation_{timestamp}"),
            comment="evaluation"
        )
    
    # Set up environment
    print("Setting up environment...")
    demand_generator = DemandGenerator()
    supply_simulator = SupplySimulator()
    grid_simulator = GridSimulator()
    pricing_model = PricingModel()
    
    env = EnergyMarketEnv(
        demand_generator=demand_generator,
        supply_simulator=supply_simulator,
        grid_simulator=grid_simulator,
        pricing_model=pricing_model,
        max_steps=max_steps
    )
    
    # Get model paths
    model_dir = "models"
    ppo_model_path = os.path.join(model_dir, "ppo_agent.pt")
    sac_model_path = os.path.join(model_dir, "sac_agent.pt")
    
    # Check if models exist
    if not os.path.exists(ppo_model_path) and not os.path.exists(sac_model_path):
        print("No trained models found. Please train models first using experiments/train.py")
        return
    
    # Evaluate agents
    agent_metrics = {}
    
    if os.path.exists(ppo_model_path):
        print("\nEvaluating PPO agent...")
        ppo_agent = load_agent('ppo', ppo_model_path, env)
        
        # Create agent-specific TensorBoard writer
        ppo_tb_writer = None
        if tb_writer:
            ppo_tb_writer = create_tensorboard_writer(
                log_dir=os.path.join("logs", "tensorboard", f"evaluation_{timestamp}_ppo"),
                agent_type="ppo",
                comment="evaluation"
            )
        
        ppo_metrics = evaluate_agent(
            ppo_agent, env, num_episodes=args.episodes, render=args.render, tb_writer=ppo_tb_writer
        )
        agent_metrics['ppo'] = ppo_metrics
        
        # Close agent-specific TensorBoard writer
        if ppo_tb_writer:
            close_tensorboard_writer(ppo_tb_writer)
    
    if os.path.exists(sac_model_path):
        print("\nEvaluating SAC agent...")
        sac_agent = load_agent('sac', sac_model_path, env)
        
        # Create agent-specific TensorBoard writer
        sac_tb_writer = None
        if tb_writer:
            sac_tb_writer = create_tensorboard_writer(
                log_dir=os.path.join("logs", "tensorboard", f"evaluation_{timestamp}_sac"),
                agent_type="sac",
                comment="evaluation"
            )
        
        sac_metrics = evaluate_agent(
            sac_agent, env, num_episodes=args.episodes, render=args.render, tb_writer=sac_tb_writer
        )
        agent_metrics['sac'] = sac_metrics
        
        # Close agent-specific TensorBoard writer
        if sac_tb_writer:
            close_tensorboard_writer(sac_tb_writer)
    
    # Run baseline strategies
    print("\nRunning baseline strategies...")
    baseline_metrics = run_baseline_strategies(
        demand_generator=demand_generator,
        supply_simulator=supply_simulator,
        grid_simulator=grid_simulator,
        pricing_model=pricing_model,
        num_episodes=args.episodes,
        max_steps=max_steps,
        tb_writer=tb_writer
    )
    
    # Compare with baselines
    print("\nComparing with baseline strategies...")
    best_agent = 'ppo' if 'ppo' in agent_metrics and ('sac' not in agent_metrics or 
                                                     agent_metrics['ppo']['avg_reward'] > agent_metrics['sac']['avg_reward']) else 'sac'
    
    comparison = compare_with_baselines(agent_metrics[best_agent], baseline_metrics)
    print("\nPerformance Comparison:")
    print(comparison)
    
    # Save comparison to CSV
    comparison.to_csv(os.path.join(output_dir, "performance_comparison.csv"), index=False)
    
    # Plot comparison
    plot_comparison(
        agent_metrics=agent_metrics[best_agent],
        baseline_metrics=baseline_metrics,
        title=f"Performance Comparison: {best_agent.upper()} vs Baselines",
        save_path=os.path.join(output_dir, f"{best_agent}_vs_baselines.png")
    )
    
    # Plot agent metrics if both agents were evaluated
    if 'ppo' in agent_metrics and 'sac' in agent_metrics:
        plot_agent_metrics(
            ppo_metrics=agent_metrics['ppo'],
            sac_metrics=agent_metrics['sac'],
            title="PPO vs SAC Performance",
            save_path=os.path.join(output_dir, "ppo_vs_sac.png")
        )
    
    # Close TensorBoard writer
    if tb_writer:
        close_tensorboard_writer(tb_writer)
    
    print(f"\nEvaluation complete. Results saved to {output_dir}")
    
    if not args.no_tensorboard:
        print("\nTo view TensorBoard logs, run:")
        print("tensorboard --logdir=logs/tensorboard")

if __name__ == "__main__":
    # Create required directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs/tensorboard', exist_ok=True)
    os.makedirs('results/evaluation', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    main() 