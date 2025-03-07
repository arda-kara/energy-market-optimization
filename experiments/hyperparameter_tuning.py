import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import optuna
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, SAC

from market_simulator.demand_generator import DemandGenerator
from market_simulator.supply_simulator import SupplySimulator
from market_simulator.grid_simulator import GridSimulator
from market_simulator.pricing_model import PricingModel
from rl_environment.energy_env import EnergyMarketEnv, make_env
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from utils.visualization import plot_training_curves
from utils.data_processing import create_results_directory

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for RL agents')
    
    parser.add_argument('--agent', type=str, default='both', 
                        choices=['ppo', 'sac', 'both'], help='Agent to optimize')
    parser.add_argument('--n-trials', type=int, default=50, 
                        help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=None, 
                        help='Timeout for optimization in seconds')
    parser.add_argument('--use-tensorboard', action='store_true',
                        help='Enable TensorBoard logging')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    return parser.parse_args()

def objective_ppo(trial, use_tensorboard=False):
    """
    Objective function for PPO hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        use_tensorboard: Whether to use TensorBoard for logging
        
    Returns:
        float: Mean reward over evaluation episodes
    """
    # Create environment
    env = EnergyMarketEnv(
        episode_length=24*3,  # 3 days for faster evaluation
        normalize_state=True
    )
    
    # Sample hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    lam = trial.suggest_float("lam", 0.9, 1.0)
    clip_ratio = trial.suggest_float("clip_ratio", 0.1, 0.3)
    entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.1)
    value_coef = trial.suggest_float("value_coef", 0.5, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 1.0)
    
    # Create agent
    agent = PPOAgent(
        env=env,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        lam=lam,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm
    )
    
    # Setup TensorBoard if enabled
    tb_writer = None
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_log_dir = os.path.join("logs", "tensorboard", f"ppo_trial_{trial.number}")
            os.makedirs(tb_log_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_log_dir)
        except ImportError:
            print("Warning: TensorBoard not available. Install with 'pip install tensorboard'")
    
    # Train agent
    try:
        metrics = agent.train(
            env=env,
            total_timesteps=50000,  # Reduced for faster optimization
            buffer_size=2048,
            batch_size=64,
            update_epochs=10,
            num_eval_episodes=3,
            eval_freq=10000,
            tb_writer=tb_writer
        )
        
        # Close TensorBoard writer
        if tb_writer is not None:
            tb_writer.close()
        
        # Return mean evaluation return
        return np.mean(metrics["eval_returns"])
    except Exception as e:
        print(f"Error in PPO trial {trial.number}: {str(e)}")
        # Close TensorBoard writer
        if tb_writer is not None:
            tb_writer.close()
        return float('-inf')  # Return negative infinity for failed trials

def objective_sac(trial, use_tensorboard=False):
    """
    Objective function for SAC hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        use_tensorboard: Whether to use TensorBoard for logging
        
    Returns:
        float: Mean reward over evaluation episodes
    """
    # Create environment
    env = EnergyMarketEnv(
        episode_length=24*3,  # 3 days for faster evaluation
        normalize_state=True
    )
    
    # Sample hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    tau = trial.suggest_float("tau", 0.001, 0.01)
    alpha = trial.suggest_float("alpha", 0.1, 0.5)
    auto_alpha = trial.suggest_categorical("auto_alpha", [True, False])
    buffer_size = trial.suggest_categorical("buffer_size", [10000, 50000, 100000])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    update_every = trial.suggest_categorical("update_every", [10, 50, 100])
    
    # Create agent
    agent = SACAgent(
        env=env,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        auto_alpha=auto_alpha
    )
    
    # Setup TensorBoard if enabled
    tb_writer = None
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_log_dir = os.path.join("logs", "tensorboard", f"sac_trial_{trial.number}")
            os.makedirs(tb_log_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_log_dir)
        except ImportError:
            print("Warning: TensorBoard not available. Install with 'pip install tensorboard'")
    
    # Train agent
    try:
        metrics = agent.train(
            env=env,
            total_timesteps=50000,  # Reduced for faster optimization
            buffer_size=buffer_size,
            batch_size=batch_size,
            update_after=1000,
            update_every=update_every,
            num_eval_episodes=3,
            eval_freq=10000,
            tb_writer=tb_writer
        )
        
        # Close TensorBoard writer
        if tb_writer is not None:
            tb_writer.close()
        
        # Return mean evaluation return
        return np.mean(metrics["eval_returns"])
    except Exception as e:
        print(f"Error in SAC trial {trial.number}: {str(e)}")
        # Close TensorBoard writer
        if tb_writer is not None:
            tb_writer.close()
        return float('-inf')  # Return negative infinity for failed trials

def optimize_hyperparameters(agent_type, n_trials=50, timeout=None, study_name=None, use_tensorboard=False):
    """
    Optimize hyperparameters for the specified agent type.
    
    Args:
        agent_type (str): Type of agent to optimize ('ppo' or 'sac')
        n_trials (int): Number of optimization trials
        timeout (int, optional): Timeout for optimization in seconds
        study_name (str, optional): Name of the study
        use_tensorboard (bool): Whether to use TensorBoard for logging
        
    Returns:
        optuna.Study: Completed optimization study
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = create_results_directory(f"hyperparameter_tuning/{agent_type}_{timestamp}")
    
    # Create study name if not provided
    if study_name is None:
        study_name = f"{agent_type}_optimization_{timestamp}"
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{output_dir}/study.db",
        load_if_exists=True
    )
    
    # Run optimization
    if agent_type == "ppo":
        study.optimize(
            lambda trial: objective_ppo(trial, use_tensorboard),
            n_trials=n_trials,
            timeout=timeout
        )
    elif agent_type == "sac":
        study.optimize(
            lambda trial: objective_sac(trial, use_tensorboard),
            n_trials=n_trials,
            timeout=timeout
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Print best parameters
    print(f"\nBest {agent_type.upper()} parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best value: {study.best_value:.4f}")
    
    # Save study results
    study_results_path = os.path.join(output_dir, f"{agent_type}_study_results.pkl")
    with open(study_results_path, "wb") as f:
        import pickle
        pickle.dump(study, f)
    
    # Plot optimization history
    plot_optimization_history(agent_type, study, output_dir)
    
    return study

def plot_optimization_history(agent_type, study, output_dir):
    """
    Plot optimization history.
    
    Args:
        agent_type (str): Type of agent ('ppo' or 'sac')
        study (optuna.Study): Completed optimization study
        output_dir (str): Directory to save plots
    """
    # Plot optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(os.path.join(output_dir, f"{agent_type}_optimization_history.png"))
    
    # Plot parameter importances
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(os.path.join(output_dir, f"{agent_type}_param_importances.png"))
    except:
        print("Could not plot parameter importances (not enough trials)")
    
    # Plot parallel coordinate plot
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(os.path.join(output_dir, f"{agent_type}_parallel_coordinate.png"))
    except:
        print("Could not plot parallel coordinate (not enough trials)")
    
    # Plot slice plot
    try:
        fig = optuna.visualization.plot_slice(study)
        fig.write_image(os.path.join(output_dir, f"{agent_type}_slice.png"))
    except:
        print("Could not plot slice (not enough trials)")
    
    # Plot contour plot
    try:
        fig = optuna.visualization.plot_contour(study)
        fig.write_image(os.path.join(output_dir, f"{agent_type}_contour.png"))
    except:
        print("Could not plot contour (not enough trials)")

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create required directories
    os.makedirs("results/hyperparameter_tuning", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)
    
    if args.agent == "both" or args.agent == "ppo":
        print("\n=== Optimizing PPO Hyperparameters ===")
        ppo_study = optimize_hyperparameters(
            agent_type="ppo",
            n_trials=args.n_trials,
            timeout=args.timeout,
            use_tensorboard=args.use_tensorboard
        )
    
    if args.agent == "both" or args.agent == "sac":
        print("\n=== Optimizing SAC Hyperparameters ===")
        sac_study = optimize_hyperparameters(
            agent_type="sac",
            n_trials=args.n_trials,
            timeout=args.timeout,
            use_tensorboard=args.use_tensorboard
        )
    
    print("\nHyperparameter optimization completed!")

if __name__ == "__main__":
    main() 