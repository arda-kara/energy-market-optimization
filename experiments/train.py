import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

from market_simulator.demand_generator import DemandGenerator
from market_simulator.supply_simulator import SupplySimulator
from market_simulator.grid_simulator import GridSimulator
from market_simulator.pricing_model import PricingModel
from rl_environment.energy_env import EnergyMarketEnv
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from utils.visualization import plot_training_curves
from utils.data_processing import (
    create_results_directory, save_training_history, 
    create_tensorboard_writer, log_episode_metrics_to_tensorboard,
    log_agent_update_to_tensorboard, log_environment_to_tensorboard,
    log_hyperparameters_to_tensorboard, close_tensorboard_writer
)

def load_best_hyperparameters(agent_type):
    """
    Load best hyperparameters from hyperparameter tuning.
    
    Args:
        agent_type (str): Type of agent ('ppo' or 'sac')
        
    Returns:
        dict: Best hyperparameters or None if not found
    """
    # Path to best hyperparameters file
    file_path = os.path.join("results", "hyperparameter_tuning", f"{agent_type}_best_params.txt")
    
    if not os.path.exists(file_path):
        print(f"No best hyperparameters found for {agent_type}. Using default parameters.")
        return None
    
    # Load hyperparameters
    params = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header line
            if ':' in line:
                key, value = line.strip().split(':', 1)
                # Convert value to appropriate type
                try:
                    # Try to convert to int
                    params[key.strip()] = int(value.strip())
                except ValueError:
                    try:
                        # Try to convert to float
                        params[key.strip()] = float(value.strip())
                    except ValueError:
                        # Keep as string
                        params[key.strip()] = value.strip()
    
    return params

def train_agent(agent_type, env, total_timesteps=100000, save_path=None, log_interval=1000, 
                use_best_params=False, use_tensorboard=True):
    """
    Train an RL agent on the energy market environment.
    
    Args:
        agent_type (str): Type of agent to train ('ppo' or 'sac')
        env (EnergyMarketEnv): Environment to train on
        total_timesteps (int): Total number of timesteps to train for
        save_path (str): Path to save the trained model
        log_interval (int): Interval for logging training progress
        use_best_params (bool): Whether to use best hyperparameters from tuning
        use_tensorboard (bool): Whether to use TensorBoard for logging
        
    Returns:
        agent: Trained agent
        dict: Training history
    """
    # Load best hyperparameters if requested
    hyperparams = None
    if use_best_params:
        hyperparams = load_best_hyperparameters(agent_type)
    
    # Create agent
    if agent_type.lower() == 'ppo':
        if hyperparams:
            agent = PPOAgent(
                env,
                learning_rate=hyperparams.get('learning_rate', 3e-4),
                n_steps=hyperparams.get('n_steps', 2048),
                batch_size=hyperparams.get('batch_size', 64),
                n_epochs=hyperparams.get('n_epochs', 10),
                gamma=hyperparams.get('gamma', 0.99),
                gae_lambda=hyperparams.get('gae_lambda', 0.95),
                clip_range=hyperparams.get('clip_range', 0.2),
                ent_coef=hyperparams.get('ent_coef', 0.0),
                vf_coef=hyperparams.get('vf_coef', 0.5)
            )
        else:
            agent = PPOAgent(env)
    elif agent_type.lower() == 'sac':
        if hyperparams:
            agent = SACAgent(
                env,
                learning_rate=hyperparams.get('learning_rate', 3e-4),
                buffer_size=hyperparams.get('buffer_size', 1000000),
                batch_size=hyperparams.get('batch_size', 256),
                tau=hyperparams.get('tau', 0.005),
                gamma=hyperparams.get('gamma', 0.99),
                train_freq=hyperparams.get('train_freq', 1),
                gradient_steps=hyperparams.get('gradient_steps', 1)
            )
        else:
            agent = SACAgent(env)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    # Initialize TensorBoard writer if requested
    tb_writer = None
    if use_tensorboard:
        tb_writer = create_tensorboard_writer(
            agent_type=agent_type,
            comment="training"
        )
        
        # Log hyperparameters
        if agent_type.lower() == 'ppo':
            hyperparams_to_log = {
                'learning_rate': agent.learning_rate,
                'n_steps': agent.n_steps,
                'batch_size': agent.batch_size,
                'n_epochs': agent.n_epochs,
                'gamma': agent.gamma,
                'gae_lambda': agent.gae_lambda,
                'clip_range': agent.clip_range,
                'ent_coef': agent.ent_coef,
                'vf_coef': agent.vf_coef
            }
        else:  # SAC
            hyperparams_to_log = {
                'learning_rate': agent.learning_rate,
                'buffer_size': agent.buffer_size,
                'batch_size': agent.batch_size,
                'tau': agent.tau,
                'gamma': agent.gamma,
                'train_freq': agent.train_freq,
                'gradient_steps': agent.gradient_steps
            }
        
        log_hyperparameters_to_tensorboard(tb_writer, hyperparams_to_log)
    
    # Initialize training history
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_costs': [],
        'episode_emissions': [],
        'episode_imbalances': [],
        'episode_renewable_usage': []
    }
    
    # Training loop
    state = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_cost = 0
    episode_emission = 0
    episode_imbalance = 0
    episode_renewable = 0
    
    print(f"Starting {agent_type.upper()} training for {total_timesteps} timesteps...")
    
    for t in range(1, total_timesteps + 1):
        # Select action
        action = agent.select_action(state)
        
        # Take step in environment
        next_state, reward, done, info = env.step(action)
        
        # Store transition in agent's memory
        agent.store_transition(state, action, reward, next_state, done)
        
        # Update agent
        if t % agent.update_frequency == 0:
            update_info = agent.update()
            
            # Log update metrics to TensorBoard
            if use_tensorboard and update_info:
                log_agent_update_to_tensorboard(
                    tb_writer, 
                    t // agent.update_frequency, 
                    update_info, 
                    agent_type
                )
        
        # Update episode statistics
        episode_reward += reward
        episode_length += 1
        episode_cost += info.get('cost', 0)
        episode_emission += info.get('emissions', 0)
        episode_imbalance += info.get('imbalance', 0)
        episode_renewable += info.get('renewable_percentage', 0)
        
        # Log environment state to TensorBoard
        if use_tensorboard and t % 100 == 0:
            log_environment_to_tensorboard(tb_writer, env, t)
        
        # Move to next state
        state = next_state
        
        # If episode is done
        if done:
            # Average renewable percentage over the episode
            episode_renewable /= episode_length
            
            # Store episode statistics
            history['episode_rewards'].append(episode_reward)
            history['episode_lengths'].append(episode_length)
            history['episode_costs'].append(episode_cost)
            history['episode_emissions'].append(episode_emission)
            history['episode_imbalances'].append(episode_imbalance)
            history['episode_renewable_usage'].append(episode_renewable)
            
            # Log episode metrics to TensorBoard
            if use_tensorboard:
                log_episode_metrics_to_tensorboard(
                    tb_writer,
                    len(history['episode_rewards']),
                    episode_reward,
                    episode_length,
                    episode_cost,
                    episode_emission,
                    episode_imbalance,
                    episode_renewable
                )
            
            # Log progress
            if len(history['episode_rewards']) % log_interval == 0:
                avg_reward = np.mean(history['episode_rewards'][-log_interval:])
                avg_length = np.mean(history['episode_lengths'][-log_interval:])
                avg_cost = np.mean(history['episode_costs'][-log_interval:])
                avg_emission = np.mean(history['episode_emissions'][-log_interval:])
                avg_imbalance = np.mean(history['episode_imbalances'][-log_interval:])
                avg_renewable = np.mean(history['episode_renewable_usage'][-log_interval:])
                
                print(f"Episode {len(history['episode_rewards'])}, Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.2f}, Avg Cost: {avg_cost:.2f}, "
                      f"Avg Emissions: {avg_emission:.2f}, Avg Imbalance: {avg_imbalance:.2f}, "
                      f"Avg Renewable %: {avg_renewable*100:.2f}%")
            
            # Create a new environment for the next episode
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
            
            # Reset state and episode statistics
            state = new_env.reset()
            episode_reward = 0
            episode_length = 0
            episode_cost = 0
            episode_emission = 0
            episode_imbalance = 0
            episode_renewable = 0
    
    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agent.save(save_path)
        print(f"Model saved to {save_path}")
    
    # Store agent-specific losses
    if agent_type.lower() == 'ppo':
        history['policy_losses'] = agent.policy_losses
        history['value_losses'] = agent.value_losses
    elif agent_type.lower() == 'sac':
        history['actor_losses'] = agent.actor_losses
        history['critic_losses'] = agent.critic_losses
        history['alpha_losses'] = agent.alpha_losses
    
    # Close TensorBoard writer
    if use_tensorboard:
        close_tensorboard_writer(tb_writer)
    
    return agent, history

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train RL agents for energy market optimization')
    parser.add_argument('--use_best_params', action='store_true', help='Use best hyperparameters from tuning')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable TensorBoard logging')
    parser.add_argument('--timesteps', type=int, default=50000, help='Total timesteps to train for')
    parser.add_argument('--agent', type=str, default='both', choices=['ppo', 'sac', 'both'], help='Agent type to train')
    parser.add_argument('--days', type=int, default=7, help='Number of days per episode')
    args = parser.parse_args()
    
    # Calculate max steps based on days
    max_steps = 24 * args.days
    
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", "training", f"training_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create models directory if it doesn't exist
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create logs directory if using TensorBoard
    if not args.no_tensorboard:
        logs_dir = os.path.join("logs", "tensorboard")
        os.makedirs(logs_dir, exist_ok=True)
    
    # Create simulators
    demand_generator = DemandGenerator()
    supply_simulator = SupplySimulator()
    grid_simulator = GridSimulator()
    pricing_model = PricingModel()
    
    # Generate demand data
    demand_data = demand_generator.generate_data(days=30, save_to_file=False)  # Generate a month of demand data
    
    # Set timestamp as index if it's not already
    if 'timestamp' in demand_data.columns:
        demand_data = demand_data.set_index('timestamp')
    
    env = EnergyMarketEnv(
        demand_data=demand_data,
        forecast_horizon=6,
        episode_length=max_steps,
        normalize_state=True
    )
    
    # Train PPO agent
    if args.agent.lower() in ['ppo', 'both']:
        print("\nTraining PPO agent...")
        ppo_agent, ppo_history = train_agent(
            agent_type='ppo',
            env=env,
            total_timesteps=args.timesteps,
            save_path=os.path.join(models_dir, "ppo_agent.pt"),
            log_interval=10,
            use_best_params=args.use_best_params,
            use_tensorboard=not args.no_tensorboard
        )
        
        # Plot and save PPO training curves
        plot_training_curves(
            ppo_history,
            title="PPO Training Progress",
            save_path=os.path.join(output_dir, "ppo_training_curves.png")
        )
        
        # Save training history
        save_training_history(ppo_history, 'ppo', output_dir)
    
    # Train SAC agent
    if args.agent.lower() in ['sac', 'both']:
        print("\nTraining SAC agent...")
        sac_agent, sac_history = train_agent(
            agent_type='sac',
            env=env,
            total_timesteps=args.timesteps,
            save_path=os.path.join(models_dir, "sac_agent.pt"),
            log_interval=10,
            use_best_params=args.use_best_params,
            use_tensorboard=not args.no_tensorboard
        )
        
        # Plot and save SAC training curves
        plot_training_curves(
            sac_history,
            title="SAC Training Progress",
            save_path=os.path.join(output_dir, "sac_training_curves.png")
        )
        
        # Save training history
        save_training_history(sac_history, 'sac', output_dir)
    
    print(f"\nTraining complete. Results saved to {output_dir}")
    print(f"Models saved to {models_dir}")
    
    if not args.no_tensorboard:
        print("\nTo view TensorBoard logs, run:")
        print("tensorboard --logdir=logs/tensorboard")
    
    print("\nTo evaluate the trained agents, run:")
    print("python -m experiments.evaluate")

if __name__ == "__main__":
    # Create required directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs/tensorboard', exist_ok=True)
    os.makedirs('results/training', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    main()