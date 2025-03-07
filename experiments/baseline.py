import argparse
import pandas as pd
from market_simulator.grid_simulator import GridSimulator
from utils.visualization import plot_demand_supply, plot_environment_metrics
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime

from market_simulator.demand_generator import DemandGenerator
from market_simulator.supply_simulator import SupplySimulator
from market_simulator.pricing_model import PricingModel
from utils.data_processing import log_metrics_to_tensorboard, create_tensorboard_writer, close_tensorboard_writer

def merit_order_allocation(demand, available_supply):
    """
    Allocate energy based on merit order (cheapest sources first).
    
    Args:
        demand (float): Current energy demand
        available_supply (dict): Available supply from each source
        
    Returns:
        dict: Allocated supply from each source
    """
    # Define cost order (cheapest to most expensive)
    cost_order = ['solar', 'wind', 'hydro', 'nuclear', 'gas', 'coal']
    
    # Initialize allocation
    allocation = {source: 0 for source in available_supply}
    remaining_demand = demand
    
    # Allocate in merit order
    for source in cost_order:
        if source in available_supply:
            allocation[source] = min(available_supply[source], remaining_demand)
            remaining_demand -= allocation[source]
            if remaining_demand <= 0:
                break
    
    return allocation

def proportional_allocation(demand, available_supply):
    """
    Allocate energy proportionally to available supply.
    
    Args:
        demand (float): Current energy demand
        available_supply (dict): Available supply from each source
        
    Returns:
        dict: Allocated supply from each source
    """
    # Calculate total available supply
    total_available = sum(available_supply.values())
    
    # Initialize allocation
    allocation = {source: 0 for source in available_supply}
    
    # Allocate proportionally
    if total_available > 0:
        for source in available_supply:
            allocation[source] = (available_supply[source] / total_available) * demand
            # Ensure we don't allocate more than available
            allocation[source] = min(allocation[source], available_supply[source])
    
    return allocation

def renewable_first_allocation(demand, available_supply):
    """
    Allocate energy from renewable sources first.
    
    Args:
        demand (float): Current energy demand
        available_supply (dict): Available supply from each source
        
    Returns:
        dict: Allocated supply from each source
    """
    # Define renewable and non-renewable sources
    renewable_sources = ['solar', 'wind', 'hydro']
    non_renewable_sources = ['nuclear', 'gas', 'coal']
    
    # Initialize allocation
    allocation = {source: 0 for source in available_supply}
    remaining_demand = demand
    
    # Allocate from renewable sources first
    for source in renewable_sources:
        if source in available_supply:
            allocation[source] = min(available_supply[source], remaining_demand)
            remaining_demand -= allocation[source]
            if remaining_demand <= 0:
                break
    
    # If demand not fully met, allocate from non-renewable sources
    if remaining_demand > 0:
        for source in non_renewable_sources:
            if source in available_supply:
                allocation[source] = min(available_supply[source], remaining_demand)
                remaining_demand -= allocation[source]
                if remaining_demand <= 0:
                    break
    
    return allocation

def run_baseline_strategy(strategy_name, strategy_func, demand_generator, supply_simulator, 
                         grid_simulator, pricing_model, num_episodes=5, max_steps=24*7, tb_writer=None):
    """
    Run a baseline allocation strategy.
    
    Args:
        strategy_name (str): Name of the strategy
        strategy_func (function): Strategy function
        demand_generator (DemandGenerator): Demand generator
        supply_simulator (SupplySimulator): Supply simulator
        grid_simulator (GridSimulator): Grid simulator
        pricing_model (PricingModel): Pricing model
        num_episodes (int): Number of episodes to run
        max_steps (int): Maximum steps per episode
        tb_writer: TensorBoard writer for logging
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    episode_costs = []
    episode_emissions = []
    episode_imbalances = []
    episode_renewable_usage = []
    
    for episode in range(num_episodes):
        # Create new instances for each episode instead of resetting
        # This is a workaround for classes that don't have reset methods
        demand_generator = DemandGenerator()
        supply_simulator = SupplySimulator()
        grid_simulator = GridSimulator()
        pricing_model = PricingModel()
        
        episode_cost = 0
        episode_emission = 0
        episode_imbalance = 0
        episode_renewable = 0
        
        for step in range(max_steps):
            # Get current demand and available supply
            current_demand = demand_generator.get_demand()
            available_supply = supply_simulator.get_available_supply()
            
            # Allocate energy using strategy
            allocation = strategy_func(current_demand, available_supply)
            
            # Update grid state
            grid_state = grid_simulator.update(current_demand, allocation)
            
            # Update pricing
            prices = pricing_model.calculate_prices(current_demand, allocation, grid_state)
            
            # Calculate metrics
            cost = sum(allocation[source] * prices[source] for source in allocation)
            emissions = sum(allocation[source] * supply_simulator.get_emission_factor(source) for source in allocation)
            imbalance = abs(sum(allocation.values()) - current_demand)
            
            # Calculate renewable percentage
            renewable_sources = ['solar', 'wind', 'hydro']
            total_allocation = sum(allocation.values())
            renewable_allocation = sum(allocation[source] for source in renewable_sources if source in allocation)
            renewable_percentage = renewable_allocation / total_allocation if total_allocation > 0 else 0
            
            # Update episode metrics
            episode_cost += cost
            episode_emission += emissions
            episode_imbalance += imbalance
            episode_renewable += renewable_percentage
            
            # Log step metrics to TensorBoard
            if tb_writer:
                step_metrics = {
                    f"{strategy_name}/cost": cost,
                    f"{strategy_name}/emissions": emissions,
                    f"{strategy_name}/imbalance": imbalance,
                    f"{strategy_name}/renewable_percentage": renewable_percentage
                }
                log_metrics_to_tensorboard(
                    tb_writer, 
                    step_metrics, 
                    episode * max_steps + step, 
                    prefix='step'
                )
            
            # Update simulators
            demand_generator.update()
            supply_simulator.update()
        
        # Average renewable percentage over the episode
        episode_renewable /= max_steps
        
        # Store episode metrics
        episode_costs.append(episode_cost)
        episode_emissions.append(episode_emission)
        episode_imbalances.append(episode_imbalance)
        episode_renewable_usage.append(episode_renewable)
        
        # Log episode metrics to TensorBoard
        if tb_writer:
            episode_metrics = {
                f"{strategy_name}/cost": episode_cost,
                f"{strategy_name}/emissions": episode_emission,
                f"{strategy_name}/imbalance": episode_imbalance,
                f"{strategy_name}/renewable_usage": episode_renewable
            }
            log_metrics_to_tensorboard(
                tb_writer, 
                episode_metrics, 
                episode, 
                prefix='episode'
            )
        
        print(f"{strategy_name} - Episode {episode+1}/{num_episodes}, Cost: {episode_cost:.2f}, "
              f"Emissions: {episode_emission:.2f}, Imbalance: {episode_imbalance:.2f}, "
              f"Renewable %: {episode_renewable*100:.2f}%")
    
    # Calculate average metrics
    metrics = {
        'avg_cost': np.mean(episode_costs),
        'avg_emissions': np.mean(episode_emissions),
        'avg_imbalance': np.mean(episode_imbalances),
        'avg_renewable_usage': np.mean(episode_renewable_usage),
        'std_cost': np.std(episode_costs),
        'std_emissions': np.std(episode_emissions),
        'std_imbalance': np.std(episode_imbalances),
        'std_renewable_usage': np.std(episode_renewable_usage),
    }
    
    # Log summary metrics to TensorBoard
    if tb_writer:
        summary_metrics = {
            f"{strategy_name}/avg_cost": metrics['avg_cost'],
            f"{strategy_name}/avg_emissions": metrics['avg_emissions'],
            f"{strategy_name}/avg_imbalance": metrics['avg_imbalance'],
            f"{strategy_name}/avg_renewable_usage": metrics['avg_renewable_usage'],
            f"{strategy_name}/std_cost": metrics['std_cost'],
            f"{strategy_name}/std_emissions": metrics['std_emissions'],
            f"{strategy_name}/std_imbalance": metrics['std_imbalance'],
            f"{strategy_name}/std_renewable_usage": metrics['std_renewable_usage']
        }
        log_metrics_to_tensorboard(
            tb_writer, 
            summary_metrics, 
            0, 
            prefix='summary'
        )
    
    return metrics

def run_baseline_strategies(demand_generator, supply_simulator, grid_simulator, pricing_model, 
                           num_episodes=5, max_steps=24*7, tb_writer=None):
    """
    Run all baseline allocation strategies.
    
    Args:
        demand_generator (DemandGenerator): Demand generator
        supply_simulator (SupplySimulator): Supply simulator
        grid_simulator (GridSimulator): Grid simulator
        pricing_model (PricingModel): Pricing model
        num_episodes (int): Number of episodes to run
        max_steps (int): Maximum steps per episode
        tb_writer: TensorBoard writer for logging
        
    Returns:
        dict: Dictionary of metrics for each strategy
    """
    # Define strategies
    strategies = {
        'merit_order': merit_order_allocation,
        'proportional': proportional_allocation,
        'renewable_first': renewable_first_allocation
    }
    
    # Run each strategy
    metrics = {}
    for name, func in strategies.items():
        print(f"\nRunning {name} strategy...")
        metrics[name] = run_baseline_strategy(
            name, func, demand_generator, supply_simulator, 
            grid_simulator, pricing_model, num_episodes, max_steps, tb_writer
        )
    
    return metrics

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run baseline energy allocation strategies')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--days', type=int, default=7, help='Number of days to simulate')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable TensorBoard logging')
    args = parser.parse_args()
    
    # Calculate max steps based on days
    max_steps = 24 * args.days
    
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", "baseline", f"baseline_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create TensorBoard writer if enabled
    tb_writer = None
    if not args.no_tensorboard:
        tb_writer = create_tensorboard_writer(
            log_dir=os.path.join("logs", "tensorboard", f"baseline_{timestamp}"),
            comment="baseline"
        )
    
    # Set up simulators
    print("Setting up simulators...")
    demand_generator = DemandGenerator()
    supply_simulator = SupplySimulator()
    grid_simulator = GridSimulator()
    pricing_model = PricingModel()
    
    # Run baseline strategies
    print("\nRunning baseline strategies...")
    metrics = run_baseline_strategies(
        demand_generator=demand_generator,
        supply_simulator=supply_simulator,
        grid_simulator=grid_simulator,
        pricing_model=pricing_model,
        num_episodes=args.episodes,
        max_steps=max_steps,
        tb_writer=tb_writer
    )
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Cost', 'Emissions', 'Grid Imbalance', 'Renewable Usage (%)'],
        'Merit Order': [
            f"{metrics['merit_order']['avg_cost']:.2f} ± {metrics['merit_order']['std_cost']:.2f}",
            f"{metrics['merit_order']['avg_emissions']:.2f} ± {metrics['merit_order']['std_emissions']:.2f}",
            f"{metrics['merit_order']['avg_imbalance']:.2f} ± {metrics['merit_order']['std_imbalance']:.2f}",
            f"{metrics['merit_order']['avg_renewable_usage']*100:.2f}% ± {metrics['merit_order']['std_renewable_usage']*100:.2f}%"
        ],
        'Proportional': [
            f"{metrics['proportional']['avg_cost']:.2f} ± {metrics['proportional']['std_cost']:.2f}",
            f"{metrics['proportional']['avg_emissions']:.2f} ± {metrics['proportional']['std_emissions']:.2f}",
            f"{metrics['proportional']['avg_imbalance']:.2f} ± {metrics['proportional']['std_imbalance']:.2f}",
            f"{metrics['proportional']['avg_renewable_usage']*100:.2f}% ± {metrics['proportional']['std_renewable_usage']*100:.2f}%"
        ],
        'Renewable First': [
            f"{metrics['renewable_first']['avg_cost']:.2f} ± {metrics['renewable_first']['std_cost']:.2f}",
            f"{metrics['renewable_first']['avg_emissions']:.2f} ± {metrics['renewable_first']['std_emissions']:.2f}",
            f"{metrics['renewable_first']['avg_imbalance']:.2f} ± {metrics['renewable_first']['std_imbalance']:.2f}",
            f"{metrics['renewable_first']['avg_renewable_usage']*100:.2f}% ± {metrics['renewable_first']['std_renewable_usage']*100:.2f}%"
        ]
    })
    
    metrics_df.to_csv(os.path.join(output_dir, "baseline_metrics.csv"), index=False)
    
    # Print metrics
    print("\nBaseline Strategy Metrics:")
    print(metrics_df)
    
    # Create plots
    for strategy in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(['Cost', 'Emissions', 'Grid Imbalance', 'Renewable Usage (%)'], 
                [metrics[strategy]['avg_cost'], 
                 metrics[strategy]['avg_emissions'], 
                 metrics[strategy]['avg_imbalance'], 
                 metrics[strategy]['avg_renewable_usage']*100],
                yerr=[metrics[strategy]['std_cost'], 
                      metrics[strategy]['std_emissions'], 
                      metrics[strategy]['std_imbalance'], 
                      metrics[strategy]['std_renewable_usage']*100])
        plt.title(f"{strategy.replace('_', ' ').title()} Strategy Metrics")
        plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{strategy}_metrics.png"))
    
    # Close TensorBoard writer
    if tb_writer:
        close_tensorboard_writer(tb_writer)
    
    print(f"\nBaseline evaluation complete. Results saved to {output_dir}")
    
    if not args.no_tensorboard:
        print("\nTo view TensorBoard logs, run:")
        print("tensorboard --logdir=logs/tensorboard")

if __name__ == "__main__":
    # Create required directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs/tensorboard', exist_ok=True)
    os.makedirs('results/baseline', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    main() 