import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 12

def plot_demand_supply(demand_data, supply_data, title="Demand-Supply Analysis", 
                       save_path=None, show=True):
    """
    Plot demand vs supply with generation mix breakdown.
    
    Args:
        demand_data (pd.Series): Time series of energy demand
        supply_data (pd.DataFrame): Time series of supply from different sources
        title (str): Plot title
        save_path (str): Path to save the plot
        show (bool): Whether to display the plot
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 1, figure=fig)
    
    # Main demand-supply plot
    ax1 = fig.add_subplot(gs[:2, :])
    ax1.plot(demand_data.index, demand_data, 'b-', label='Demand')
    ax1.plot(demand_data.index, supply_data.sum(axis=1), 'g--', label='Total Supply')
    ax1.fill_between(demand_data.index, demand_data, supply_data.sum(axis=1), 
                    where=(supply_data.sum(axis=1) >= demand_data),
                    facecolor='lightgreen', alpha=0.3, label='Surplus')
    ax1.fill_between(demand_data.index, demand_data, supply_data.sum(axis=1),
                    where=(supply_data.sum(axis=1) < demand_data),
                    facecolor='salmon', alpha=0.3, label='Deficit')
    ax1.set_title(title)
    ax1.set_ylabel('Power (MW)')
    ax1.legend(loc='upper left')
    
    # Generation mix plot
    ax2 = fig.add_subplot(gs[2, :])
    colors = plt.cm.tab20.colors
    bottom = np.zeros(len(supply_data))
    for i, source in enumerate(supply_data.columns):
        ax2.bar(supply_data.index, supply_data[source], bottom=bottom, 
               label=source, color=colors[i % 20], alpha=0.8)
        bottom += supply_data[source]
    ax2.set_ylabel('Generation Mix')
    ax2.legend(ncol=3, loc='upper right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_market_prices(price_data, volatility_window=24, title="Market Prices Analysis",
                       save_path=None, show=True):
    """
    Plot electricity prices with volatility and statistical analysis.
    
    Args:
        price_data (pd.Series): Time series of electricity prices
        volatility_window (int): Window for rolling volatility calculation
        title (str): Plot title
        save_path (str): Path to save the plot
        show (bool): Whether to display the plot
    """
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    
    # Price timeline
    axs[0].plot(price_data.index, price_data, color='purple')
    axs[0].set_title(f'{title} - Price Timeline')
    axs[0].set_ylabel('Price ($/MWh)')
    
    # Volatility analysis
    rolling_vol = price_data.rolling(volatility_window).std()
    axs[1].plot(price_data.index, rolling_vol, color='darkorange')
    axs[1].set_title(f'{title} - {volatility_window}-hour Rolling Volatility')
    axs[1].set_ylabel('Volatility')
    
    # Price distribution
    sns.histplot(price_data, kde=True, ax=axs[2], color='teal')
    axs[2].set_title(f'{title} - Price Distribution')
    axs[2].set_xlabel('Price ($/MWh)')
    axs[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_environment_metrics(simulation_results, title="Simulation Metrics", 
                            save_path=None, show=True):
    """
    Plot key simulation metrics from grid operations.
    
    Args:
        simulation_results (pd.DataFrame): DataFrame containing simulation metrics
        title (str): Plot title
        save_path (str): Path to save the plot
        show (bool): Whether to display the plot
    """
    fig, axs = plt.subplots(4, 1, figsize=(15, 16))
    
    # Cost and Emissions
    axs[0].plot(simulation_results['total_cost'], color='red', label='Total Cost')
    axs[0].set_ylabel('Cost ($)', color='red')
    ax2 = axs[0].twinx()
    ax2.plot(simulation_results['total_emissions'], color='green', label='Total Emissions')
    ax2.set_ylabel('Emissions (tons CO2)', color='green')
    axs[0].set_title(f'{title} - Costs & Emissions')
    
    # Grid Stability
    axs[1].plot(simulation_results['grid_stability'], color='blue')
    axs[1].axhline(0.5, color='red', linestyle='--', label='Stability Threshold')
    axs[1].set_ylabel('Grid Stability (0-1)')
    axs[1].set_title(f'{title} - Grid Stability')
    axs[1].legend()
    
    # Renewable Penetration
    axs[2].plot(simulation_results['renewable_fraction']*100, color='green')
    axs[2].set_ylabel('Renewable (%)')
    axs[2].set_ylim(0, 100)
    axs[2].set_title(f'{title} - Renewable Energy Penetration')
    
    # Energy Balance
    axs[3].plot(simulation_results['energy_balance'], color='purple')
    axs[3].axhline(0, color='black', linestyle='--')
    axs[3].set_ylabel('Energy Balance (MW)')
    axs[3].set_title(f'{title} - Energy Balance (Supply - Demand)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_rl_training(metrics, window=100, title="RL Training Progress", 
                    save_path=None, show=True):
    """
    Visualize RL training metrics with smoothing.
    
    Args:
        metrics (dict): Dictionary of training metrics
        window (int): Smoothing window for metrics
        title (str): Plot title
        save_path (str): Path to save the plot
        show (bool): Whether to display the plot
    """
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    
    # Episode Returns
    returns = pd.Series(metrics['episode_returns'])
    axs[0].plot(returns.rolling(window).mean(), label='Smoothed')
    axs[0].plot(returns, alpha=0.3, label='Raw')
    axs[0].set_ylabel('Episode Return')
    axs[0].set_title(f'{title} - Episode Returns')
    axs[0].legend()
    
    # Losses
    if 'policy_loss' in metrics:
        axs[1].plot(pd.Series(metrics['policy_loss']).rolling(window).mean(), label='Policy Loss')
    if 'value_loss' in metrics:
        axs[1].plot(pd.Series(metrics['value_loss']).rolling(window).mean(), label='Value Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_title(f'{title} - Training Losses')
    axs[1].legend()
    
    # Additional Metrics
    if 'entropy' in metrics:
        axs[2].plot(pd.Series(metrics['entropy']).rolling(window).mean(), label='Entropy')
    if 'alpha' in metrics:
        axs[2].plot(pd.Series(metrics['alpha']).rolling(window).mean(), label='Alpha')
    axs[2].set_ylabel('Metric Value')
    axs[2].set_title(f'{title} - Additional Metrics')
    axs[2].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_action_distribution(actions, source_names, title="Action Distribution",
                            save_path=None, show=True):
    """
    Visualize the distribution of actions from RL policy.
    
    Args:
        actions (np.ndarray): Array of actions from policy
        source_names (list): List of energy source names
        title (str): Plot title
        save_path (str): Path to save the plot
        show (bool): Whether to display the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Convert actions to DataFrame for easier plotting
    action_df = pd.DataFrame(actions, columns=source_names)
    
    # Create boxplot
    sns.boxplot(data=action_df, palette="viridis")
    plt.xticks(rotation=45)
    plt.title(f'{title} - Action Distribution by Source')
    plt.ylabel('Allocation Percentage')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_correlation_heatmap(data, title="Feature Correlation", 
                             save_path=None, show=True):
    """
    Plot correlation heatmap for market features.
    
    Args:
        data (pd.DataFrame): DataFrame containing market features
        title (str): Plot title
        save_path (str): Path to save the plot
        show (bool): Whether to display the plot
    """
    plt.figure(figsize=(14, 12))
    
    # Calculate correlations
    corr = data.corr()
    
    # Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
               cbar_kws={'label': 'Correlation Coefficient'})
    plt.title(f'{title} - Feature Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_comparison(agent_metrics, metric_name, save_path=None, title=None, window_size=10):
    """
    Plot comparison of a specific metric across different agents.
    
    Args:
        agent_metrics (dict): Dictionary with agent names as keys and metrics as values
        metric_name (str): Name of the metric to compare
        save_path (str): Path to save the plot
        title (str): Plot title
        window_size (int): Window size for smoothing
    """
    plt.figure(figsize=(12, 8))
    
    # Map common metric names to their possible variations in different agents
    metric_mapping = {
        'rewards': ['episode_rewards', 'episode_returns'],
        'policy_loss': ['policy_losses', 'actor_losses'],
        'value_loss': ['value_losses', 'critic_losses'],
        'entropy': ['entropies'],
        'kl': ['kls'],
        'explained_variance': ['explained_variances'],
        'alpha': ['alphas', 'alpha_values']
    }
    
    # Get all possible metric keys for the requested metric
    possible_keys = metric_mapping.get(metric_name, [metric_name])
    
    for agent_name, metrics in agent_metrics.items():
        # Try each possible key for the requested metric
        for key in possible_keys:
            if key in metrics:
                values = metrics[key]
                # Plot raw values with low alpha
                plt.plot(values, alpha=0.3, label=f"{agent_name} (raw)")
                
                # Plot smoothed values if there are enough data points
                if len(values) > window_size:
                    smoothed = pd.Series(values).rolling(window_size).mean()
                    plt.plot(smoothed, linewidth=2, label=f"{agent_name} (smoothed)")
                break  # Use the first matching key
    
    plt.title(title or f"Comparison of {metric_name}")
    plt.xlabel("Steps")
    plt.ylabel(metric_name.capitalize())
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_agent_metrics(ppo_metrics, sac_metrics, title="PPO vs SAC Performance", save_path=None):
    """
    Plot comparison between PPO and SAC agents.
    
    Args:
        ppo_metrics (dict): Metrics from PPO agent evaluation
        sac_metrics (dict): Metrics from SAC agent evaluation
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    metrics = ['avg_reward', 'avg_cost', 'avg_emissions', 'avg_imbalance', 'avg_renewable_usage']
    metric_labels = ['Reward', 'Cost', 'Emissions', 'Grid Imbalance', 'Renewable Usage']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        # Get values and errors
        ppo_val = ppo_metrics[metric]
        ppo_err = ppo_metrics[f'std_{metric.split("_")[1]}']
        sac_val = sac_metrics[metric]
        sac_err = sac_metrics[f'std_{metric.split("_")[1]}']
        
        # Create bar chart
        x = np.arange(2)
        width = 0.35
        
        ax.bar(x[0], ppo_val, width, yerr=ppo_err, label='PPO', color='#1f77b4', alpha=0.7)
        ax.bar(x[1], sac_val, width, yerr=sac_err, label='SAC', color='#ff7f0e', alpha=0.7)
        
        # Add labels and title
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(['PPO', 'SAC'])
        
        # Only add legend to the first subplot
        if i == 0:
            ax.legend()
    
    plt.suptitle(title, size=16)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_curves(training_history, save_path=None, title="Training Progress"):
    """
    Plot training curves for RL agents.
    
    Args:
        training_history (dict): Dictionary containing training metrics
        save_path (str): Path to save the plot
        title (str): Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot reward
    ax = axes[0, 0]
    # Check for both episode_rewards and episode_returns (for compatibility with different agents)
    if 'episode_rewards' in training_history:
        rewards = training_history['episode_rewards']
        ax.plot(rewards, label='Episode Reward')
        # Add smoothed curve if there are enough points
        if len(rewards) > 10:
            smoothed = pd.Series(rewards).rolling(10).mean()
            ax.plot(smoothed, label='Smoothed (10)', color='red')
    elif 'episode_returns' in training_history:
        rewards = training_history['episode_returns']
        ax.plot(rewards, label='Episode Return')
        # Add smoothed curve if there are enough points
        if len(rewards) > 10:
            smoothed = pd.Series(rewards).rolling(10).mean()
            ax.plot(smoothed, label='Smoothed (10)', color='red')
    ax.set_title('Reward per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    
    # Plot loss
    ax = axes[0, 1]
    if 'policy_losses' in training_history:
        policy_losses = training_history['policy_losses']
        ax.plot(policy_losses, label='Policy Loss')
        # Add smoothed curve if there are enough points
        if len(policy_losses) > 10:
            smoothed = pd.Series(policy_losses).rolling(10).mean()
            ax.plot(smoothed, label='Smoothed Policy Loss', linestyle='--', color='blue')
    if 'value_losses' in training_history:
        value_losses = training_history['value_losses']
        ax.plot(value_losses, label='Value Loss')
        # Add smoothed curve if there are enough points
        if len(value_losses) > 10:
            smoothed = pd.Series(value_losses).rolling(10).mean()
            ax.plot(smoothed, label='Smoothed Value Loss', linestyle='--', color='green')
    if 'actor_losses' in training_history:
        actor_losses = training_history['actor_losses']
        ax.plot(actor_losses, label='Actor Loss')
        # Add smoothed curve if there are enough points
        if len(actor_losses) > 10:
            smoothed = pd.Series(actor_losses).rolling(10).mean()
            ax.plot(smoothed, label='Smoothed Actor Loss', linestyle='--', color='orange')
    if 'critic_losses' in training_history:
        critic_losses = training_history['critic_losses']
        ax.plot(critic_losses, label='Critic Loss')
        # Add smoothed curve if there are enough points
        if len(critic_losses) > 10:
            smoothed = pd.Series(critic_losses).rolling(10).mean()
            ax.plot(smoothed, label='Smoothed Critic Loss', linestyle='--', color='red')
    ax.set_title('Loss Components')
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    
    # Plot evaluation returns
    ax = axes[1, 0]
    if 'eval_returns' in training_history and 'eval_timesteps' in training_history:
        ax.plot(training_history['eval_timesteps'], training_history['eval_returns'], 
                label='Evaluation Return', marker='o')
    ax.set_title('Evaluation Returns')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    
    # Plot additional metrics
    ax = axes[1, 1]
    if 'entropies' in training_history:
        ax.plot(training_history['entropies'], label='Entropy')
    if 'kls' in training_history:
        ax.plot(training_history['kls'], label='KL Divergence')
    if 'explained_variances' in training_history:
        ax.plot(training_history['explained_variances'], label='Explained Variance')
    if 'alpha_values' in training_history:
        ax.plot(training_history['alpha_values'], label='Alpha')
    elif 'alphas' in training_history:
        ax.plot(training_history['alphas'], label='Alpha')
    ax.set_title('Additional Metrics')
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    
    plt.suptitle(title, size=16)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_comprehensive_comparison(agent_metrics, save_path=None, title="Agent Comparison"):
    """
    Plot comprehensive comparison between different RL agents with multiple metrics.
    
    Args:
        agent_metrics (dict): Dictionary with agent names as keys and training metrics as values
        save_path (str): Path to save the plot
        title (str): Plot title
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Define colors for different agents
    colors = {
        'ppo': '#1f77b4',  # blue
        'sac': '#ff7f0e',  # orange
        'ddpg': '#2ca02c', # green
        'td3': '#d62728'   # red
    }
    
    # Metric mappings for different agent implementations
    metric_mappings = {
        'rewards': ['episode_rewards', 'episode_returns'],
        'policy_loss': ['policy_losses', 'actor_losses'],
        'value_loss': ['value_losses', 'critic_losses'],
        'entropy': ['entropies'],
        'alpha': ['alphas', 'alpha_values']
    }
    
    # Plot rewards
    ax = axes[0, 0]
    for agent_name, metrics in agent_metrics.items():
        color = colors.get(agent_name.lower(), None)
        # Try different reward keys
        for key in metric_mappings['rewards']:
            if key in metrics and len(metrics[key]) > 0:
                rewards = pd.Series(metrics[key])
                ax.plot(rewards, alpha=0.3, color=color)
                if len(rewards) > 10:
                    ax.plot(rewards.rolling(10).mean(), 
                           label=f'{agent_name.upper()} Reward', 
                           color=color,
                           linewidth=2)
                break
    ax.set_title('Reward per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    
    # Plot policy losses
    ax = axes[0, 1]
    for agent_name, metrics in agent_metrics.items():
        color = colors.get(agent_name.lower(), None)
        # Try different policy loss keys
        for key in metric_mappings['policy_loss']:
            if key in metrics and len(metrics[key]) > 0:
                losses = pd.Series(metrics[key])
                ax.plot(losses, alpha=0.3, color=color)
                if len(losses) > 10:
                    ax.plot(losses.rolling(10).mean(), 
                           label=f'{agent_name.upper()} Policy Loss', 
                           color=color,
                           linewidth=2)
                break
    ax.set_title('Policy Losses')
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    
    # Plot value losses
    ax = axes[0, 2]
    for agent_name, metrics in agent_metrics.items():
        color = colors.get(agent_name.lower(), None)
        # Try different value loss keys
        for key in metric_mappings['value_loss']:
            if key in metrics and len(metrics[key]) > 0:
                losses = pd.Series(metrics[key])
                ax.plot(losses, alpha=0.3, color=color)
                if len(losses) > 10:
                    ax.plot(losses.rolling(10).mean(), 
                           label=f'{agent_name.upper()} Value Loss', 
                           color=color,
                           linewidth=2)
                break
    ax.set_title('Value Losses')
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    
    # Plot evaluation returns
    ax = axes[1, 0]
    for agent_name, metrics in agent_metrics.items():
        if 'eval_returns' in metrics and 'eval_timesteps' in metrics and len(metrics['eval_returns']) > 0:
            eval_returns = metrics['eval_returns']
            eval_steps = metrics['eval_timesteps']
            ax.plot(eval_steps, eval_returns, 
                   label=f'{agent_name.upper()} Eval Return', 
                   color=colors.get(agent_name.lower(), None),
                   marker='o')
    ax.set_title('Evaluation Returns')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    
    # Plot entropy/alpha
    ax = axes[1, 1]
    for agent_name, metrics in agent_metrics.items():
        color = colors.get(agent_name.lower(), None)
        # Try entropy
        if 'entropies' in metrics and len(metrics['entropies']) > 0:
            entropies = pd.Series(metrics['entropies'])
            ax.plot(entropies, alpha=0.3, color=color)
            if len(entropies) > 10:
                ax.plot(entropies.rolling(10).mean(), 
                       label=f'{agent_name.upper()} Entropy', 
                       color=color,
                       linewidth=2)
        
        # Try alpha values
        for key in metric_mappings['alpha']:
            if key in metrics and len(metrics[key]) > 0:
                alphas = pd.Series(metrics[key])
                ax.plot(alphas, alpha=0.3, color=color, linestyle='--')
                if len(alphas) > 10:
                    ax.plot(alphas.rolling(10).mean(), 
                           label=f'{agent_name.upper()} Alpha', 
                           color=color,
                           linewidth=2,
                           linestyle='--')
                break
    ax.set_title('Entropy and Alpha')
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    
    # Plot episode lengths
    ax = axes[1, 2]
    for agent_name, metrics in agent_metrics.items():
        if 'episode_lengths' in metrics and len(metrics['episode_lengths']) > 0:
            lengths = pd.Series(metrics['episode_lengths'])
            ax.plot(lengths, alpha=0.3, color=colors.get(agent_name.lower(), None))
            if len(lengths) > 10:
                ax.plot(lengths.rolling(10).mean(), 
                       label=f'{agent_name.upper()} Episode Length', 
                       color=colors.get(agent_name.lower(), None),
                       linewidth=2)
    ax.set_title('Episode Lengths')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length')
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    
    plt.suptitle(title, size=16)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    """Test visualization functions with sample data."""
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=24*7, freq='H')
    demand = 1000 + 500 * np.sin(np.linspace(0, 2*np.pi, 24*7))
    
    supply = pd.DataFrame({
        'Solar': 300 * np.clip(np.sin(np.linspace(-np.pi/2, 3*np.pi/2, 24*7)), 0, 1),
        'Wind': 200 * np.random.rand(24*7),
        'Gas': np.clip(demand - 300 - 200, 0, 600),
        'Coal': np.full(24*7, 200)
    }, index=dates)
    
    prices = 50 + 20 * np.random.randn(24*7) + 0.5*(demand - supply.sum(axis=1))
    
    # Create sample simulation results
    sim_results = pd.DataFrame({
        'total_cost': supply.sum(axis=1) * 0.05,
        'total_emissions': supply['Gas']*0.4 + supply['Coal']*0.9,
        'grid_stability': 0.8 + 0.2*np.random.rand(24*7),
        'renewable_fraction': (supply['Solar'] + supply['Wind']) / supply.sum(axis=1),
        'energy_balance': supply.sum(axis=1) - demand
    }, index=dates)
    
    # Test visualizations
    plot_demand_supply(pd.Series(demand, index=dates), supply, show=False)
    plot_market_prices(pd.Series(prices, index=dates), show=False)
    plot_environment_metrics(sim_results, show=False)
    
    # Create sample RL metrics
    rl_metrics = {
        'episode_returns': np.random.randn(1000).cumsum(),
        'policy_loss': np.abs(np.random.randn(1000)),
        'value_loss': np.abs(np.random.randn(1000)),
        'entropy': np.random.rand(1000),
        'alpha': np.linspace(0.2, 0.1, 1000)
    }
    plot_rl_training(rl_metrics, show=False)
    
    # Sample action distribution
    actions = np.random.dirichlet(np.ones(4), size=1000)
    plot_action_distribution(actions, ['Solar', 'Wind', 'Gas', 'Coal'], show=False)
    
    # Correlation heatmap
    plot_correlation_heatmap(sim_results, show=False)

if __name__ == "__main__":
    main()
