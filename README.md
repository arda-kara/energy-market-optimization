# AI-Powered Energy Market Optimization

This project implements an AI-driven energy distribution optimization system using Reinforcement Learning (RL). The system dynamically allocates electricity supply from multiple sources (solar, wind, hydro, fossil fuels) based on real-time demand and pricing models to optimize for cost, emissions, and grid stability.

## Project Overview

The Energy Market Optimization project simulates an electricity market where an RL agent must make decisions about energy allocation from different sources to meet demand while optimizing multiple objectives:

1. **Minimizing Cost**: Reduce the overall cost of energy production
2. **Reducing Emissions**: Minimize carbon emissions from energy generation
3. **Maintaining Grid Stability**: Ensure the grid remains stable and reliable
4. **Maximizing Renewable Usage**: Prioritize renewable energy sources when possible

The project uses custom implementations of state-of-the-art RL algorithms (PPO and SAC) to learn optimal allocation strategies in this complex environment.

## Project Structure

- `market_simulator/`: Energy market simulation components
  - `demand_generator.py`: Generates synthetic energy demand data
  - `supply_simulator.py`: Simulates energy supply from different sources
  - `pricing_model.py`: Implements dynamic pricing based on demand-supply balance
  - `grid_simulator.py`: Simulates grid balancing and stability

- `rl_environment/`: Reinforcement learning environment
  - `energy_env.py`: Custom OpenAI Gym environment for energy allocation
  - `reward_functions.py`: Different reward functions for the RL agents

- `agents/`: Implementation of RL agents
  - `ppo_agent.py`: Proximal Policy Optimization (PPO) agent implementation
  - `sac_agent.py`: Soft Actor-Critic (SAC) agent implementation

- `utils/`: Utility functions and helpers
  - `data_processing.py`: Functions for data processing and transformation
  - `visualization.py`: Functions for visualizing results and agent performance

- `experiments/`: Scripts for running experiments and evaluations
  - `run_experiment.py`: Script for running experiments with different agents
  - `agent_comparison.py`: Script for comparing different agent implementations
  - `hyperparameter_tuning.py`: Script for optimizing agent hyperparameters
  - `baseline.py`: Implementation of baseline allocation strategies

- `models/`: Directory for storing trained models
- `results/`: Directory for storing experiment results
- `plots/`: Directory for storing visualizations

## Key Features

- **Custom RL Environment**: A flexible OpenAI Gym environment that simulates an energy market
- **Multiple RL Algorithms**: Implementations of PPO and SAC algorithms for comparison
- **Hyperparameter Optimization**: Automated tuning of agent parameters using Optuna
- **Agent Comparison**: Tools for comparing different RL algorithms on the same environment
- **Visualization Tools**: Comprehensive visualization of training metrics and agent performance
- **Experiment Framework**: Flexible framework for running and analyzing experiments

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/energy-market-optimization.git
cd energy-market-optimization

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
# This is important to ensure Python can find all the modules
python install_dev.py
```

If you encounter import errors when running scripts, make sure you've installed the package in development mode as shown above.

## Usage

### Running Experiments

The `run_experiment.py` script provides a flexible way to run experiments with different agents:

```bash
# Run an experiment with both PPO and SAC agents
python experiments/run_experiment.py

# Run an experiment with only the PPO agent
python experiments/run_experiment.py --agents ppo

# Run an experiment with more timesteps
python experiments/run_experiment.py --timesteps 200000

# Run an experiment with a longer episode length (in hours)
python experiments/run_experiment.py --episode-length 168

# Run an experiment with more evaluation episodes
python experiments/run_experiment.py --eval-episodes 20

# Run an experiment with a specific random seed
python experiments/run_experiment.py --seed 123
```

### Comparing Agents

The `agent_comparison.py` script allows for detailed comparison between different agents:

```bash
# Run the agent comparison with default settings
python experiments/agent_comparison.py

# Run the agent comparison with only specific agents
python experiments/agent_comparison.py --agents ppo sac

# Run the agent comparison with more timesteps
python experiments/agent_comparison.py --timesteps 200000

# Run the agent comparison with a longer episode length (in hours)
python experiments/agent_comparison.py --episode-length 168

# Run the agent comparison with more evaluation episodes
python experiments/agent_comparison.py --eval-episodes 20
```

This will train the specified agents with their configurations and compare their performance. The results will be saved in the `results/` directory, including training metrics, evaluation returns, and comparison plots.

### Using TensorBoard for Visualization

The project supports TensorBoard for real-time visualization of training metrics. To use TensorBoard:
```bash
pip install tensorboard
```

2. Run an experiment with TensorBoard enabled:
```bash
# Run an experiment with TensorBoard logging
python experiments/run_experiment.py --use-tensorboard

# Run agent comparison with TensorBoard logging
python experiments/agent_comparison.py --use-tensorboard

# Run hyperparameter tuning with TensorBoard logging
python experiments/hyperparameter_tuning.py --use-tensorboard
```

3. Start TensorBoard to view the metrics:
```bash
# Using the provided script
python run_tensorboard.py

# Or directly with the tensorboard command
tensorboard --logdir=./logs/tensorboard
```

4. Open your browser and navigate to `http://localhost:6006` to view the TensorBoard dashboard.

TensorBoard provides real-time visualization of:
- Training rewards and episode lengths
- Policy and value losses
- Entropy and KL divergence
- Evaluation returns
- Network gradients and weights

### Hyperparameter Tuning

The `hyperparameter_tuning.py` script uses Optuna to find optimal hyperparameters for the agents:

```bash
# Run hyperparameter tuning for both PPO and SAC agents
python experiments/hyperparameter_tuning.py

# Run hyperparameter tuning for a specific agent
python experiments/hyperparameter_tuning.py --agent ppo

# Run hyperparameter tuning with more trials
python experiments/hyperparameter_tuning.py --n_trials 50

# Run hyperparameter tuning with a time limit
python experiments/hyperparameter_tuning.py --timeout 3600  # 1 hour
```

## Reinforcement Learning Environment

The `EnergyMarketEnv` class in `rl_environment/energy_env.py` implements a custom OpenAI Gym environment for energy market optimization:

### State Space

The environment state includes:
- Current energy demand
- Forecasted demand for future time steps
- Available capacity for each energy source
- Current electricity price
- Time features (hour of day, day of week)
- Grid stability metrics

### Action Space

The action space consists of allocation percentages for each energy source, determining how much energy to draw from each source to meet the current demand.

### Reward Function

The reward function balances multiple objectives:
- Cost minimization: Reduce the overall cost of energy production
- Emission reduction: Minimize carbon emissions
- Grid stability: Maintain a stable and reliable grid
- Demand satisfaction: Ensure demand is met without shortages or excessive surplus

## RL Agents

### PPO Agent

The Proximal Policy Optimization (PPO) agent in `agents/ppo_agent.py` implements the PPO algorithm with the following components:

- **Actor-Critic Network**: A neural network with shared layers for policy and value functions
- **PPO Buffer**: A buffer for storing trajectories and computing advantages
- **Policy Update**: Implementation of the PPO clipped objective function

### SAC Agent

The Soft Actor-Critic (SAC) agent in `agents/sac_agent.py` implements the SAC algorithm with the following components:

- **Actor Network**: A neural network for the policy function
- **Critic Networks**: Dual Q-networks for value estimation
- **Replay Buffer**: A buffer for storing and sampling transitions
- **Entropy Regularization**: Automatic tuning of the entropy coefficient

## Visualization

The project includes comprehensive visualization tools in `utils/visualization.py`:

- **Training Curves**: Visualize rewards, losses, and other metrics during training
- **Agent Comparison**: Compare performance between different agents
- **Comprehensive Comparison**: Compare multiple metrics across different agents in a single plot
- **Action Distribution**: Analyze the distribution of actions taken by agents
- **Environment Metrics**: Visualize cost, emissions, grid stability, and renewable usage

### Visualizing Existing Results

The `visualize_results.py` script allows you to generate visualizations from existing experiment results:

```bash
# Visualize results from a specific experiment
python -m utils.visualize_results --results-dir ./results/experiment_20230615_120000

# Save visualizations to a different directory
python -m utils.visualize_results --results-dir ./results/experiment_20230615_120000 --output-dir ./plots/experiment_analysis

# Specify which metrics to compare
python -m utils.visualize_results --results-dir ./results/experiment_20230615_120000 --metrics rewards policy_loss value_loss entropy
```

This script will generate:
1. Individual training curves for each agent
2. A comprehensive comparison of all agents (if multiple agents are present)
3. Individual metric comparisons for each specified metric

### Visualization Functions

The project provides several visualization functions:

- `plot_training_curves`: Plots training curves for a single agent, including rewards, losses, and other metrics
- `plot_comparison`: Plots a comparison of a specific metric across different agents
- `plot_comprehensive_comparison`: Plots a comprehensive comparison of multiple metrics across different agents

These functions can be used directly in your code or through the `visualize_results.py` script.

## Results and Analysis

After running experiments, results are saved in the `results/` directory, including:

- Training metrics (rewards, losses, etc.)
- Evaluation returns
- Comparison between agents
- Hyperparameter optimization results

## Project Phases and Current Status

1. **Baseline Energy Market Simulation** ✅
   - Generate synthetic energy demand data
   - Implement basic grid balancing simulation
   - Define initial energy allocation policy

2. **Reinforcement Learning Environment Setup** ✅
   - Create custom RL environment
   - Define observation and action spaces
   - Implement basic reward function

3. **AI Model Training & Optimization** ✅
   - Implement PPO/SAC-based RL agents
   - Train models using simulated scenarios
   - Introduce real-world constraints

4. **Validation & Deployment** ✅
   - Evaluate AI vs traditional distribution
   - Optimize hyperparameters
   - Test on real-world datasets

5. **Advanced Optimization Strategies** ✅
   - Implement objective-based dispatch optimization
   - Add progress tracking for long simulations
   - Enhance grid stability modeling

## Recent Improvements

- **Enhanced Agent Implementations**: Updated PPO and SAC implementations for better performance
- **Improved Experiment Framework**: Created a flexible framework for running and analyzing experiments
- **Hyperparameter Tuning**: Added automated hyperparameter optimization using Optuna
- **Agent Comparison**: Added tools for comparing different RL algorithms
- **Visualization Enhancements**: Improved visualization of training metrics and agent performance
- **TensorBoard Integration**: Added support for real-time visualization of training metrics using TensorBoard
- **Detailed Logging**: Enhanced logging to provide more information about training progress

## Troubleshooting

### Common Issues

1. **Missing directories**: The scripts will automatically create required directories, but if you encounter errors, make sure the following directories exist:
   ```
   plots/
   data/
   results/
   models/
   logs/
   ```

2. **CUDA out of memory**: If you encounter CUDA out of memory errors, try reducing the batch size or model size.

3. **NaN values in training**: If you see NaN values during training, try reducing the learning rate or adjusting the clipping parameters.

4. **DataFrame concatenation warnings**: If you see warnings about DataFrame concatenation with empty or NA entries, these have been addressed in the latest version and can be safely ignored.

5. **Matrix shape errors**: If you encounter matrix shape errors, ensure that your state and action dimensions match the environment's observation and action spaces.

## Future Work

- **Energy Storage Integration**: Add support for battery storage systems
- **Demand Response Modeling**: Implement demand-side management strategies
- **Multi-Region Simulation**: Extend the model to handle multiple interconnected regions
- **Real-Time Optimization**: Improve performance for real-time decision making
- **Advanced ML Models**: Explore transformer-based models for time-series forecasting
- **Multi-Agent Systems**: Implement multi-agent reinforcement learning for distributed control

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym for the reinforcement learning environment framework
- PyTorch for the deep learning framework
- Optuna for hyperparameter optimization

## Code Examples

### Data Generation for Testing

The project includes utilities for generating synthetic data for testing and development:

```python
# Generate synthetic demand data
from market_simulator.demand_generator import generate_demand_data

# Generate 7 days of hourly demand data
demand_data = generate_demand_data(
    days=7,
    base_demand=1000,  # Base demand in MW
    daily_pattern=True,  # Include daily patterns
    weekly_pattern=True,  # Include weekly patterns
    noise_level=0.1  # Add 10% random noise
)

# Save the generated data
import pandas as pd
pd.DataFrame(demand_data, columns=['Demand']).to_csv('data/synthetic_demand.csv')
```

```python
# Generate synthetic supply data
from market_simulator.supply_simulator import generate_supply_data

# Generate supply data for different energy sources
supply_data = generate_supply_data(
    days=7,
    sources=['solar', 'wind', 'hydro', 'gas', 'coal'],
    capacities=[500, 400, 300, 800, 600],  # Capacities in MW
    availability_patterns={
        'solar': 'daily',  # Solar follows daily pattern
        'wind': 'random',  # Wind is more random
        'hydro': 'stable',  # Hydro is stable
        'gas': 'on_demand',  # Gas can be used on demand
        'coal': 'on_demand'  # Coal can be used on demand
    }
)

# Save the generated data
import pandas as pd
pd.DataFrame(supply_data).to_csv('data/synthetic_supply.csv')
```

### Running the Environment

```python
# Create and run the environment
from rl_environment.energy_env import EnergyMarketEnv

# Create environment
env = EnergyMarketEnv(
    episode_length=24*7,  # 7 days
    normalize_state=True,
    reward_function='balanced'  # 'balanced', 'cost_focused', 'emission_focused'
)

# Reset environment
state, _ = env.reset()

# Run a simple loop
total_reward = 0
for _ in range(24*7):  # 7 days
    # Take a random action
    action = env.action_space.sample()
    
    # Step the environment
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Update state and reward
    state = next_state
    total_reward += reward
    
    # Check if episode is done
    if terminated or truncated:
        break

print(f"Total reward: {total_reward}")
```

### Training Agents

#### Training a PPO Agent

```python
# Train a PPO agent
from agents.ppo_agent import PPOAgent
from rl_environment.energy_env import EnergyMarketEnv
import torch

# Create environment
env = EnergyMarketEnv(episode_length=24*7, normalize_state=True)

# Create PPO agent
ppo_agent = PPOAgent(
    env=env,
    hidden_dim=256,
    lr=3e-4,
    gamma=0.99,
    clip_ratio=0.2,
    target_kl=0.01,
    entropy_coef=0.01
)

# Train the agent
ppo_metrics = ppo_agent.train(
    env=env,
    total_timesteps=100000,
    buffer_size=2048,
    batch_size=64,
    update_epochs=10,
    num_eval_episodes=5,
    eval_freq=10000,
    save_freq=50000,
    save_path='models/ppo_agent',
    tb_writer=None  # Set to a TensorBoard writer if needed
)

# Save the trained agent
ppo_agent.save('models/ppo_agent_final.pt')

# Evaluate the agent
eval_return = ppo_agent.evaluate(env, num_episodes=10)
print(f"PPO evaluation return: {eval_return}")
```

#### Training a SAC Agent

```python
# Train a SAC agent
from agents.sac_agent import SACAgent
from rl_environment.energy_env import EnergyMarketEnv
import torch

# Create environment
env = EnergyMarketEnv(episode_length=24*7, normalize_state=True)

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

# Train the agent
sac_metrics = sac_agent.train(
    env=env,
    total_timesteps=100000,
    batch_size=256,
    buffer_size=1000000,
    update_after=1000,
    update_every=50,
    num_eval_episodes=5,
    eval_freq=10000,
    save_freq=50000,
    save_path='models/sac_agent',
    tb_writer=None  # Set to a TensorBoard writer if needed
)

# Save the trained agent
sac_agent.save('models/sac_agent_final.pt')

# Evaluate the agent
eval_return = sac_agent.evaluate(env, num_episodes=10)
print(f"SAC evaluation return: {eval_return}")
```

### Visualization Examples

#### Plotting Training Curves

```python
# Plot training curves for a single agent
from utils.visualization import plot_training_curves
import pickle

# Load metrics from a file
with open('results/sac_metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

# Plot training curves
plot_training_curves(
    metrics,
    save_path='plots/sac_training_curves.png',
    title='SAC Training Curves'
)
```

#### Comparing Agents

```python
# Compare different agents
from utils.visualization import plot_comparison, plot_comprehensive_comparison
import pickle

# Load metrics for different agents
agent_metrics = {}
with open('results/ppo_metrics.pkl', 'rb') as f:
    agent_metrics['ppo'] = pickle.load(f)
with open('results/sac_metrics.pkl', 'rb') as f:
    agent_metrics['sac'] = pickle.load(f)

# Plot comprehensive comparison
plot_comprehensive_comparison(
    agent_metrics,
    save_path='plots/agent_comprehensive_comparison.png',
    title='Comprehensive Agent Comparison'
)

# Plot comparison of specific metrics
metrics_to_compare = ['rewards', 'policy_loss', 'value_loss']
for metric in metrics_to_compare:
    plot_comparison(
        agent_metrics,
        metric,
        save_path=f'plots/agent_comparison_{metric}.png',
        title=f'Agent Comparison - {metric.capitalize()}'
    )
```

#### Visualizing Environment Metrics

```python
# Visualize environment metrics
from utils.visualization import plot_environment_metrics
import pandas as pd

# Load or create simulation results
simulation_results = {
    'demand': [1000, 1100, 950, 1050, 1200],
    'supply': [1050, 1150, 1000, 1100, 1250],
    'prices': [50, 55, 48, 52, 60],
    'emissions': [500, 550, 480, 520, 600],
    'renewable_percentage': [0.3, 0.35, 0.28, 0.32, 0.4]
}

# Plot environment metrics
plot_environment_metrics(
    simulation_results,
    title='Simulation Metrics',
    save_path='plots/environment_metrics.png'
)
```

#### Visualizing Demand-Supply Balance

```python
# Visualize demand-supply balance
from utils.visualization import plot_demand_supply
import numpy as np

# Generate or load demand and supply data
hours = 24 * 7  # One week
demand_data = 1000 + 200 * np.sin(np.linspace(0, 14*np.pi, hours)) + 50 * np.random.randn(hours)
supply_data = 1050 + 150 * np.sin(np.linspace(0, 14*np.pi, hours) + np.pi/4) + 50 * np.random.randn(hours)

# Plot demand-supply balance
plot_demand_supply(
    demand_data,
    supply_data,
    title='Demand-Supply Analysis',
    save_path='plots/demand_supply_balance.png'
)
```

#### Visualizing Market Prices

```python
# Visualize market prices
from utils.visualization import plot_market_prices
import numpy as np

# Generate or load price data
hours = 24 * 30  # One month
price_data = 50 + 10 * np.sin(np.linspace(0, 60*np.pi, hours)) + 5 * np.random.randn(hours)

# Plot market prices
plot_market_prices(
    price_data,
    volatility_window=24,  # 24-hour window for volatility calculation
    title='Market Prices Analysis',
    save_path='plots/market_prices.png'
)
```

#### Visualizing Action Distribution

```python
# Visualize action distribution
from utils.visualization import plot_action_distribution
import numpy as np

# Generate or load action data
num_actions = 100
num_sources = 5
actions = np.random.rand(num_actions, num_sources)
actions = actions / actions.sum(axis=1, keepdims=True)  # Normalize to sum to 1
source_names = ['Solar', 'Wind', 'Hydro', 'Gas', 'Coal']

# Plot action distribution
plot_action_distribution(
    actions,
    source_names,
    title='Action Distribution',
    save_path='plots/action_distribution.png'
)
```

#### Visualizing Feature Correlation

```python
# Visualize feature correlation
from utils.visualization import plot_correlation_heatmap
import pandas as pd
import numpy as np

# Generate or load feature data
num_samples = 100
data = pd.DataFrame({
    'Demand': 1000 + 200 * np.random.randn(num_samples),
    'Price': 50 + 10 * np.random.randn(num_samples),
    'Temperature': 20 + 5 * np.random.randn(num_samples),
    'Wind_Speed': 15 + 5 * np.random.randn(num_samples),
    'Solar_Irradiance': 500 + 100 * np.random.randn(num_samples)
})

# Add some correlations
data['Price'] = data['Price'] + 0.1 * data['Demand']
data['Solar_Output'] = 0.7 * data['Solar_Irradiance'] + 50 * np.random.randn(num_samples)
data['Wind_Output'] = 0.6 * data['Wind_Speed'] + 30 * np.random.randn(num_samples)

# Plot correlation heatmap
plot_correlation_heatmap(
    data,
    title='Feature Correlation',
    save_path='plots/feature_correlation.png'
)
```

### Data Processing Examples

```python
# Create results directory
from utils.data_processing import create_results_directory
import os

# Create a timestamped results directory
results_dir = create_results_directory('experiment')
print(f"Results will be saved to: {results_dir}")

# Save and load data
import pickle

# Save data
data = {'metrics': {'rewards': [1, 2, 3, 4, 5]}}
with open(os.path.join(results_dir, 'data.pkl'), 'wb') as f:
    pickle.dump(data, f)

# Load data
with open(os.path.join(results_dir, 'data.pkl'), 'rb') as f:
    loaded_data = pickle.load(f)
```

### Hyperparameter Tuning

```python
# Hyperparameter tuning with Optuna
import optuna
from agents.ppo_agent import PPOAgent
from rl_environment.energy_env import EnergyMarketEnv

def objective(trial):
    # Create environment
    env = EnergyMarketEnv(episode_length=24*7, normalize_state=True)
    
    # Define hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    clip_ratio = trial.suggest_float('clip_ratio', 0.1, 0.3)
    
    # Create and train agent
    agent = PPOAgent(
        env=env,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        clip_ratio=clip_ratio
    )
    
    # Train for a short period
    metrics = agent.train(
        env=env,
        total_timesteps=50000,
        buffer_size=2048,
        batch_size=64,
        num_eval_episodes=5
    )
    
    # Return the mean evaluation return
    return np.mean(metrics['eval_returns'])

# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Print best parameters
print("Best parameters:", study.best_params)
print("Best value:", study.best_value)
```

## Results and Analysis

After running experiments, results are saved in the `results/` directory, including:

- Training metrics (rewards, losses, etc.)
- Evaluation returns
- Comparison between agents
- Hyperparameter optimization results 