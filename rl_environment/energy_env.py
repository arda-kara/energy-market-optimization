import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from market_simulator.demand_generator import DemandGenerator
from market_simulator.supply_simulator import SupplySimulator
from market_simulator.pricing_model import PricingModel
from market_simulator.grid_simulator import GridSimulator

class EnergyMarketEnv(gym.Env):
    """
    Custom OpenAI Gym environment for energy market optimization.
    
    The environment simulates an energy market where an agent must allocate
    electricity from different sources to meet demand while minimizing costs,
    emissions, and maintaining grid stability.
    
    State space:
        - Current demand
        - Forecasted demand for next N time steps
        - Available capacity for each energy source
        - Current price
        - Time features (hour, day of week, etc.)
        - Grid stability
        
    Action space:
        - Allocation percentages for each energy source
        
    Reward:
        Combination of:
        - Cost minimization
        - Emission reduction
        - Grid stability
        - Demand satisfaction
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, demand_data=None, forecast_horizon=6, episode_length=24*7,
                 reward_weights=None, use_forecast=True, normalize_state=True):
        """
        Initialize the environment.
        
        Args:
            demand_data (pd.DataFrame, optional): Pre-generated demand data
            forecast_horizon (int): Number of future time steps to include in state
            episode_length (int): Number of steps per episode
            reward_weights (dict, optional): Weights for different reward components
            use_forecast (bool): Whether to include demand forecast in state
            normalize_state (bool): Whether to normalize state values
        """
        super(EnergyMarketEnv, self).__init__()
        
        # Store configuration
        self.forecast_horizon = forecast_horizon
        self.episode_length = episode_length
        self.use_forecast = use_forecast
        self.normalize_state = normalize_state
        
        # Set default reward weights if not provided
        if reward_weights is None:
            self.reward_weights = {
                'cost': -1.0,
                'emissions': -0.5,
                'stability': 0.3,
                'imbalance': -1.0,
                'renewable': 0.2
            }
        else:
            self.reward_weights = reward_weights
        
        # Create simulator components
        self.demand_generator = DemandGenerator()
        self.supply_simulator = SupplySimulator()
        self.grid_simulator = GridSimulator(
            demand_generator=self.demand_generator,
            supply_simulator=self.supply_simulator
        )
        self.pricing_model = PricingModel()
        
        # If demand data is provided, use it
        if demand_data is not None:
            self.demand_data = demand_data
            self.use_external_demand = True
        else:
            self.demand_data = self.demand_generator.generate_data(days=30, save_to_file=False)
            self.use_external_demand = False
        
        # Set timestamp as index if it's not already
        if 'timestamp' in self.demand_data.columns:
            self.demand_data = self.demand_data.set_index('timestamp')
        
        # Define observation and action spaces
        self._define_spaces()
        
        # Initialize state
        self.current_step = 0
        self.current_episode = 0
        self.state = None
        self.done = False
        
        # For rendering
        self.fig = None
        self.ax = None
        
        # For tracking metrics
        self.episode_metrics = {
            'rewards': [],
            'costs': [],
            'emissions': [],
            'stability': [],
            'imbalance': [],
            'renewable_fraction': []
        }
        
        # Reset environment
        self.reset()
        
        # Add these properties to EnergyMarketEnv class
        self._observation_space = self.observation_space
        self._action_space = self.action_space
        
        # Update the reward range
        self.reward_range = (-float('inf'), float('inf'))
    
    def _define_spaces(self):
        """Define action and observation spaces."""
        # Get the number of energy sources
        available_supply = self.supply_simulator.get_available_supply()
        self.num_sources = len(available_supply)
        self.source_names = list(available_supply.keys())
        
        # Action space: Continuous values between -1 and 1 for each source
        # These will be converted to allocation percentages
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.num_sources,),
            dtype=np.float32
        )
        
        # Observation space components
        obs_components = []
        
        # Current demand
        obs_components.append(1)  # Current demand
        
        # Demand forecast (if enabled)
        if self.use_forecast:
            obs_components.append(self.forecast_horizon)  # Future demand forecast
        
        # Available capacity for each source
        obs_components.append(self.num_sources)  # Available capacity
        
        # Current price
        obs_components.append(1)  # Market price
        
        # Time features (hour of day, day of week)
        obs_components.append(2)  # Hour of day (sin, cos)
        obs_components.append(2)  # Day of week (sin, cos)
        
        # Grid state
        obs_components.append(2)  # Grid stability and imbalance
        
        # Total observation space size
        obs_size = sum(obs_components)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed (int, optional): Random seed
            options (dict, optional): Additional options
            
        Returns:
            tuple: (observation, info)
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Reset simulators
        self.grid_simulator.reset()
        self.supply_simulator.reset()
        self.pricing_model.reset()
        
        # Reset episode state
        self.current_step = 0
        self.done = False
        
        # Choose a random starting point in the demand data
        max_start = len(self.demand_data) - self.episode_length - self.forecast_horizon
        if max_start > 0:
            self.episode_start_idx = np.random.randint(0, max_start)
        else:
            self.episode_start_idx = 0
            
        # Get initial state
        self.state = self._get_state()
        
        # Reset episode metrics
        self.episode_metrics = {
            'rewards': [],
            'costs': [],
            'emissions': [],
            'stability': [],
            'imbalance': [],
            'renewable_fraction': []
        }
        
        # Return initial observation and info
        info = {}
        return self.state, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (np.ndarray): Action to take
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Ensure action is in correct format
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Get current demand
        current_idx = self.episode_start_idx + self.current_step
        if current_idx < len(self.demand_data):
            current_demand = self.demand_data.iloc[current_idx]['demand']
        else:
            # If we've run out of data, end the episode
            self.done = True
            return self.state, 0.0, True, False, {}
        
        # Get available supply
        timestamp = self.demand_data.index[current_idx]
        available_supply = self.supply_simulator.get_available_capacities(timestamp)
        
        # Convert action to allocation
        allocation = {}
        for i, source_name in enumerate(available_supply.keys()):
            # Convert from [-1, 1] to [0, 1] range
            allocation_percent = (action[i] + 1) / 2
            # Apply to available capacity
            allocation[source_name] = allocation_percent * available_supply[source_name]
        
        # Update grid state
        grid_state = self.grid_simulator.update(current_demand, allocation)
        
        # Calculate prices
        prices = self.pricing_model.calculate_prices(current_demand, allocation, grid_state)
        
        # Calculate costs and emissions
        total_cost = sum(allocation[source] * self.supply_simulator.get_cost_factor(source) 
                         for source in allocation)
        total_emissions = sum(allocation[source] * self.supply_simulator.get_emission_factor(source) 
                             for source in allocation)
        
        # Calculate renewable fraction
        renewable_sources = [source for source in allocation.keys() 
                            if 'solar' in source.lower() or 'wind' in source.lower() or 'hydro' in source.lower()]
        renewable_output = sum(allocation.get(source, 0) for source in renewable_sources)
        total_output = sum(allocation.values())
        renewable_fraction = renewable_output / total_output if total_output > 0 else 0
        
        # Prepare step result for reward calculation
        step_result = {
            'demand': current_demand,
            'allocation': allocation,
            'grid_state': grid_state,
            'prices': prices,
            'total_cost': total_cost,
            'total_emissions': total_emissions,
            'energy_balance': grid_state['imbalance'],
            'grid_stability': grid_state['stability'],
            'renewable_fraction': renewable_fraction
        }
        
        # Calculate reward
        reward = self._calculate_reward(step_result)
        
        # Update state
        self.current_step += 1
        self.state = self._get_state()
        
        # Check if episode is done
        terminated = False
        truncated = False
        
        if self.current_step >= self.episode_length:
            truncated = True
        
        if grid_state['stability'] < 0.1:  # Blackout condition
            terminated = True
        
        # Update episode metrics
        self.episode_metrics['rewards'].append(reward)
        self.episode_metrics['costs'].append(total_cost)
        self.episode_metrics['emissions'].append(total_emissions)
        self.episode_metrics['stability'].append(grid_state['stability'])
        self.episode_metrics['imbalance'].append(grid_state['imbalance'])
        self.episode_metrics['renewable_fraction'].append(renewable_fraction)
        
        # Prepare info dict
        info = {
            'demand': current_demand,
            'total_supply': sum(allocation.values()),
            'grid_stability': grid_state['stability'],
            'imbalance': grid_state['imbalance'],
            'total_cost': total_cost,
            'total_emissions': total_emissions,
            'renewable_fraction': renewable_fraction,
            'step': self.current_step,
            'episode': self.current_episode
        }
        
        return self.state, reward, terminated, truncated, info
    
    def _get_state(self):
        """
        Get the current state observation.
        
        Returns:
            np.ndarray: State observation
        """
        # Get current index in demand data
        current_idx = self.episode_start_idx + self.current_step
        if current_idx >= len(self.demand_data):
            # If we've run out of data, use the last available data point
            current_idx = len(self.demand_data) - 1
            
        # Get current timestamp
        timestamp = self.demand_data.index[current_idx]
        
        # Get current demand
        current_demand = self.demand_data.iloc[current_idx]['demand']
        
        # Get demand forecast if enabled
        forecast = []
        if self.use_forecast:
            for i in range(1, self.forecast_horizon + 1):
                forecast_idx = current_idx + i
                if forecast_idx < len(self.demand_data):
                    forecast.append(self.demand_data.iloc[forecast_idx]['demand'])
                else:
                    # If we've run out of data, repeat the last value
                    forecast.append(forecast[-1] if forecast else current_demand)
        
        # Get available capacity for each source
        available_capacity = list(self.supply_simulator.get_available_capacities(timestamp).values())
        
        # Get current price
        market_price = self.pricing_model.calculate_price(
            demand=current_demand,
            available_capacity=sum(available_capacity),
            timestamp=timestamp,
            renewable_fraction=0.5  # Default value, will be updated in step
        )
        
        # Get time features
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        
        # Convert time to cyclical features
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # Get grid state
        grid_state = self.grid_simulator.get_grid_state()
        grid_stability = grid_state['stability']
        grid_imbalance = grid_state['imbalance']
        
        # Combine all features into a flat array
        state = [current_demand]
        
        if self.use_forecast:
            state.extend(forecast)
            
        state.extend(available_capacity)
        state.append(market_price)
        state.extend([hour_sin, hour_cos, day_sin, day_cos])
        state.extend([grid_stability, grid_imbalance])
        
        # Convert to numpy array
        state = np.array(state, dtype=np.float32)
        
        # Normalize state if enabled
        if self.normalize_state:
            state = self._normalize_state(state)
            
        return state
    
    def _normalize_state(self, state):
        """
        Normalize state values to improve learning.
        
        Args:
            state (np.ndarray): Raw state values
            
        Returns:
            np.ndarray: Normalized state values
        """
        # Simple normalization based on typical ranges
        # This could be improved with running statistics during training
        
        # Create a copy to avoid modifying the original
        normalized = state.copy()
        
        # Current position in the normalized array
        pos = 0
        
        # Normalize demand (current and forecast) - typically in range 0-10000
        demand_scale = 5000.0
        
        # Current demand
        normalized[pos] = state[pos] / demand_scale
        pos += 1
        
        # Demand forecast
        if self.use_forecast:
            for i in range(self.forecast_horizon):
                normalized[pos + i] = state[pos + i] / demand_scale
            pos += self.forecast_horizon
        
        # Available capacity - typically in range 0-2000 for each source
        capacity_scale = 1000.0
        for i in range(self.num_sources):
            normalized[pos + i] = state[pos + i] / capacity_scale
        pos += self.num_sources
        
        # Market price - typically in range 0-500
        price_scale = 100.0
        normalized[pos] = state[pos] / price_scale
        pos += 1
        
        # Time features (sin/cos) - already normalized
        # Hour and day of week (sin, cos)
        pos += 4
        
        # Grid stability and imbalance
        # Stability is already in [0, 1]
        # Imbalance needs normalization - typically in range -1000 to 1000
        imbalance_scale = 500.0
        normalized[pos + 1] = state[pos + 1] / imbalance_scale
        
        return normalized
    
    def _calculate_reward(self, step_result):
        """
        Calculate reward based on step results.
        
        Args:
            step_result (dict): Results from the current step
            
        Returns:
            float: Reward value
        """
        # Extract relevant metrics
        demand = step_result['demand']
        total_cost = step_result['total_cost']
        total_emissions = step_result['total_emissions']
        energy_balance = step_result['energy_balance']
        grid_stability = step_result['grid_stability']
        renewable_fraction = step_result['renewable_fraction']
        
        # Normalize metrics for reward calculation
        # This ensures different components have similar scales
        
        # Cost: Lower is better
        # Normalize based on a reasonable cost per MWh
        avg_cost_per_mwh = 50.0  # Average cost in currency units per MWh
        max_cost = demand * avg_cost_per_mwh * 2  # Maximum expected cost (2x average)
        normalized_cost = total_cost / max_cost if max_cost > 0 else 0
        
        # Emissions: Lower is better
        # Normalize based on a reasonable emission per MWh
        avg_emissions_per_mwh = 0.5  # Average emissions in tons CO2 per MWh
        max_emissions = demand * avg_emissions_per_mwh * 2  # Maximum expected emissions (2x average)
        normalized_emissions = total_emissions / max_emissions if max_emissions > 0 else 0
        
        # Energy balance: Closer to zero is better
        # Normalize based on demand
        normalized_balance = min(abs(energy_balance) / (demand * 0.2), 1.0) if demand > 0 else 0
        
        # Grid stability: Higher is better (already normalized 0-1)
        
        # Renewable fraction: Higher is better (already normalized 0-1)
        
        # Calculate reward components
        cost_reward = self.reward_weights['cost'] * normalized_cost
        emissions_reward = self.reward_weights['emissions'] * normalized_emissions
        stability_reward = self.reward_weights['stability'] * grid_stability
        balance_reward = self.reward_weights['imbalance'] * normalized_balance
        renewable_reward = self.reward_weights['renewable'] * renewable_fraction
        
        # Combine components
        reward = cost_reward + emissions_reward + stability_reward + balance_reward + renewable_reward
        
        return reward
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): Rendering mode ('human' or 'rgb_array')
            
        Returns:
            numpy.ndarray or None: RGB array if mode is 'rgb_array', None otherwise
        """
        if mode not in self.metadata['render_modes']:
            raise ValueError(f"Unsupported render mode: {mode}")
        
        # If no history, nothing to render
        if len(self.episode_metrics['rewards']) == 0:
            return None
        
        # Create figure if it doesn't exist
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(3, 2, figsize=(15, 10))
            plt.tight_layout(pad=3.0)
        
        # Clear axes
        for row in self.ax:
            for a in row:
                a.clear()
        
        # Get data for plotting
        steps = list(range(len(self.episode_metrics['rewards'])))
        rewards = self.episode_metrics['rewards']
        costs = self.episode_metrics['costs']
        emissions = self.episode_metrics['emissions']
        stability = self.episode_metrics['stability']
        imbalance = self.episode_metrics['imbalance']
        renewable = self.episode_metrics['renewable_fraction']
        
        # Plot rewards
        self.ax[0, 0].plot(steps, rewards, 'b-')
        self.ax[0, 0].set_title('Rewards')
        self.ax[0, 0].set_xlabel('Step')
        self.ax[0, 0].set_ylabel('Reward')
        
        # Plot costs
        self.ax[0, 1].plot(steps, costs, 'r-')
        self.ax[0, 1].set_title('Costs')
        self.ax[0, 1].set_xlabel('Step')
        self.ax[0, 1].set_ylabel('Cost')
        
        # Plot emissions
        self.ax[1, 0].plot(steps, emissions, 'g-')
        self.ax[1, 0].set_title('Emissions')
        self.ax[1, 0].set_xlabel('Step')
        self.ax[1, 0].set_ylabel('Emissions (tons CO2)')
        
        # Plot grid stability
        self.ax[1, 1].plot(steps, stability, 'c-')
        self.ax[1, 1].set_title('Grid Stability')
        self.ax[1, 1].set_xlabel('Step')
        self.ax[1, 1].set_ylabel('Stability (0-1)')
        self.ax[1, 1].set_ylim(0, 1)
        
        # Plot energy imbalance
        self.ax[2, 0].plot(steps, imbalance, 'm-')
        self.ax[2, 0].set_title('Energy Imbalance')
        self.ax[2, 0].set_xlabel('Step')
        self.ax[2, 0].set_ylabel('Imbalance')
        
        # Plot renewable fraction
        self.ax[2, 1].plot(steps, renewable, 'y-')
        self.ax[2, 1].set_title('Renewable Fraction')
        self.ax[2, 1].set_xlabel('Step')
        self.ax[2, 1].set_ylabel('Fraction (0-1)')
        self.ax[2, 1].set_ylim(0, 1)
        
        # Update layout
        plt.tight_layout()
        
        # Display or return
        if mode == 'human':
            plt.pause(0.1)
            return None
        elif mode == 'rgb_array':
            # Convert plot to RGB array
            self.fig.canvas.draw()
            img = np.array(self.fig.canvas.renderer.buffer_rgba())
            return img
    
    def close(self):
        """
        Close the environment and clean up resources.
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def main():
    """
    Test the environment with random actions.
    """
    import time
    
    # Create environment
    env = EnergyMarketEnv(
        episode_length=24*3,  # 3 days
        use_forecast=True,
        normalize_state=True
    )
    
    # Run a few episodes with random actions
    for episode in range(3):
        print(f"\nEpisode {episode+1}")
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update total reward
            total_reward += reward
            
            # Print info every few steps
            if step % 6 == 0:
                print(f"Step {step}: Reward = {reward:.2f}, "
                      f"Stability = {info['grid_stability']:.2f}, "
                      f"Renewable = {info['renewable_fraction']*100:.1f}%")
            
            # Render
            env.render()
            time.sleep(0.1)
            
            step += 1
        
        print(f"Episode {episode+1} finished with total reward: {total_reward:.2f}")
    
    # Close environment
    env.close()
    
    return env

if __name__ == "__main__":
    main()

def make_env(env_kwargs=None):
    """
    Create a vectorized environment for stable-baselines3.
    
    Args:
        env_kwargs (dict, optional): Keyword arguments for the environment
        
    Returns:
        function: Function that creates an environment instance
    """
    def _init():
        env_kwargs = env_kwargs or {}
        env = EnergyMarketEnv(**env_kwargs)
        env = Monitor(env)
        return env
    
    return _init 