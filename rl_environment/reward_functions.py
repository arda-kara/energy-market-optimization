import numpy as np

class RewardFunction:
    """Base class for reward functions."""
    
    def __init__(self, **kwargs):
        """Initialize reward function with optional parameters."""
        self.params = kwargs
    
    def calculate(self, step_result):
        """
        Calculate reward based on step results.
        
        Args:
            step_result (dict): Results from the current step
            
        Returns:
            float: Reward value
        """
        raise NotImplementedError("Subclasses must implement calculate method")


class CostMinimizationReward(RewardFunction):
    """Reward function focused on minimizing cost."""
    
    def __init__(self, cost_weight=-1.0, balance_penalty=-0.5):
        """
        Initialize cost minimization reward function.
        
        Args:
            cost_weight (float): Weight for cost component (negative to minimize)
            balance_penalty (float): Weight for energy imbalance penalty (negative)
        """
        super().__init__(cost_weight=cost_weight, balance_penalty=balance_penalty)
    
    def calculate(self, step_result):
        """
        Calculate reward based on cost minimization.
        
        Args:
            step_result (dict): Results from the current step
            
        Returns:
            float: Reward value
        """
        # Extract relevant metrics
        demand = step_result['demand']
        total_cost = step_result['total_cost']
        energy_balance = step_result['energy_balance']
        
        # Normalize metrics
        normalized_cost = total_cost / (demand + 1e-6)  # Cost per MW
        normalized_balance = abs(energy_balance) / (demand + 1e-6)  # Absolute imbalance as fraction of demand
        
        # Calculate reward components
        cost_reward = self.params['cost_weight'] * normalized_cost
        balance_penalty = self.params['balance_penalty'] * normalized_balance
        
        # Combine components
        reward = cost_reward + balance_penalty
        
        return reward


class EmissionMinimizationReward(RewardFunction):
    """Reward function focused on minimizing emissions."""
    
    def __init__(self, emission_weight=-1.0, balance_penalty=-0.5):
        """
        Initialize emission minimization reward function.
        
        Args:
            emission_weight (float): Weight for emission component (negative to minimize)
            balance_penalty (float): Weight for energy imbalance penalty (negative)
        """
        super().__init__(emission_weight=emission_weight, balance_penalty=balance_penalty)
    
    def calculate(self, step_result):
        """
        Calculate reward based on emission minimization.
        
        Args:
            step_result (dict): Results from the current step
            
        Returns:
            float: Reward value
        """
        # Extract relevant metrics
        demand = step_result['demand']
        total_emissions = step_result['total_emissions']
        energy_balance = step_result['energy_balance']
        
        # Normalize metrics
        normalized_emissions = total_emissions / (demand + 1e-6)  # Emissions per MW
        normalized_balance = abs(energy_balance) / (demand + 1e-6)  # Absolute imbalance as fraction of demand
        
        # Calculate reward components
        emission_reward = self.params['emission_weight'] * normalized_emissions
        balance_penalty = self.params['balance_penalty'] * normalized_balance
        
        # Combine components
        reward = emission_reward + balance_penalty
        
        return reward


class GridStabilityReward(RewardFunction):
    """Reward function focused on maintaining grid stability."""
    
    def __init__(self, stability_weight=1.0, balance_penalty=-1.0):
        """
        Initialize grid stability reward function.
        
        Args:
            stability_weight (float): Weight for stability component (positive)
            balance_penalty (float): Weight for energy imbalance penalty (negative)
        """
        super().__init__(stability_weight=stability_weight, balance_penalty=balance_penalty)
    
    def calculate(self, step_result):
        """
        Calculate reward based on grid stability.
        
        Args:
            step_result (dict): Results from the current step
            
        Returns:
            float: Reward value
        """
        # Extract relevant metrics
        demand = step_result['demand']
        grid_stability = step_result['grid_stability']
        energy_balance = step_result['energy_balance']
        
        # Normalize metrics
        normalized_balance = abs(energy_balance) / (demand + 1e-6)  # Absolute imbalance as fraction of demand
        
        # Calculate reward components
        stability_reward = self.params['stability_weight'] * grid_stability
        balance_penalty = self.params['balance_penalty'] * normalized_balance
        
        # Combine components
        reward = stability_reward + balance_penalty
        
        # Add large penalty for blackouts (very low stability)
        if grid_stability < 0.2:
            reward -= 10.0
        
        return reward


class RenewableMaximizationReward(RewardFunction):
    """Reward function focused on maximizing renewable energy usage."""
    
    def __init__(self, renewable_weight=1.0, balance_penalty=-0.5):
        """
        Initialize renewable maximization reward function.
        
        Args:
            renewable_weight (float): Weight for renewable component (positive)
            balance_penalty (float): Weight for energy imbalance penalty (negative)
        """
        super().__init__(renewable_weight=renewable_weight, balance_penalty=balance_penalty)
    
    def calculate(self, step_result):
        """
        Calculate reward based on renewable energy usage.
        
        Args:
            step_result (dict): Results from the current step
            
        Returns:
            float: Reward value
        """
        # Extract relevant metrics
        demand = step_result['demand']
        renewable_fraction = step_result['renewable_fraction']
        energy_balance = step_result['energy_balance']
        
        # Normalize metrics
        normalized_balance = abs(energy_balance) / (demand + 1e-6)  # Absolute imbalance as fraction of demand
        
        # Calculate reward components
        renewable_reward = self.params['renewable_weight'] * renewable_fraction
        balance_penalty = self.params['balance_penalty'] * normalized_balance
        
        # Combine components
        reward = renewable_reward + balance_penalty
        
        return reward


class CompositeReward(RewardFunction):
    """Composite reward function combining multiple objectives."""
    
    def __init__(self, cost_weight=-0.4, emission_weight=-0.3, 
                 stability_weight=0.2, renewable_weight=0.1, balance_weight=-1.0):
        """
        Initialize composite reward function.
        
        Args:
            cost_weight (float): Weight for cost component
            emission_weight (float): Weight for emission component
            stability_weight (float): Weight for stability component
            renewable_weight (float): Weight for renewable component
            balance_weight (float): Weight for energy balance component
        """
        super().__init__(
            cost_weight=cost_weight,
            emission_weight=emission_weight,
            stability_weight=stability_weight,
            renewable_weight=renewable_weight,
            balance_weight=balance_weight
        )
    
    def calculate(self, step_result):
        """
        Calculate reward based on multiple objectives.
        
        Args:
            step_result (dict): Results from the current step
            
        Returns:
            float: Reward value
        """
        # Extract relevant metrics
        demand = step_result['demand']
        total_cost = step_result['total_cost']
        total_emissions = step_result['total_emissions']
        grid_stability = step_result['grid_stability']
        renewable_fraction = step_result['renewable_fraction']
        energy_balance = step_result['energy_balance']
        
        # Normalize metrics
        normalized_cost = total_cost / (demand + 1e-6)  # Cost per MW
        normalized_emissions = total_emissions / (demand + 1e-6)  # Emissions per MW
        normalized_balance = abs(energy_balance) / (demand + 1e-6)  # Absolute imbalance as fraction of demand
        
        # Calculate reward components
        cost_reward = self.params['cost_weight'] * normalized_cost
        emission_reward = self.params['emission_weight'] * normalized_emissions
        stability_reward = self.params['stability_weight'] * grid_stability
        renewable_reward = self.params['renewable_weight'] * renewable_fraction
        balance_reward = self.params['balance_weight'] * normalized_balance
        
        # Combine components
        reward = (
            cost_reward +
            emission_reward +
            stability_reward +
            renewable_reward +
            balance_reward
        )
        
        # Add large penalty for blackouts (very low stability)
        if grid_stability < 0.2:
            reward -= 10.0
        
        return reward


class AdaptiveReward(RewardFunction):
    """
    Adaptive reward function that changes weights based on current conditions.
    
    For example, it might prioritize stability during high demand periods,
    emissions during low demand periods, and cost during price spikes.
    """
    
    def __init__(self, base_weights=None):
        """
        Initialize adaptive reward function.
        
        Args:
            base_weights (dict, optional): Base weights for different components
        """
        if base_weights is None:
            base_weights = {
                'cost': -0.4,
                'emission': -0.3,
                'stability': 0.2,
                'renewable': 0.1,
                'balance': -1.0
            }
        
        super().__init__(**base_weights)
    
    def calculate(self, step_result):
        """
        Calculate reward with adaptive weights based on current conditions.
        
        Args:
            step_result (dict): Results from the current step
            
        Returns:
            float: Reward value
        """
        # Extract relevant metrics
        demand = step_result['demand']
        total_cost = step_result['total_cost']
        total_emissions = step_result['total_emissions']
        grid_stability = step_result['grid_stability']
        renewable_fraction = step_result['renewable_fraction']
        energy_balance = step_result['energy_balance']
        price = step_result['price']
        
        # Normalize metrics
        normalized_cost = total_cost / (demand + 1e-6)  # Cost per MW
        normalized_emissions = total_emissions / (demand + 1e-6)  # Emissions per MW
        normalized_balance = abs(energy_balance) / (demand + 1e-6)  # Absolute imbalance as fraction of demand
        
        # Get base weights
        weights = self.params.copy()
        
        # Adjust weights based on conditions
        
        # During high demand, prioritize stability
        if demand > 1500:  # Threshold for high demand
            weights['stability'] *= 2.0
            weights['balance'] *= 1.5
        
        # During price spikes, prioritize cost
        if price > 100:  # Threshold for price spike
            weights['cost'] *= 1.5
        
        # When stability is low, prioritize it even more
        if grid_stability < 0.5:
            weights['stability'] *= 2.0
            weights['balance'] *= 2.0
        
        # When renewable availability is high, prioritize renewable usage
        if renewable_fraction > 0.7:
            weights['renewable'] *= 1.5
        
        # Calculate reward components
        cost_reward = weights['cost'] * normalized_cost
        emission_reward = weights['emission'] * normalized_emissions
        stability_reward = weights['stability'] * grid_stability
        renewable_reward = weights['renewable'] * renewable_fraction
        balance_reward = weights['balance'] * normalized_balance
        
        # Combine components
        reward = (
            cost_reward +
            emission_reward +
            stability_reward +
            renewable_reward +
            balance_reward
        )
        
        # Add large penalty for blackouts (very low stability)
        if grid_stability < 0.2:
            reward -= 10.0
        
        return reward


class CurriculumReward(RewardFunction):
    """Reward function that evolves with agent's performance."""
    
    def __init__(self, stages, **kwargs):
        super().__init__(**kwargs)
        self.stages = sorted(stages, key=lambda x: x['threshold'])
        self.current_stage = 0
        self.best_performance = -np.inf
        
    def update_stage(self, episode_return):
        """Update curriculum stage based on agent performance."""
        if episode_return > self.best_performance:
            self.best_performance = episode_return
            for i, stage in enumerate(self.stages):
                if self.best_performance >= stage['threshold']:
                    self.current_stage = i
                    
    def calculate(self, step_result):
        """Calculate reward with current curriculum stage weights."""
        stage_weights = self.stages[self.current_stage]['weights']
        return super()._calculate_reward(step_result, stage_weights)


def get_reward_function(name, **kwargs):
    """
    Factory function to create reward functions by name.
    
    Args:
        name (str): Name of the reward function
        **kwargs: Additional parameters for the reward function
        
    Returns:
        RewardFunction: Instantiated reward function
    """
    reward_functions = {
        'cost': CostMinimizationReward,
        'emission': EmissionMinimizationReward,
        'stability': GridStabilityReward,
        'renewable': RenewableMaximizationReward,
        'composite': CompositeReward,
        'adaptive': AdaptiveReward
    }
    
    if name == 'curriculum':
        return CurriculumReward(
            stages=[
                {'threshold': -1000, 'weights': {'balance': -2.0}},
                {'threshold': -500, 'weights': {'balance': -1.5, 'cost': -0.5}},
                {'threshold': 0, 'weights': {'balance': -1.0, 'cost': -0.5, 'emissions': -0.3}}
            ],
            **kwargs
        )
    
    if name not in reward_functions:
        raise ValueError(f"Unknown reward function: {name}")
    
    return reward_functions[name](**kwargs) 