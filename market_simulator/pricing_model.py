import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

class PricingModel:
    """
    Class for implementing dynamic pricing based on demand-supply balance.
    """
    
    def __init__(self, base_prices=None, volatility=0.1, demand_sensitivity=0.5, 
                 renewable_discount=0.2, carbon_tax=20, base_price=50, 
                 price_cap=500, price_floor=10, scarcity_threshold=0.8):
        """
        Initialize the pricing model.
        
        Args:
            base_prices (dict): Base prices for each energy source ($/MWh)
            volatility (float): Price volatility factor
            demand_sensitivity (float): Sensitivity of prices to demand-supply ratio
            renewable_discount (float): Discount factor for renewable energy
            carbon_tax (float): Carbon tax ($/ton CO2)
            base_price (float): Base price for market-wide pricing
            price_cap (float): Maximum price
            price_floor (float): Minimum price
            scarcity_threshold (float): Threshold for scarcity pricing
        """
        # Set default base prices if not provided
        if base_prices is None:
            self.base_prices = {
                'solar': 40,
                'wind': 45,
                'hydro': 50,
                'nuclear': 60,
                'gas': 70,
                'coal': 65
            }
        else:
            self.base_prices = base_prices
        
        self.volatility = volatility
        self.demand_sensitivity = demand_sensitivity
        self.renewable_discount = renewable_discount
        self.carbon_tax = carbon_tax
        
        # Market-wide pricing parameters
        self.base_price = base_price
        self.price_cap = price_cap
        self.price_floor = price_floor
        self.scarcity_threshold = scarcity_threshold
        
        # Initialize emission factors (tons CO2 / MWh)
        self.emission_factors = {
            'solar': 0.0,
            'wind': 0.0,
            'hydro': 0.0,
            'nuclear': 0.0,
            'gas': 0.4,
            'coal': 0.9
        }
        
        # Initialize time parameters
        self.start_time = datetime(2023, 1, 1, 0, 0, 0)
        self.current_time = self.start_time
        self.time_step = timedelta(hours=1)
        
        # Initialize price history with proper dtypes to avoid FutureWarning
        columns = ['timestamp'] + list(self.base_prices.keys())
        data = {col: pd.Series(dtype='float64' if col != 'timestamp' else 'datetime64[ns]') 
                for col in columns}
        self.price_history = pd.DataFrame(data)
        
        # Initialize current prices
        self.current_prices = self.base_prices.copy()
    
    def calculate_prices(self, demand, allocation, grid_state=None):
        """
        Calculate prices based on demand, supply allocation, and grid state.
        
        Args:
            demand (float): Current energy demand
            allocation (dict): Allocated supply from each source
            grid_state (dict, optional): Current grid state
            
        Returns:
            dict: Calculated prices for each energy source
        """
        # Calculate total supply
        total_supply = sum(allocation.values())
        
        # Calculate demand-supply ratio
        demand_supply_ratio = demand / max(total_supply, 1)
        
        # Calculate market pressure factor
        market_pressure = demand_supply_ratio ** self.demand_sensitivity
        
        # Calculate prices for each source
        prices = {}
        for source, base_price in self.base_prices.items():
            # Start with base price
            price = base_price
            
            # Apply market pressure
            price *= market_pressure
            
            # Apply carbon tax for non-renewable sources
            price += self.emission_factors.get(source, 0) * self.carbon_tax
            
            # Apply renewable discount
            if source in ['solar', 'wind', 'hydro']:
                price *= (1 - self.renewable_discount)
            
            # Apply random volatility
            price *= (1 + np.random.normal(0, self.volatility))
            
            # Ensure price is non-negative
            price = max(0, price)
            
            prices[source] = price
        
        # Apply grid stability factor if grid state is provided
        if grid_state is not None and 'stability' in grid_state:
            stability = grid_state['stability']
            # Increase prices when grid stability is low
            stability_factor = 1 + max(0, (0.8 - stability)) * 0.5
            for source in prices:
                prices[source] *= stability_factor
        
        # Update current prices
        self.current_prices = prices
        
        # Update price history - fixed to avoid FutureWarning
        new_row_data = {'timestamp': [self.current_time]}
        for source, price in prices.items():
            new_row_data[source] = [price]
        new_row = pd.DataFrame(new_row_data)
        
        # Filter out empty or NA entries before concatenation to avoid FutureWarning
        filtered_price_history = self.price_history.dropna(how='all', axis=1)
        filtered_new_row = new_row.dropna(how='all', axis=1)
        self.price_history = pd.concat([filtered_price_history, filtered_new_row], ignore_index=True)
        
        # Update time
        self.current_time += self.time_step
        
        return prices
    
    def calculate_price(self, demand, available_capacity, timestamp=None, 
                        renewable_fraction=0.0, previous_price=None):
        """
        Calculate a single market-wide price.
        
        Args:
            demand (float): Current energy demand
            available_capacity (float): Available capacity
            timestamp (datetime, optional): Current time
            renewable_fraction (float): Fraction of renewable energy
            previous_price (float, optional): Previous price
            
        Returns:
            float: Calculated price
        """
        # Start with base price
        price = self.base_price
        
        # Calculate capacity factor (higher price when capacity is scarce)
        capacity_factor = 1.0
        if available_capacity > 0:
            capacity_ratio = demand / available_capacity
            if capacity_ratio > self.scarcity_threshold:
                # Exponential increase when approaching capacity limit
                scarcity_factor = np.exp(5 * (capacity_ratio - self.scarcity_threshold))
                capacity_factor = 1.0 + scarcity_factor
        
        # Apply capacity factor
        price *= capacity_factor
        
        # Apply renewable discount
        price *= (1.0 - 0.3 * renewable_fraction)
        
        # Apply time-of-day factor if timestamp is provided
        if timestamp is not None:
            hour = timestamp.hour
            # Peak hours (7-9 AM, 5-8 PM)
            if (7 <= hour < 10) or (17 <= hour < 20):
                price *= 1.2
            # Off-peak hours (11 PM - 6 AM)
            elif hour < 6 or hour >= 23:
                price *= 0.8
        
        # Apply random volatility
        price *= (1 + np.random.normal(0, self.volatility))
        
        # Apply price smoothing if previous price is provided
        if previous_price is not None:
            # Limit price change to 20%
            max_change = 0.2 * previous_price
            if abs(price - previous_price) > max_change:
                if price > previous_price:
                    price = previous_price + max_change
                else:
                    price = previous_price - max_change
        
        # Ensure price is within bounds
        price = max(self.price_floor, min(self.price_cap, price))
        
        return price
    
    def simulate_prices(self, demand_data, supply_allocations, grid_states=None, 
                        save_to_file=True, file_path=None):
        """
        Simulate prices over time.
        
        Args:
            demand_data (pd.DataFrame): Demand data with timestamp and demand columns
            supply_allocations (pd.DataFrame): Supply allocations with timestamp and source columns
            grid_states (pd.DataFrame, optional): Grid states with timestamp and state columns
            save_to_file (bool): Whether to save results to a file
            file_path (str): Path to save the results file
            
        Returns:
            pd.DataFrame: Price simulation results
        """
        # Reset price history
        self.price_history = pd.DataFrame(columns=['timestamp'] + list(self.base_prices.keys()))
        self.current_time = self.start_time
        
        # Ensure data is properly aligned
        if len(demand_data) != len(supply_allocations):
            raise ValueError("Demand data and supply allocations must have the same length")
        
        # Simulate prices
        for i in range(len(demand_data)):
            demand = demand_data.iloc[i]['demand']
            
            # Extract allocation for current timestamp
            allocation = {}
            for source in supply_allocations.columns:
                if source != 'timestamp':
                    allocation[source] = supply_allocations.iloc[i][source]
            
            # Extract grid state if provided
            grid_state = None
            if grid_states is not None and i < len(grid_states):
                grid_state = {
                    'stability': grid_states.iloc[i]['stability'],
                    'imbalance': grid_states.iloc[i]['imbalance'],
                    'renewable_fraction': grid_states.iloc[i]['renewable_fraction'],
                    'blackout_risk': grid_states.iloc[i]['blackout_risk']
                }
            
            # Calculate prices
            self.calculate_prices(demand, allocation, grid_state)
        
        # Save results to file if requested
        if save_to_file:
            if file_path is None:
                file_path = os.path.join('data', 'price_simulation.csv')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to CSV
            self.price_history.to_csv(file_path, index=False)
            print(f"Price simulation results saved to {file_path}")
        
        return self.price_history
    
    def plot_prices(self, days=7, save_to_file=True, file_path=None):
        """
        Plot price history for visualization.
        
        Args:
            days (int): Number of days to plot
            save_to_file (bool): Whether to save plot to a file
            file_path (str): Path to save the plot file
        """
        if self.price_history.empty:
            print("No price history available. Simulate prices first.")
            return
        
        # Get data for specified days
        plot_data = self.price_history.iloc[:days*24]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot prices for each source
        for source in self.base_prices:
            plt.plot(plot_data['timestamp'], plot_data[source], label=source.capitalize())
        
        plt.title(f'Energy Prices - {days} Days')
        plt.xlabel('Time')
        plt.ylabel('Price ($/MWh)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save to file if requested
        if save_to_file:
            if file_path is None:
                file_path = os.path.join('plots', 'price_plot.png')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save plot
            plt.savefig(file_path)
            print(f"Price plot saved to {file_path}")
        
        plt.show()
    
    def get_current_prices(self):
        """
        Get the current prices.
        
        Returns:
            dict: Current prices for each energy source
        """
        return self.current_prices
    
    def reset(self):
        """
        Reset the pricing model to initial state.
        """
        self.current_time = self.start_time
        self.current_prices = self.base_prices.copy()
        self.price_history = pd.DataFrame(columns=['timestamp'] + list(self.base_prices.keys()))

def main():
    """
    Main function to demonstrate the pricing model.
    """
    # Create pricing model
    pricing_model = PricingModel()
    
    # Generate sample demand and supply data
    timestamps = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(24*7)]
    
    # Sample demand data
    demand_data = pd.DataFrame({
        'timestamp': timestamps,
        'demand': [1000 + 500 * np.sin(i * np.pi / 12) + np.random.normal(0, 50) for i in range(24*7)]
    })
    
    # Sample supply allocations
    supply_allocations = pd.DataFrame({
        'timestamp': timestamps,
        'solar': [300 * np.sin(max(0, min(i % 24 - 6, 18 - i % 24)) * np.pi / 12) for i in range(24*7)],
        'wind': [200 + 100 * np.random.random() for _ in range(24*7)],
        'hydro': [300 + 50 * np.random.random() for _ in range(24*7)],
        'gas': [400 + 100 * np.random.random() for _ in range(24*7)],
        'coal': [500 + 100 * np.random.random() for _ in range(24*7)]
    })
    
    # Sample grid states
    grid_states = pd.DataFrame({
        'timestamp': timestamps,
        'stability': [0.8 + 0.1 * np.random.random() for _ in range(24*7)],
        'imbalance': [0.1 * np.random.random() for _ in range(24*7)],
        'renewable_fraction': [0.4 + 0.2 * np.random.random() for _ in range(24*7)],
        'blackout_risk': [0.05 * np.random.random() for _ in range(24*7)]
    })
    
    # Simulate prices
    price_history = pricing_model.simulate_prices(demand_data, supply_allocations, grid_states)
    
    # Plot prices
    pricing_model.plot_prices(days=7)
    
    print("Price simulation complete.")

if __name__ == "__main__":
    main() 