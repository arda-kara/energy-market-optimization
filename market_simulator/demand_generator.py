import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

class DemandGenerator:
    """
    Generates synthetic energy demand data based on realistic patterns.
    
    Features:
    - Daily patterns (peak hours in morning and evening)
    - Weekly patterns (weekday vs weekend differences)
    - Seasonal patterns (higher demand in winter and summer)
    - Random noise to simulate unpredictability
    - Special events (holidays, extreme weather) causing demand spikes
    """
    
    def __init__(self, base_demand=1000, daily_variation=0.2, weekly_variation=0.15, 
                 seasonal_variation=0.3, noise_level=0.05, daily_pattern=True, 
                 weekly_pattern=True, seasonal_pattern=True, random_noise=True):
        """
        Initialize the demand generator with configurable parameters.
        
        Args:
            base_demand (float): Base demand level in MW
            daily_variation (float): Amplitude of daily variations (0-1)
            weekly_variation (float): Amplitude of weekly variations (0-1)
            seasonal_variation (float): Amplitude of seasonal variations (0-1)
            noise_level (float): Level of random noise (0-1)
            daily_pattern (bool): Whether to include daily patterns
            weekly_pattern (bool): Whether to include weekly patterns
            seasonal_pattern (bool): Whether to include seasonal patterns
            random_noise (bool): Whether to add random noise
        """
        self.base_demand = base_demand
        self.daily_variation = daily_variation
        self.weekly_variation = weekly_variation
        self.seasonal_variation = seasonal_variation
        self.noise_level = noise_level
        
        # Pattern flags
        self.daily_pattern = daily_pattern
        self.weekly_pattern = weekly_pattern
        self.seasonal_pattern = seasonal_pattern
        self.random_noise = random_noise
        
        # Initialize time parameters
        self.start_time = datetime(2023, 1, 1, 0, 0, 0)
        self.current_time = self.start_time
        self.time_step = timedelta(hours=1)
        
        # Initialize demand data
        self.demand_data = pd.DataFrame(columns=['timestamp', 'demand'])
        
        # Current demand value
        self.current_demand = self._calculate_demand(self.current_time)
    
    def _calculate_demand(self, time):
        """
        Calculate demand for a specific time.
        
        Args:
            time (datetime): Time to calculate demand for
            
        Returns:
            float: Calculated demand in MW
        """
        demand = self.base_demand
        
        # Add daily pattern (peak during day, low at night)
        if self.daily_pattern:
            hour = time.hour
            # Morning ramp-up (6-9 AM)
            if 6 <= hour < 9:
                demand *= 1.0 + 0.5 * (hour - 6) / 3
            # Daytime plateau (9 AM - 5 PM)
            elif 9 <= hour < 17:
                demand *= 1.5
            # Evening peak (5-8 PM)
            elif 17 <= hour < 20:
                demand *= 1.8
            # Evening ramp-down (8-11 PM)
            elif 20 <= hour < 23:
                demand *= 1.8 - 0.6 * (hour - 20) / 3
            # Night valley (11 PM - 6 AM)
            else:
                demand *= 0.7
        
        # Add weekly pattern (lower on weekends)
        if self.weekly_pattern:
            weekday = time.weekday()
            # Weekend (Saturday and Sunday)
            if weekday >= 5:
                demand *= 0.8
        
        # Add seasonal pattern
        if self.seasonal_pattern:
            month = time.month
            # Winter (Dec-Feb)
            if month in [12, 1, 2]:
                demand *= 1.3
            # Summer (Jun-Aug)
            elif month in [6, 7, 8]:
                demand *= 1.2
            # Spring/Fall
            else:
                demand *= 1.0
        
        # Add random noise
        if self.random_noise:
            noise = np.random.normal(0, self.noise_level)
            demand *= (1 + noise)
        
        return max(0, demand)  # Ensure non-negative demand
    
    def generate_data(self, days=365, save_to_file=True, file_path=None):
        """
        Generate synthetic demand data for a specified number of days.
        
        Args:
            days (int): Number of days to generate data for
            save_to_file (bool): Whether to save data to a file
            file_path (str): Path to save the data file
            
        Returns:
            pd.DataFrame: Generated demand data
        """
        # Reset time
        self.current_time = self.start_time
        
        # Generate data
        timestamps = []
        demands = []
        
        for _ in range(days * 24):  # Hourly data for specified days
            demand = self._calculate_demand(self.current_time)
            timestamps.append(self.current_time)
            demands.append(demand)
            self.current_time += self.time_step
        
        # Create DataFrame
        self.demand_data = pd.DataFrame({
            'timestamp': timestamps,
            'demand': demands
        })
        
        # Save to file if requested
        if save_to_file:
            if file_path is None:
                file_path = os.path.join('data', 'demand_data.csv')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to CSV
            self.demand_data.to_csv(file_path, index=False)
            print(f"Demand data saved to {file_path}")
        
        return self.demand_data
    
    def plot_demand(self, days=7, save_to_file=True, file_path=None):
        """
        Plot demand data for visualization.
        
        Args:
            days (int): Number of days to plot
            save_to_file (bool): Whether to save plot to a file
            file_path (str): Path to save the plot file
        """
        if self.demand_data.empty:
            print("No demand data available. Generate data first.")
            return
        
        # Get data for specified days
        plot_data = self.demand_data.iloc[:days*24]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(plot_data['timestamp'], plot_data['demand'])
        plt.title(f'Synthetic Energy Demand - {days} Days')
        plt.xlabel('Time')
        plt.ylabel('Demand (MW)')
        plt.grid(True)
        plt.tight_layout()
        
        # Save to file if requested
        if save_to_file:
            if file_path is None:
                file_path = os.path.join('plots', 'demand_plot.png')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save plot
            plt.savefig(file_path)
            print(f"Demand plot saved to {file_path}")
        
        plt.show()
    
    def update(self):
        """
        Update the current time and demand.
        """
        self.current_time += self.time_step
        self.current_demand = self._calculate_demand(self.current_time)
    
    def get_demand(self):
        """
        Get the current demand value.
        
        Returns:
            float: Current demand in MW
        """
        return self.current_demand
    
    def reset(self):
        """
        Reset the demand generator to the start time.
        """
        self.current_time = self.start_time
        self.current_demand = self._calculate_demand(self.current_time)

def main():
    """
    Main function to demonstrate the demand generator.
    """
    # Create demand generator
    demand_gen = DemandGenerator(base_demand=1000)
    
    # Generate data for one year
    demand_data = demand_gen.generate_data(days=365)
    
    # Plot one week of data
    demand_gen.plot_demand(days=7)
    
    print("Demand generation complete.")

if __name__ == "__main__":
    main() 