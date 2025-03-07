import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import tqdm

from market_simulator.demand_generator import DemandGenerator
from market_simulator.supply_simulator import SupplySimulator
from market_simulator.pricing_model import PricingModel

class GridSimulator:
    """
    Simulates the electricity grid by integrating demand, supply, and pricing.
    
    Features:
    - Simulates grid operation over time
    - Tracks energy balance, costs, emissions, and prices
    - Supports different dispatch strategies
    - Handles grid stability constraints
    """
    
    def __init__(self, demand_generator=None, supply_simulator=None, pricing_model=None, base_stability=0.9, stability_threshold=0.5, 
                 imbalance_penalty=0.1, renewable_bonus=0.05):
        """
        Initialize the grid simulator.
        
        Args:
            demand_generator (DemandGenerator, optional): Demand generator instance
            supply_simulator (SupplySimulator, optional): Supply simulator instance
            pricing_model (PricingModel, optional): Pricing model instance
            base_stability (float): Base grid stability (0-1)
            stability_threshold (float): Threshold for grid instability
            imbalance_penalty (float): Penalty factor for supply-demand imbalance
            renewable_bonus (float): Bonus factor for renewable energy usage
        """
        # Create components if not provided
        self.demand_generator = demand_generator or DemandGenerator()
        self.supply_simulator = supply_simulator or SupplySimulator()
        self.pricing_model = pricing_model or PricingModel()
        
        # Store configuration parameters
        self.base_stability = base_stability
        self.stability_threshold = stability_threshold
        self.imbalance_penalty = imbalance_penalty
        self.renewable_bonus = renewable_bonus
        
        # If supply simulator is empty, create default portfolio
        if not self.supply_simulator.sources:
            self.supply_simulator.create_default_portfolio()
        
        # Simulation state
        self.current_time = None
        self.simulation_results = None
        self.grid_stability = 1.0  # 0-1 scale, 1 is perfectly stable
        self.time_index = None
        
        # Initialize grid state
        self.grid_state = {
            'stability': base_stability,
            'imbalance': 0.0,
            'renewable_fraction': 0.0,
            'blackout_risk': 0.0
        }
        
        # Initialize grid history with proper dtypes to avoid FutureWarning
        self.grid_history = pd.DataFrame({
            'timestamp': pd.Series(dtype='datetime64[ns]'),
            'stability': pd.Series(dtype='float64'),
            'imbalance': pd.Series(dtype='float64'), 
            'renewable_fraction': pd.Series(dtype='float64'),
            'blackout_risk': pd.Series(dtype='float64')
        })
        
        # Initialize time parameters
        self.start_time = datetime(2023, 1, 1, 0, 0, 0)
        self.current_time = self.start_time
        self.time_step = timedelta(hours=1)
    
    def initialize_simulation(self, start_time='2023-01-01', periods=8760, freq='h'):
        """
        Initialize a new simulation.
        
        Args:
            start_time (str): Start time in 'YYYY-MM-DD' format
            periods (int): Number of periods to simulate
            freq (str): Frequency of simulation ('h' for hourly)
            
        Returns:
            pd.DatetimeIndex: Simulation time index
        """
        self.current_time = pd.Timestamp(start_time)
        self.time_index = pd.date_range(start=start_time, periods=periods, freq=freq)
        self.simulation_results = None
        self.grid_stability = 1.0
        
        return self.time_index
    
    def run_simulation(self, demand_data=None, dispatch_strategy='merit_order', 
                       include_stability=True, save_results=True, result_path=None,
                       show_progress=True, optimization_objective='balanced'):
        """
        Run a full simulation over the time index.
        
        Args:
            demand_data (pd.DataFrame, optional): Pre-generated demand data
            dispatch_strategy (str): Strategy for dispatching generation
                Options: 'merit_order', 'proportional', 'renewable_first', 'optimized'
            include_stability (bool): Whether to include grid stability constraints
            save_results (bool): Whether to save results to CSV
            result_path (str, optional): Path to save results
            show_progress (bool): Whether to show a progress bar
            optimization_objective (str): Objective for optimized dispatch strategy
                Options: 'cost', 'emissions', 'renewable', 'stability', 'balanced'
            
        Returns:
            pd.DataFrame: Simulation results
        """
        if self.time_index is None:
            self.initialize_simulation()
        
        # Generate demand data if not provided
        if demand_data is None:
            demand_data = self.demand_generator.generate_data(
                days=len(self.time_index) // 24,
                save_to_file=False
            )
        
        results = []
        previous_price = None
        
        # Create iterator with optional progress bar
        time_iterator = tqdm.tqdm(enumerate(self.time_index), total=len(self.time_index), 
                                 desc=f"Running {dispatch_strategy} simulation", 
                                 disable=not show_progress)
        
        # Run simulation for each time step
        for i, timestamp in time_iterator:
            self.current_time = timestamp
            
            # Get demand for this time step
            if isinstance(demand_data, pd.DataFrame) and 'demand' in demand_data.columns:
                if timestamp in demand_data.index:
                    demand = demand_data.loc[timestamp, 'demand']
                else:
                    demand = demand_data.iloc[i % len(demand_data)]['demand']
            else:
                demand = demand_data[i % len(demand_data)]
            
            # Get available capacity
            available_capacities = self.supply_simulator.get_available_capacities(timestamp)
            total_capacity = sum(available_capacities.values())
            
            # Dispatch generation based on strategy
            if dispatch_strategy == 'merit_order':
                dispatch = self._merit_order_dispatch(demand, timestamp)
            elif dispatch_strategy == 'proportional':
                dispatch = self._proportional_dispatch(demand, timestamp)
            elif dispatch_strategy == 'renewable_first':
                dispatch = self._renewable_first_dispatch(demand, timestamp)
            elif dispatch_strategy == 'optimized':
                dispatch = self.optimize_dispatch(demand, timestamp, optimization_objective)
            else:
                raise ValueError(f"Unknown dispatch strategy: {dispatch_strategy}")
            
            # Set outputs and get actual generation
            actual_outputs = self.supply_simulator.set_outputs(dispatch, timestamp)
            total_output = sum(actual_outputs.values())
            
            # Calculate energy balance
            energy_balance = total_output - demand
            
            # Update grid stability if enabled
            if include_stability:
                self._update_grid_stability(energy_balance, demand)
            
            # Calculate costs and emissions
            costs = self.supply_simulator.get_costs(actual_outputs)
            emissions = self.supply_simulator.get_emissions(actual_outputs)
            total_cost = sum(costs.values())
            total_emissions = sum(emissions.values())
            
            # Calculate renewable fraction
            renewable_sources = []
            for source in self.supply_simulator.sources:
                if "Solar" in source.name or "Wind" in source.name or "Hydro" in source.name:
                    renewable_sources.append(source.name)
            
            renewable_output = sum(actual_outputs.get(source, 0) for source in renewable_sources)
            renewable_fraction = renewable_output / total_output if total_output > 0 else 0
            
            # Calculate price
            price = self.pricing_model.calculate_price(
                demand=demand,
                available_capacity=total_capacity,
                timestamp=timestamp,
                renewable_fraction=renewable_fraction,
                previous_price=previous_price
            )
            previous_price = price
            
            # Record results
            results.append({
                'timestamp': timestamp,
                'demand': demand,
                'total_capacity': total_capacity,
                'total_output': total_output,
                'energy_balance': energy_balance,
                'grid_stability': self.grid_stability,
                'renewable_output': renewable_output,
                'renewable_fraction': renewable_fraction,
                'total_cost': total_cost,
                'total_emissions': total_emissions,
                'price': price,
                'outputs': actual_outputs,
                'costs': costs,
                'emissions': emissions
            })
        
        # Convert results to DataFrame
        self.simulation_results = pd.DataFrame(results)
        
        # Save results if requested
        if save_results:
            self._save_results(result_path)
        
        return self.simulation_results
    
    def _merit_order_dispatch(self, demand, timestamp):
        """
        Dispatch energy sources in merit order (cheapest first).
        
        Args:
            demand (float): Demand to meet in MW
            timestamp (pd.Timestamp): Current time
            
        Returns:
            dict: Dictionary mapping source names to outputs
        """
        # Use the supply simulator's merit order dispatch method
        return self.supply_simulator._merit_order_dispatch(demand, timestamp)
    
    def _proportional_dispatch(self, demand, timestamp):
        """
        Dispatch energy sources proportionally to their available capacity.
        
        Args:
            demand (float): Demand to meet in MW
            timestamp (pd.Timestamp): Current time
            
        Returns:
            dict: Dictionary mapping source names to outputs
        """
        available_capacities = self.supply_simulator.get_available_capacities(timestamp)
        total_capacity = sum(available_capacities.values())
        
        dispatch = {}
        for source in self.supply_simulator.sources:
            available = available_capacities[source.name]
            if total_capacity > 0:
                fraction = available / total_capacity
                dispatch[source.name] = demand * fraction
            else:
                dispatch[source.name] = 0
        
        return dispatch
    
    def _renewable_first_dispatch(self, demand, timestamp):
        """
        Dispatch renewable sources first, then conventional sources in merit order.
        
        Args:
            demand (float): Demand to meet in MW
            timestamp (pd.Timestamp): Current time
            
        Returns:
            dict: Dictionary mapping source names to outputs
        """
        renewable_sources = ['Solar PV', 'Wind Farm', 'Hydro Plant']
        conventional_sources = ['Natural Gas', 'Coal']
        
        dispatch = {source.name: 0 for source in self.supply_simulator.sources}
        remaining_demand = demand
        
        # First, dispatch renewable sources
        for name in renewable_sources:
            source = self.supply_simulator.get_source_by_name(name)
            if source:
                available = source.get_available_capacity(timestamp)
                output = min(available, remaining_demand)
                dispatch[name] = output
                remaining_demand -= output
        
        # Then, dispatch conventional sources in merit order
        if remaining_demand > 0:
            # Sort conventional sources by cost
            sorted_sources = sorted(
                [self.supply_simulator.get_source_by_name(name) for name in conventional_sources 
                 if self.supply_simulator.get_source_by_name(name)],
                key=lambda s: s.cost_per_mwh
            )
            
            for source in sorted_sources:
                available = source.get_available_capacity(timestamp)
                
                if available > 0:
                    # For fossil plants, respect minimum output
                    if hasattr(source, 'min_output') and remaining_demand > 0:
                        min_output = source.capacity * source.min_output
                        if 0 < remaining_demand < min_output:
                            # Skip if remaining demand is less than minimum output
                            continue
                    
                    # Dispatch this source
                    output = min(available, remaining_demand)
                    dispatch[source.name] = output
                    remaining_demand -= output
                    
                    # Stop if demand is met
                    if remaining_demand <= 0:
                        break
        
        return dispatch
    
    def _update_grid_stability(self, energy_balance, demand):
        """
        Update grid stability based on energy balance.
        
        Args:
            energy_balance (float): Energy balance (generation - demand) in MW
            demand (float): Current demand in MW
        """
        # Calculate imbalance as percentage of demand
        if demand > 0:
            imbalance_pct = abs(energy_balance) / demand
        else:
            imbalance_pct = abs(energy_balance) / 1.0  # Avoid division by zero
        
        # Update stability (exponential decay based on imbalance)
        stability_decay = np.exp(-5 * imbalance_pct)
        
        # Grid stability has memory (70% previous, 30% current)
        self.grid_stability = 0.7 * self.grid_stability + 0.3 * stability_decay
        
        # Ensure stability is between 0 and 1
        self.grid_stability = max(0, min(1, self.grid_stability))
    
    def _save_results(self, result_path=None):
        """
        Save simulation results to CSV.
        
        Args:
            result_path (str, optional): Path to save results
        """
        if self.simulation_results is None:
            print("No simulation results to save.")
            return
        
        # Create default path if not provided
        if result_path is None:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            result_path = f"data/grid_simulation_{self.time_index[0].strftime('%Y%m%d')}.csv"
        
        # Save main results
        self.simulation_results[[
            'timestamp', 'demand', 'total_capacity', 'total_output', 
            'energy_balance', 'grid_stability', 'renewable_fraction',
            'total_cost', 'total_emissions', 'price'
        ]].to_csv(result_path, index=False)
        
        print(f"Simulation results saved to {result_path}")
    
    def plot_results(self, days=7, start_day=0):
        """
        Plot simulation results.
        
        Args:
            days (int): Number of days to plot
            start_day (int): Starting day index
        """
        if self.simulation_results is None:
            print("No simulation results to plot.")
            return
        
        # Extract subset of results
        start_idx = start_day * 24
        end_idx = start_idx + days * 24
        results = self.simulation_results.iloc[start_idx:end_idx].copy()
        
        # Create figure
        plt.figure(figsize=(15, 15))
        
        # Plot 1: Demand, Generation, and Price
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(results['timestamp'], results['demand'], 'b-', label='Demand (MW)')
        ax1.plot(results['timestamp'], results['total_output'], 'g-', label='Generation (MW)')
        ax1.set_ylabel('Power (MW)')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        ax1b = ax1.twinx()
        ax1b.plot(results['timestamp'], results['price'], 'r-', label='Price ($/MWh)')
        ax1b.set_ylabel('Price ($/MWh)')
        ax1b.legend(loc='upper right')
        
        plt.title('Demand, Generation, and Price')
        
        # Plot 2: Generation Mix
        ax2 = plt.subplot(4, 1, 2)
        
        # Extract generation by source
        sources = list(results['outputs'].iloc[0].keys())
        generation_by_source = {}
        
        for source in sources:
            generation_by_source[source] = [
                outputs[source] for outputs in results['outputs']
            ]
        
        # Stacked area chart
        bottom = np.zeros(len(results))
        for source in sources:
            values = generation_by_source[source]
            ax2.fill_between(results['timestamp'], bottom, bottom + values, label=source)
            bottom += values
        
        ax2.set_ylabel('Generation (MW)')
        ax2.legend(loc='upper right')
        ax2.grid(True)
        
        plt.title('Generation Mix')
        
        # Plot 3: Grid Stability and Energy Balance
        ax3 = plt.subplot(4, 1, 3)
        ax3.plot(results['timestamp'], results['grid_stability'], 'b-', label='Grid Stability')
        ax3.set_ylabel('Stability (0-1)')
        ax3.set_ylim(0, 1.1)
        ax3.legend(loc='upper left')
        ax3.grid(True)
        
        ax3b = ax3.twinx()
        ax3b.plot(results['timestamp'], results['energy_balance'], 'r-', label='Energy Balance (MW)')
        ax3b.set_ylabel('Energy Balance (MW)')
        ax3b.legend(loc='upper right')
        
        plt.title('Grid Stability and Energy Balance')
        
        # Plot 4: Costs and Emissions
        ax4 = plt.subplot(4, 1, 4)
        ax4.plot(results['timestamp'], results['total_cost'], 'g-', label='Total Cost ($)')
        ax4.set_ylabel('Cost ($)')
        ax4.legend(loc='upper left')
        ax4.grid(True)
        
        ax4b = ax4.twinx()
        ax4b.plot(results['timestamp'], results['total_emissions'], 'r-', label='Emissions (tons CO2)')
        ax4b.set_ylabel('Emissions (tons CO2)')
        ax4b.legend(loc='upper right')
        
        plt.title('Costs and Emissions')
        plt.xlabel('Time')
        
        plt.tight_layout()
        plt.show()
    
    def get_summary_statistics(self):
        """
        Calculate summary statistics for the simulation.
        
        Returns:
            dict: Dictionary of summary statistics
        """
        if self.simulation_results is None:
            print("No simulation results available.")
            return {}
        
        # Calculate statistics
        stats = {
            'total_demand': self.simulation_results['demand'].sum(),
            'total_generation': self.simulation_results['total_output'].sum(),
            'total_cost': self.simulation_results['total_cost'].sum(),
            'total_emissions': self.simulation_results['total_emissions'].sum(),
            'average_price': self.simulation_results['price'].mean(),
            'max_price': self.simulation_results['price'].max(),
            'min_price': self.simulation_results['price'].min(),
            'average_renewable_fraction': self.simulation_results['renewable_fraction'].mean(),
            'average_grid_stability': self.simulation_results['grid_stability'].mean(),
            'energy_balance_mean': self.simulation_results['energy_balance'].mean(),
            'energy_balance_std': self.simulation_results['energy_balance'].std(),
            'blackout_hours': (self.simulation_results['grid_stability'] < 0.2).sum(),
            'low_stability_hours': ((self.simulation_results['grid_stability'] >= 0.2) & 
                                   (self.simulation_results['grid_stability'] < 0.5)).sum(),
        }
        
        # Calculate generation by source
        if 'outputs' in self.simulation_results.columns:
            # Get the list of sources from the first row
            first_outputs = self.simulation_results['outputs'].iloc[0]
            sources = list(first_outputs.keys())
            
            for source in sources:
                try:
                    generation = sum(outputs.get(source, 0) for outputs in self.simulation_results['outputs'])
                    stats[f'{source}_generation'] = generation
                    stats[f'{source}_percentage'] = generation / stats['total_generation'] * 100
                except KeyError:
                    # Skip if source not found
                    continue
        
        return stats
    
    def update(self, demand, allocation):
        """
        Update grid state based on demand and supply allocation.
        
        Args:
            demand (float): Current energy demand
            allocation (dict): Allocated supply from each source
            
        Returns:
            dict: Updated grid state
        """
        # Calculate total supply
        total_supply = sum(allocation.values())
        
        # Calculate imbalance
        imbalance = abs(total_supply - demand) / max(demand, 1)
        
        # Calculate renewable fraction
        renewable_sources = ['solar', 'wind', 'hydro']
        renewable_supply = sum(allocation.get(source, 0) for source in renewable_sources)
        renewable_fraction = renewable_supply / total_supply if total_supply > 0 else 0
        
        # Calculate stability
        stability = self.grid_state['stability']
        
        # Reduce stability based on imbalance
        stability -= imbalance * self.imbalance_penalty
        
        # Increase stability based on renewable usage
        stability += renewable_fraction * self.renewable_bonus
        
        # Ensure stability is within bounds
        stability = max(0, min(1, stability))
        
        # Calculate blackout risk
        blackout_risk = 0
        if stability < self.stability_threshold:
            # Exponentially increasing risk as stability decreases
            blackout_risk = ((self.stability_threshold - stability) / self.stability_threshold) ** 2
        
        # Update grid state
        self.grid_state = {
            'stability': stability,
            'imbalance': imbalance,
            'renewable_fraction': renewable_fraction,
            'blackout_risk': blackout_risk
        }
        
        # Update grid history - fixed to avoid FutureWarning
        new_row = pd.DataFrame({
            'timestamp': [self.current_time],
            'stability': [stability],
            'imbalance': [imbalance],
            'renewable_fraction': [renewable_fraction],
            'blackout_risk': [blackout_risk]
        })
        
        # Filter out empty or NA entries before concatenation to avoid FutureWarning
        filtered_grid_history = self.grid_history.dropna(how='all', axis=1)
        filtered_new_row = new_row.dropna(how='all', axis=1)
        self.grid_history = pd.concat([filtered_grid_history, filtered_new_row], ignore_index=True)
        
        # Update time
        self.current_time += self.time_step
        
        return self.grid_state
    
    def simulate_grid(self, demand_data, supply_allocations, save_to_file=True, file_path=None):
        """
        Simulate grid operation over time.
        
        Args:
            demand_data (pd.DataFrame): Demand data with timestamp and demand columns
            supply_allocations (pd.DataFrame): Supply allocations with timestamp and source columns
            save_to_file (bool): Whether to save results to a file
            file_path (str): Path to save the results file
            
        Returns:
            pd.DataFrame: Grid simulation results
        """
        # Reset grid state and history
        self.reset()
        
        # Ensure data is properly aligned
        if len(demand_data) != len(supply_allocations):
            raise ValueError("Demand data and supply allocations must have the same length")
        
        # Simulate grid operation
        for i in range(len(demand_data)):
            demand = demand_data.iloc[i]['demand']
            
            # Extract allocation for current timestamp
            allocation = {}
            for source in supply_allocations.columns:
                if source != 'timestamp':
                    allocation[source] = supply_allocations.iloc[i][source]
            
            # Update grid state
            self.update(demand, allocation)
        
        # Save results to file if requested
        if save_to_file:
            if file_path is None:
                file_path = os.path.join('data', 'grid_simulation.csv')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to CSV
            self.grid_history.to_csv(file_path, index=False)
            print(f"Grid simulation results saved to {file_path}")
        
        return self.grid_history
    
    def plot_grid_metrics(self, days=7, save_to_file=True, file_path=None):
        """
        Plot grid metrics for visualization.
        
        Args:
            days (int): Number of days to plot
            save_to_file (bool): Whether to save plot to a file
            file_path (str): Path to save the plot file
        """
        if self.grid_history.empty:
            print("No grid history available. Simulate grid operation first.")
            return
        
        # Get data for specified days
        plot_data = self.grid_history.iloc[:days*24]
        
        # Create plot
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot stability
        axs[0].plot(plot_data['timestamp'], plot_data['stability'], 'b-')
        axs[0].axhline(y=self.stability_threshold, color='r', linestyle='--', label='Stability Threshold')
        axs[0].set_ylabel('Grid Stability')
        axs[0].set_ylim(0, 1)
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot imbalance
        axs[1].plot(plot_data['timestamp'], plot_data['imbalance'], 'g-')
        axs[1].set_ylabel('Supply-Demand Imbalance')
        axs[1].grid(True)
        
        # Plot renewable fraction and blackout risk
        ax2 = axs[2].twinx()
        axs[2].plot(plot_data['timestamp'], plot_data['renewable_fraction'], 'g-', label='Renewable Fraction')
        ax2.plot(plot_data['timestamp'], plot_data['blackout_risk'], 'r-', label='Blackout Risk')
        axs[2].set_ylabel('Renewable Fraction')
        ax2.set_ylabel('Blackout Risk')
        axs[2].set_ylim(0, 1)
        ax2.set_ylim(0, 1)
        
        # Add legend
        lines1, labels1 = axs[2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        axs[2].grid(True)
        axs[2].set_xlabel('Time')
        
        plt.suptitle(f'Grid Metrics - {days} Days')
        plt.tight_layout()
        
        # Save to file if requested
        if save_to_file:
            if file_path is None:
                file_path = os.path.join('plots', 'grid_metrics.png')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save plot
            plt.savefig(file_path)
            print(f"Grid metrics plot saved to {file_path}")
        
        plt.show()
    
    def get_grid_state(self):
        """
        Get the current grid state.
        
        Returns:
            dict: Current grid state
        """
        return self.grid_state
    
    def optimize_dispatch(self, demand, timestamp, objective='balanced'):
        """
        Optimize dispatch based on the specified objective.
        
        Args:
            demand (float): Current demand in MW
            timestamp (datetime): Current time
            objective (str): Optimization objective ('cost', 'emissions', 'renewable', 'balanced')
            
        Returns:
            dict: Optimized dispatch
        """
        # Get different dispatch strategies
        merit_order_dispatch = self._merit_order_dispatch(demand, timestamp)
        renewable_first_dispatch = self._renewable_first_dispatch(demand, timestamp)
        proportional_dispatch = self._proportional_dispatch(demand, timestamp)
        
        # Calculate metrics for each strategy
        strategies = {
            'merit_order': merit_order_dispatch,
            'renewable_first': renewable_first_dispatch,
            'proportional': proportional_dispatch
        }
        
        metrics = {}
        
        # Get renewable source names
        renewable_sources = []
        for source in self.supply_simulator.sources:
            if "Solar" in source.name or "Wind" in source.name or "Hydro" in source.name:
                renewable_sources.append(source.name)
        
        for name, dispatch in strategies.items():
            # Set outputs to get actual generation
            actual_outputs = self.supply_simulator.set_outputs(dispatch, timestamp)
            
            # Calculate costs and emissions
            costs = self.supply_simulator.get_costs(actual_outputs)
            emissions = self.supply_simulator.get_emissions(actual_outputs)
            total_cost = sum(costs.values())
            total_emissions = sum(emissions.values())
            
            # Calculate renewable fraction
            total_output = sum(actual_outputs.values())
            renewable_output = sum(actual_outputs.get(source, 0) for source in renewable_sources)
            renewable_fraction = renewable_output / total_output if total_output > 0 else 0
            
            # Store metrics
            metrics[name] = {
                'cost': total_cost,
                'emissions': total_emissions,
                'renewable_fraction': renewable_fraction
            }
        
        # Choose strategy based on objective
        if objective == 'cost':
            # Choose strategy with lowest cost
            best_strategy = min(metrics.items(), key=lambda x: x[1]['cost'])[0]
        elif objective == 'emissions':
            # Choose strategy with lowest emissions
            best_strategy = min(metrics.items(), key=lambda x: x[1]['emissions'])[0]
        elif objective == 'renewable':
            # Choose strategy with highest renewable fraction
            best_strategy = max(metrics.items(), key=lambda x: x[1]['renewable_fraction'])[0]
        elif objective == 'balanced':
            # Normalize metrics and choose strategy with best overall score
            min_cost = min(m['cost'] for m in metrics.values())
            max_cost = max(m['cost'] for m in metrics.values())
            min_emissions = min(m['emissions'] for m in metrics.values())
            max_emissions = max(m['emissions'] for m in metrics.values())
            
            scores = {}
            for name, metric in metrics.items():
                # Normalize metrics (0 is best, 1 is worst)
                if max_cost > min_cost:
                    cost_score = (metric['cost'] - min_cost) / (max_cost - min_cost)
                else:
                    cost_score = 0
                    
                if max_emissions > min_emissions:
                    emissions_score = (metric['emissions'] - min_emissions) / (max_emissions - min_emissions)
                else:
                    emissions_score = 0
                    
                renewable_score = 1 - metric['renewable_fraction']  # Invert so 0 is best
                
                # Calculate weighted score (lower is better)
                scores[name] = 0.4 * cost_score + 0.3 * emissions_score + 0.3 * renewable_score
            
            # Choose strategy with lowest score
            best_strategy = min(scores.items(), key=lambda x: x[1])[0]
        else:
            # Default to merit order
            best_strategy = 'merit_order'
        
        return strategies[best_strategy]
    
    def reset(self):
        """
        Reset the grid simulator to initial state.
        """
        self.grid_state = {
            'stability': self.base_stability,
            'imbalance': 0.0,
            'renewable_fraction': 0.0,
            'blackout_risk': 0.0
        }
        self.current_time = self.start_time
        self.grid_stability = 1.0
        
        # Reset grid history with proper dtypes
        self.grid_history = pd.DataFrame({
            'timestamp': pd.Series(dtype='datetime64[ns]'),
            'stability': pd.Series(dtype='float64'),
            'imbalance': pd.Series(dtype='float64'), 
            'renewable_fraction': pd.Series(dtype='float64'),
            'blackout_risk': pd.Series(dtype='float64')
        })

def main():
    """Create and run a grid simulation."""
    # Create components
    demand_generator = DemandGenerator(
        base_demand=1000,
        daily_variation=0.3,
        weekly_variation=0.15,
        seasonal_variation=0.4,
        noise_level=0.05
    )
    
    supply_simulator = SupplySimulator()
    supply_simulator.create_default_portfolio()
    
    pricing_model = PricingModel(
        base_price=50,
        price_cap=500,
        price_floor=10,
        scarcity_threshold=0.8,
        volatility=0.1
    )
    
    # Create grid simulator
    grid_simulator = GridSimulator(
        demand_generator=demand_generator,
        supply_simulator=supply_simulator,
        pricing_model=pricing_model,
        base_stability=0.9,
        stability_threshold=0.5,
        imbalance_penalty=0.1,
        renewable_bonus=0.05
    )
    
    # Initialize simulation
    grid_simulator.initialize_simulation(
        start_time='2023-01-01',
        periods=24*7,  # One week
        freq='h'
    )
    
    # Run simulation
    results = grid_simulator.run_simulation(
        dispatch_strategy='merit_order',
        include_stability=True,
        save_results=True
    )
    
    # Print summary statistics
    stats = grid_simulator.get_summary_statistics()
    print("\nSimulation Summary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Plot results
    grid_simulator.plot_results(days=7)

if __name__ == "__main__":
    main() 