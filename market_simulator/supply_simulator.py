import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

class EnergySource:
    """Base class for all energy sources."""
    
    def __init__(self, name, capacity, cost_per_mwh, emission_factor, ramp_rate=None):
        """
        Initialize an energy source.
        
        Args:
            name (str): Name of the energy source
            capacity (float): Maximum capacity in MW
            cost_per_mwh (float): Cost per MWh in currency units
            emission_factor (float): CO2 emissions in tons per MWh
            ramp_rate (float, optional): Maximum change in output per hour (% of capacity)
        """
        self.name = name
        self.capacity = capacity
        self.cost_per_mwh = cost_per_mwh
        self.emission_factor = emission_factor
        self.ramp_rate = ramp_rate if ramp_rate is not None else 1.0  # Default: can ramp fully in 1 hour
        self.current_output = 0
    
    def get_available_capacity(self, timestamp):
        """
        Get available capacity at the given timestamp.
        
        Args:
            timestamp (pd.Timestamp): Current time
            
        Returns:
            float: Available capacity in MW
        """
        return self.capacity
    
    def set_output(self, output, timestamp=None):
        """
        Set the output level, respecting ramp rate constraints.
        
        Args:
            output (float): Desired output in MW
            timestamp (pd.Timestamp, optional): Current time
            
        Returns:
            float: Actual output after applying constraints
        """
        # Ensure output is within capacity
        output = max(0, min(output, self.get_available_capacity(timestamp)))
        
        # Apply ramp rate constraint
        max_change = self.capacity * self.ramp_rate
        if abs(output - self.current_output) > max_change:
            if output > self.current_output:
                output = self.current_output + max_change
            else:
                output = self.current_output - max_change
        
        self.current_output = output
        return output
    
    def get_cost(self, output):
        """
        Calculate cost for the given output.
        
        Args:
            output (float): Output in MW
            
        Returns:
            float: Cost in currency units
        """
        return output * self.cost_per_mwh
    
    def get_emissions(self, output):
        """
        Calculate emissions for the given output.
        
        Args:
            output (float): Output in MW
            
        Returns:
            float: Emissions in tons of CO2
        """
        return output * self.emission_factor


class SolarPV(EnergySource):
    """Solar photovoltaic energy source with time-dependent availability."""
    
    def __init__(self, capacity, cost_per_mwh=0, emission_factor=0, 
                 efficiency=0.15, latitude=40.0):
        """
        Initialize a solar PV energy source.
        
        Args:
            capacity (float): Maximum capacity in MW
            cost_per_mwh (float): Cost per MWh (typically very low for solar)
            emission_factor (float): CO2 emissions (typically zero for solar)
            efficiency (float): Solar panel efficiency
            latitude (float): Latitude of the installation (affects solar irradiance)
        """
        super().__init__("Solar PV", capacity, cost_per_mwh, emission_factor, ramp_rate=1.0)
        self.efficiency = efficiency
        self.latitude = latitude
    
    def get_available_capacity(self, timestamp):
        """
        Get available capacity based on time of day and year.
        
        Args:
            timestamp (pd.Timestamp): Current time
            
        Returns:
            float: Available capacity in MW
        """
        if timestamp is None:
            return 0
        
        # Calculate solar irradiance based on time of day and year
        hour = timestamp.hour + timestamp.minute / 60
        day_of_year = timestamp.dayofyear
        
        # Approximate daylight hours based on latitude and day of year
        day_angle = 2 * np.pi * (day_of_year - 1) / 365
        declination = 23.45 * np.sin(day_angle - 2 * np.pi * 60 / 365) * np.pi / 180
        
        # Daylight hours calculation
        cos_hour_angle = -np.tan(self.latitude * np.pi / 180) * np.tan(declination)
        cos_hour_angle = max(-1, min(1, cos_hour_angle))  # Ensure within valid range
        daylight_hours = 2 * np.arccos(cos_hour_angle) * 12 / np.pi
        
        # Sunrise and sunset times
        sunrise = 12 - daylight_hours / 2
        sunset = 12 + daylight_hours / 2
        
        # No production outside daylight hours
        if hour < sunrise or hour > sunset:
            return 0
        
        # Solar irradiance peaks at noon
        solar_factor = np.sin(np.pi * (hour - sunrise) / (sunset - sunrise))
        
        # Seasonal variation (higher in summer)
        seasonal_factor = 0.5 + 0.5 * np.sin((day_of_year - 172) * 2 * np.pi / 365)
        
        # Random cloud cover (0.5-1.0 multiplier)
        cloud_factor = 0.5 + 0.5 * np.random.random()
        
        # Combine factors
        available_capacity = self.capacity * solar_factor * seasonal_factor * cloud_factor
        
        return max(0, available_capacity)


class WindFarm(EnergySource):
    """Wind farm energy source with stochastic wind patterns."""
    
    def __init__(self, capacity, cost_per_mwh=0, emission_factor=0, 
                 mean_wind_speed=8.0, wind_volatility=0.3):
        """
        Initialize a wind farm energy source.
        
        Args:
            capacity (float): Maximum capacity in MW
            cost_per_mwh (float): Cost per MWh (typically very low for wind)
            emission_factor (float): CO2 emissions (typically zero for wind)
            mean_wind_speed (float): Mean wind speed in m/s
            wind_volatility (float): Volatility of wind speed
        """
        super().__init__("Wind Farm", capacity, cost_per_mwh, emission_factor, ramp_rate=0.5)
        self.mean_wind_speed = mean_wind_speed
        self.wind_volatility = wind_volatility
        self.current_wind_speed = mean_wind_speed
        
        # Wind turbine power curve parameters
        self.cut_in_speed = 3.0  # m/s
        self.rated_speed = 12.0  # m/s
        self.cut_out_speed = 25.0  # m/s
    
    def _update_wind_speed(self, timestamp):
        """
        Update wind speed using a mean-reverting stochastic process.
        
        Args:
            timestamp (pd.Timestamp): Current time
        """
        if timestamp is None:
            return
        
        # Mean reversion factor
        mean_reversion = 0.1
        
        # Seasonal component (windier in winter)
        day_of_year = timestamp.dayofyear
        seasonal_factor = 1.0 + 0.2 * np.sin((day_of_year + 60) * 2 * np.pi / 365)
        seasonal_mean = self.mean_wind_speed * seasonal_factor
        
        # Diurnal component (windier during day)
        hour = timestamp.hour
        diurnal_factor = 1.0 + 0.1 * np.sin((hour - 6) * 2 * np.pi / 24)
        
        # Combined mean
        adjusted_mean = seasonal_mean * diurnal_factor
        
        # Mean-reverting random walk
        drift = mean_reversion * (adjusted_mean - self.current_wind_speed)
        volatility = self.wind_volatility * np.sqrt(self.current_wind_speed)
        random_component = volatility * np.random.normal()
        
        # Update wind speed
        self.current_wind_speed = max(0, self.current_wind_speed + drift + random_component)
    
    def _wind_to_power(self, wind_speed):
        """
        Convert wind speed to power output using a power curve.
        
        Args:
            wind_speed (float): Wind speed in m/s
            
        Returns:
            float: Power output as a fraction of capacity (0-1)
        """
        if wind_speed < self.cut_in_speed or wind_speed > self.cut_out_speed:
            return 0
        elif wind_speed >= self.rated_speed:
            return 1.0
        else:
            # Cubic relationship between cut-in and rated speed
            return ((wind_speed - self.cut_in_speed) / 
                    (self.rated_speed - self.cut_in_speed)) ** 3
    
    def get_available_capacity(self, timestamp):
        """
        Get available capacity based on current wind speed.
        
        Args:
            timestamp (pd.Timestamp): Current time
            
        Returns:
            float: Available capacity in MW
        """
        self._update_wind_speed(timestamp)
        power_fraction = self._wind_to_power(self.current_wind_speed)
        return self.capacity * power_fraction


class HydroPlant(EnergySource):
    """Hydroelectric power plant with seasonal water availability."""
    
    def __init__(self, capacity, cost_per_mwh, emission_factor=0.01, 
                 reservoir_capacity=None, initial_level=0.7):
        """
        Initialize a hydroelectric power plant.
        
        Args:
            capacity (float): Maximum capacity in MW
            cost_per_mwh (float): Cost per MWh
            emission_factor (float): CO2 emissions (typically very low for hydro)
            reservoir_capacity (float, optional): Reservoir capacity in MWh
            initial_level (float): Initial reservoir level (0-1)
        """
        super().__init__("Hydro Plant", capacity, cost_per_mwh, emission_factor, ramp_rate=0.3)
        
        # If reservoir capacity not specified, assume 1 week at full capacity
        self.reservoir_capacity = reservoir_capacity if reservoir_capacity else capacity * 24 * 7
        self.reservoir_level = initial_level * self.reservoir_capacity
        
        # Seasonal inflow parameters
        self.base_inflow = capacity * 0.5  # Base inflow in MW
        self.seasonal_amplitude = 0.5  # Seasonal variation amplitude
    
    def _calculate_inflow(self, timestamp):
        """
        Calculate water inflow based on season.
        
        Args:
            timestamp (pd.Timestamp): Current time
            
        Returns:
            float: Inflow in MW-equivalent
        """
        if timestamp is None:
            return self.base_inflow
        
        # Seasonal pattern (higher in spring due to snowmelt)
        day_of_year = timestamp.dayofyear
        seasonal_factor = 1.0 + self.seasonal_amplitude * np.sin((day_of_year - 90) * 2 * np.pi / 365)
        
        # Random variation (0.8-1.2 multiplier)
        random_factor = 0.8 + 0.4 * np.random.random()
        
        return self.base_inflow * seasonal_factor * random_factor
    
    def update_reservoir(self, output, timestamp, hours=1):
        """
        Update reservoir level based on output and inflow.
        
        Args:
            output (float): Power output in MW
            timestamp (pd.Timestamp): Current time
            hours (float): Time period in hours
            
        Returns:
            float: New reservoir level
        """
        inflow = self._calculate_inflow(timestamp) * hours
        outflow = output * hours
        
        # Update reservoir level
        self.reservoir_level = min(
            self.reservoir_capacity,
            max(0, self.reservoir_level + inflow - outflow)
        )
        
        return self.reservoir_level
    
    def get_available_capacity(self, timestamp):
        """
        Get available capacity based on reservoir level.
        
        Args:
            timestamp (pd.Timestamp): Current time
            
        Returns:
            float: Available capacity in MW
        """
        # If reservoir is empty, only inflow is available
        if self.reservoir_level <= 0:
            return self._calculate_inflow(timestamp)
        
        # Otherwise, full capacity is available (limited by reservoir)
        max_from_reservoir = self.reservoir_level / 1.0  # Assume 1 hour operation
        return min(self.capacity, max_from_reservoir)
    
    def set_output(self, output, timestamp=None):
        """
        Set the output level, respecting ramp rate and reservoir constraints.
        
        Args:
            output (float): Desired output in MW
            timestamp (pd.Timestamp, optional): Current time
            
        Returns:
            float: Actual output after applying constraints
        """
        # Apply standard constraints
        output = super().set_output(output, timestamp)
        
        # Update reservoir
        self.update_reservoir(output, timestamp)
        
        return output


class FossilFuelPlant(EnergySource):
    """Fossil fuel power plant with fuel costs and emissions."""
    
    def __init__(self, name, capacity, cost_per_mwh, emission_factor, 
                 min_output=0.3, ramp_rate=0.1, startup_cost=1000, 
                 maintenance_interval=1000, maintenance_duration=24):
        """
        Initialize a fossil fuel power plant.
        
        Args:
            name (str): Name of the plant (e.g., "Coal", "Natural Gas")
            capacity (float): Maximum capacity in MW
            cost_per_mwh (float): Base cost per MWh
            emission_factor (float): CO2 emissions in tons per MWh
            min_output (float): Minimum output as fraction of capacity when running
            ramp_rate (float): Maximum change in output per hour (% of capacity)
            startup_cost (float): Cost to start up the plant from cold
            maintenance_interval (float): Hours between maintenance
            maintenance_duration (float): Hours of maintenance
        """
        super().__init__(name, capacity, cost_per_mwh, emission_factor, ramp_rate)
        self.min_output = min_output
        self.startup_cost = startup_cost
        self.maintenance_interval = maintenance_interval
        self.maintenance_duration = maintenance_duration
        
        # State variables
        self.is_running = False
        self.hours_since_startup = 0
        self.hours_since_maintenance = 0
        self.in_maintenance = False
        self.maintenance_hours_left = 0
    
    def get_available_capacity(self, timestamp):
        """
        Get available capacity considering maintenance status.
        
        Args:
            timestamp (pd.Timestamp): Current time
            
        Returns:
            float: Available capacity in MW
        """
        if self.in_maintenance:
            return 0
        return self.capacity
    
    def update_state(self, hours=1):
        """
        Update plant state after operating for specified hours.
        
        Args:
            hours (float): Time period in hours
        """
        if self.is_running:
            self.hours_since_startup += hours
            self.hours_since_maintenance += hours
            
            # Check if maintenance is needed
            if self.hours_since_maintenance >= self.maintenance_interval:
                self.in_maintenance = True
                self.maintenance_hours_left = self.maintenance_duration
                self.is_running = False
        
        # Update maintenance status
        if self.in_maintenance:
            self.maintenance_hours_left -= hours
            if self.maintenance_hours_left <= 0:
                self.in_maintenance = False
                self.hours_since_maintenance = 0
    
    def set_output(self, output, timestamp=None):
        """
        Set the output level, respecting plant constraints.
        
        Args:
            output (float): Desired output in MW
            timestamp (pd.Timestamp, optional): Current time
            
        Returns:
            float: Actual output after applying constraints
        """
        # Check if in maintenance
        if self.in_maintenance:
            return 0
        
        # Check if starting up
        was_running = self.is_running
        
        # If output is below minimum and not zero, set to minimum
        if 0 < output < self.capacity * self.min_output:
            output = self.capacity * self.min_output
        
        # Apply standard constraints
        output = super().set_output(output, timestamp)
        
        # Update running status
        self.is_running = (output > 0)
        
        # Update plant state
        self.update_state()
        
        return output
    
    def get_cost(self, output):
        """
        Calculate cost including startup cost if applicable.
        
        Args:
            output (float): Output in MW
            
        Returns:
            float: Cost in currency units
        """
        # Base cost
        cost = super().get_cost(output)
        
        # Add startup cost if starting from off state
        if output > 0 and not self.is_running:
            cost += self.startup_cost
        
        return cost


class SupplySimulator:
    """
    Class for simulating energy supply from different sources.
    """
    
    def __init__(self, include_solar=True, include_wind=True, include_hydro=True,
                 include_nuclear=True, include_gas=True, include_coal=True):
        """
        Initialize the supply simulator.
        
        Args:
            include_solar (bool): Whether to include solar energy
            include_wind (bool): Whether to include wind energy
            include_hydro (bool): Whether to include hydro energy
            include_nuclear (bool): Whether to include nuclear energy
            include_gas (bool): Whether to include gas energy
            include_coal (bool): Whether to include coal energy
        """
        self.include_solar = include_solar
        self.include_wind = include_wind
        self.include_hydro = include_hydro
        self.include_nuclear = include_nuclear
        self.include_gas = include_gas
        self.include_coal = include_coal
        
        # Initialize time parameters
        self.start_time = datetime(2023, 1, 1, 0, 0, 0)
        self.current_time = self.start_time
        self.time_step = timedelta(hours=1)
        
        # Initialize supply capacities (MW)
        self.capacities = {
            'solar': 500 if include_solar else 0,
            'wind': 800 if include_wind else 0,
            'hydro': 600 if include_hydro else 0,
            'nuclear': 1000 if include_nuclear else 0,
            'gas': 1200 if include_gas else 0,
            'coal': 1500 if include_coal else 0
        }
        
        # Initialize emission factors (tons CO2 / MWh)
        self.emission_factors = {
            'solar': 0.0,
            'wind': 0.0,
            'hydro': 0.0,
            'nuclear': 0.0,
            'gas': 0.4,
            'coal': 0.9
        }
        
        # Initialize cost factors ($ / MWh)
        self.cost_factors = {
            'solar': 0.0,
            'wind': 0.0,
            'hydro': 5.0,
            'nuclear': 10.0,
            'gas': 40.0,
            'coal': 30.0
        }
        
        # Initialize availability factors (0-1)
        self.availability_factors = {
            'solar': self._calculate_solar_availability(self.current_time),
            'wind': self._calculate_wind_availability(self.current_time),
            'hydro': self._calculate_hydro_availability(self.current_time),
            'nuclear': 0.9,  # High availability
            'gas': 0.95,     # Very high availability
            'coal': 0.9      # High availability
        }
        
        # Initialize supply data
        self.supply_data = pd.DataFrame()
        
        # Create and populate sources list with actual energy source objects
        self.sources = []
        
        # Add solar if included
        if include_solar:
            # SolarPV constructor sets name to "Solar PV" internally
            self.sources.append(SolarPV(
                capacity=self.capacities['solar'],
                cost_per_mwh=self.cost_factors['solar'],
                emission_factor=self.emission_factors['solar']
            ))
        
        # Add wind if included
        if include_wind:
            # WindFarm constructor should set name to "Wind Farm"
            self.sources.append(WindFarm(
                capacity=self.capacities['wind'],
                cost_per_mwh=self.cost_factors['wind'],
                emission_factor=self.emission_factors['wind']
            ))
        
        # Add hydro if included
        if include_hydro:
            # HydroPlant constructor should set name to "Hydro"
            self.sources.append(HydroPlant(
                capacity=self.capacities['hydro'],
                cost_per_mwh=self.cost_factors['hydro'],
                emission_factor=self.emission_factors['hydro']
            ))
        
        # Add nuclear if included
        if include_nuclear:
            self.sources.append(EnergySource(
                name="Nuclear",
                capacity=self.capacities['nuclear'],
                cost_per_mwh=self.cost_factors['nuclear'],
                emission_factor=self.emission_factors['nuclear'],
                ramp_rate=0.05  # Nuclear has slow ramp rate
            ))
        
        # Add gas if included
        if include_gas:
            self.sources.append(FossilFuelPlant(
                name="Gas",
                capacity=self.capacities['gas'],
                cost_per_mwh=self.cost_factors['gas'],
                emission_factor=self.emission_factors['gas'],
                ramp_rate=0.2  # Gas has fast ramp rate
            ))
        
        # Add coal if included
        if include_coal:
            self.sources.append(FossilFuelPlant(
                name="Coal",
                capacity=self.capacities['coal'],
                cost_per_mwh=self.cost_factors['coal'],
                emission_factor=self.emission_factors['coal'],
                ramp_rate=0.1  # Coal has moderate ramp rate
            ))
        
        # Calculate current available supply
        self.current_supply = self._calculate_available_supply(self.current_time)
    
    def _calculate_solar_availability(self, time):
        """
        Calculate solar availability based on time of day and season.
        
        Args:
            time (datetime): Current time
            
        Returns:
            float: Solar availability factor (0-1)
        """
        hour = time.hour
        month = time.month
        
        # No solar at night
        if hour < 6 or hour >= 20:
            return 0.0
        
        # Peak solar hours (10 AM - 4 PM)
        if 10 <= hour < 16:
            peak_factor = 1.0
        # Ramp up (6 AM - 10 AM)
        elif 6 <= hour < 10:
            peak_factor = (hour - 6) / 4
        # Ramp down (4 PM - 8 PM)
        else:
            peak_factor = (20 - hour) / 4
        
        # Seasonal adjustment
        if 3 <= month <= 8:  # Spring and Summer
            season_factor = 1.0
        elif month in [2, 9]:  # Late Winter, Early Fall
            season_factor = 0.7
        else:  # Winter
            season_factor = 0.5
        
        # Random weather effects
        weather_factor = np.random.uniform(0.6, 1.0)
        
        return peak_factor * season_factor * weather_factor
    
    def _calculate_wind_availability(self, time):
        """
        Calculate wind availability based on time of day and season.
        
        Args:
            time (datetime): Current time
            
        Returns:
            float: Wind availability factor (0-1)
        """
        hour = time.hour
        month = time.month
        
        # Wind tends to be stronger at night
        if hour < 6 or hour >= 20:
            time_factor = np.random.uniform(0.7, 1.0)
        else:
            time_factor = np.random.uniform(0.4, 0.9)
        
        # Seasonal adjustment
        if month in [3, 4, 9, 10]:  # Spring and Fall
            season_factor = np.random.uniform(0.8, 1.0)
        elif month in [12, 1, 2]:  # Winter
            season_factor = np.random.uniform(0.7, 0.9)
        else:  # Summer
            season_factor = np.random.uniform(0.5, 0.8)
        
        return time_factor * season_factor
    
    def _calculate_hydro_availability(self, time):
        """
        Calculate hydro availability based on season.
        
        Args:
            time (datetime): Current time
            
        Returns:
            float: Hydro availability factor (0-1)
        """
        month = time.month
        
        # Seasonal adjustment
        if month in [3, 4, 5]:  # Spring (snowmelt)
            base_factor = np.random.uniform(0.8, 1.0)
        elif month in [6, 7, 8]:  # Summer (dry season)
            base_factor = np.random.uniform(0.6, 0.8)
        elif month in [9, 10, 11]:  # Fall (rainy season)
            base_factor = np.random.uniform(0.7, 0.9)
        else:  # Winter
            base_factor = np.random.uniform(0.7, 0.85)
        
        return base_factor
    
    def _calculate_available_supply(self, time):
        """
        Calculate available supply from all sources at a specific time.
        
        Args:
            time (datetime): Time to calculate supply for
            
        Returns:
            dict: Available supply from each source
        """
        # Update availability factors
        self.availability_factors = {
            'solar': self._calculate_solar_availability(time),
            'wind': self._calculate_wind_availability(time),
            'hydro': self._calculate_hydro_availability(time),
            'nuclear': 0.9,  # High availability
            'gas': 0.95,     # Very high availability
            'coal': 0.9      # High availability
        }
        
        # Calculate available supply
        available_supply = {}
        for source, capacity in self.capacities.items():
            available_supply[source] = capacity * self.availability_factors[source]
        
        return available_supply
    
    def generate_data(self, days=365, save_to_file=True, file_path=None):
        """
        Generate synthetic supply data for a specified number of days.
        
        Args:
            days (int): Number of days to generate data for
            save_to_file (bool): Whether to save data to a file
            file_path (str): Path to save the data file
            
        Returns:
            pd.DataFrame: Generated supply data
        """
        # Reset time
        self.current_time = self.start_time
        
        # Generate data
        timestamps = []
        supply_data = {source: [] for source in self.capacities}
        
        for _ in range(days * 24):  # Hourly data for specified days
            available_supply = self._calculate_available_supply(self.current_time)
            timestamps.append(self.current_time)
            
            for source in self.capacities:
                supply_data[source].append(available_supply[source])
            
            self.current_time += self.time_step
        
        # Create DataFrame
        data = {'timestamp': timestamps}
        data.update(supply_data)
        self.supply_data = pd.DataFrame(data)
        
        # Save to file if requested
        if save_to_file:
            if file_path is None:
                file_path = os.path.join('data', 'supply_data.csv')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save to CSV
            self.supply_data.to_csv(file_path, index=False)
            print(f"Supply data saved to {file_path}")
        
        return self.supply_data
    
    def plot_supply(self, days=7, save_to_file=True, file_path=None):
        """
        Plot supply data for visualization.
        
        Args:
            days (int): Number of days to plot
            save_to_file (bool): Whether to save plot to a file
            file_path (str): Path to save the plot file
        """
        if self.supply_data.empty:
            print("No supply data available. Generate data first.")
            return
        
        # Get data for specified days
        plot_data = self.supply_data.iloc[:days*24]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot each source
        for source in self.capacities:
            if self.capacities[source] > 0:
                plt.plot(plot_data['timestamp'], plot_data[source], label=source.capitalize())
        
        # Plot total supply
        total_supply = plot_data[[s for s in self.capacities]].sum(axis=1)
        plt.plot(plot_data['timestamp'], total_supply, 'k--', label='Total Supply')
        
        plt.title(f'Synthetic Energy Supply - {days} Days')
        plt.xlabel('Time')
        plt.ylabel('Supply (MW)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save to file if requested
        if save_to_file:
            if file_path is None:
                file_path = os.path.join('plots', 'supply_plot.png')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save plot
            plt.savefig(file_path)
            print(f"Supply plot saved to {file_path}")
        
        plt.show()
    
    def update(self):
        """
        Update the current time and available supply.
        """
        self.current_time += self.time_step
        self.current_supply = self._calculate_available_supply(self.current_time)
    
    def get_available_supply(self):
        """
        Get the current available supply from all sources.
        
        Returns:
            dict: Available supply from each source
        """
        return self.current_supply
    
    def get_available_capacities(self, timestamp=None):
        """
        Get available capacities at the given timestamp.
        
        Args:
            timestamp (datetime, optional): Current time
            
        Returns:
            dict: Available capacities by source name
        """
        if timestamp is None:
            timestamp = self.current_time
        
        # Get available supply by source type (solar, wind, etc.)
        available_by_type = self._calculate_available_supply(timestamp)
        
        # Map to actual source names (Solar PV, Wind Farm, etc.)
        available_by_name = {}
        
        # Create a mapping from source type to source name
        source_type_to_name = {}
        for source in self.sources:
            if "Solar" in source.name:
                source_type_to_name['solar'] = source.name
            elif "Wind" in source.name:
                source_type_to_name['wind'] = source.name
            elif "Hydro" in source.name:
                source_type_to_name['hydro'] = source.name
            elif "Nuclear" in source.name:
                source_type_to_name['nuclear'] = source.name
            elif "Gas" in source.name:
                source_type_to_name['gas'] = source.name
            elif "Coal" in source.name:
                source_type_to_name['coal'] = source.name
        
        # Map available supply to source names
        for source_type, capacity in available_by_type.items():
            if source_type in source_type_to_name:
                available_by_name[source_type_to_name[source_type]] = capacity
        
        return available_by_name
    
    def get_emission_factor(self, source):
        """
        Get the emission factor for a specific energy source.
        
        Args:
            source (str): Energy source
            
        Returns:
            float: Emission factor (tons CO2 / MWh)
        """
        return self.emission_factors.get(source, 0.0)
    
    def get_cost_factor(self, source):
        """
        Get the cost factor for a specific energy source.
        
        Args:
            source (str): Energy source
            
        Returns:
            float: Cost factor ($ / MWh)
        """
        return self.cost_factors.get(source, 0.0)
    
    def get_source_by_name(self, name):
        """
        Get a source by name.
        
        Args:
            name (str): Source name
            
        Returns:
            object: Source object or None if not found
        """
        # This is a compatibility method for code that expects source objects
        # In this simplified implementation, we return a dictionary with the source properties
        if name in self.capacities:
            return {
                'name': name,
                'capacity': self.capacities[name],
                'cost_per_mwh': self.cost_factors[name],
                'emission_factor': self.emission_factors[name]
            }
        return None
    
    def set_outputs(self, dispatch, timestamp=None):
        """
        Set outputs for all sources.
        
        Args:
            dispatch (dict): Desired outputs by source
            timestamp (datetime, optional): Current time
            
        Returns:
            dict: Actual outputs after applying constraints
        """
        # In this simplified implementation, we just return the dispatch
        # In a more complex implementation, this would apply constraints
        return dispatch
    
    def get_costs(self, outputs):
        """
        Calculate costs for the given outputs.
        
        Args:
            outputs (dict): Outputs by source
            
        Returns:
            dict: Costs by source
        """
        costs = {}
        for source, output in outputs.items():
            costs[source] = output * self.cost_factors.get(source, 0)
        return costs
    
    def get_emissions(self, outputs):
        """
        Calculate emissions for the given outputs.
        
        Args:
            outputs (dict): Outputs by source
            
        Returns:
            dict: Emissions by source
        """
        emissions = {}
        for source, output in outputs.items():
            emissions[source] = output * self.emission_factors.get(source, 0)
        return emissions
    
    def _merit_order_dispatch(self, demand, timestamp=None):
        """
        Dispatch energy sources in merit order (cheapest first).
        
        Args:
            demand (float): Demand to meet in MW
            timestamp (datetime, optional): Current time
            
        Returns:
            dict: Dictionary mapping source names to outputs
        """
        if timestamp is None:
            timestamp = self.current_time
        
        # Get available capacities
        available_capacities = self.get_available_capacities(timestamp)
        
        # Create a mapping from source type to source name
        source_type_to_name = {}
        for source in self.sources:
            if "Solar" in source.name:
                source_type_to_name['solar'] = source.name
            elif "Wind" in source.name:
                source_type_to_name['wind'] = source.name
            elif "Hydro" in source.name:
                source_type_to_name['hydro'] = source.name
            elif "Nuclear" in source.name:
                source_type_to_name['nuclear'] = source.name
            elif "Gas" in source.name:
                source_type_to_name['gas'] = source.name
            elif "Coal" in source.name:
                source_type_to_name['coal'] = source.name
        
        # Sort sources by cost
        sorted_sources = sorted(
            [(source_type, self.cost_factors[source_type]) for source_type in self.capacities if source_type in source_type_to_name],
            key=lambda x: x[1]
        )
        
        # Dispatch in merit order
        dispatch = {source.name: 0 for source in self.sources}
        remaining_demand = demand
        
        for source_type, _ in sorted_sources:
            source_name = source_type_to_name[source_type]
            available = available_capacities[source_name]
            if available > 0 and remaining_demand > 0:
                output = min(available, remaining_demand)
                dispatch[source_name] = output
                remaining_demand -= output
        
        return dispatch
    
    def create_default_portfolio(self):
        """
        Create a default portfolio of energy sources.
        This is a compatibility method for code that expects a portfolio.
        """
        # The portfolio is already created in __init__
        # This method is just for compatibility
        pass
    
    def reset(self):
        """
        Reset the supply simulator to the start time.
        """
        self.current_time = self.start_time
        self.current_supply = self._calculate_available_supply(self.current_time)

def main():
    """
    Main function to demonstrate the supply simulator.
    """
    # Create supply simulator
    supply_sim = SupplySimulator()
    
    # Generate data for one year
    supply_data = supply_sim.generate_data(days=365)
    
    # Plot one week of data
    supply_sim.plot_supply(days=7)
    
    print("Supply simulation complete.")

if __name__ == "__main__":
    main() 