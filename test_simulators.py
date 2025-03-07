#!/usr/bin/env python
"""
Test script to verify that all simulator components are working correctly.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from market_simulator.demand_generator import DemandGenerator
from market_simulator.supply_simulator import SupplySimulator
from market_simulator.grid_simulator import GridSimulator
from market_simulator.pricing_model import PricingModel

def test_demand_generator():
    """Test the DemandGenerator class."""
    print("\n=== Testing DemandGenerator ===")
    
    # Create demand generator
    demand_gen = DemandGenerator(base_demand=1000)
    
    # Test get_demand method
    demand = demand_gen.get_demand()
    print(f"Current demand: {demand:.2f} MW")
    
    # Test update method
    demand_gen.update()
    new_demand = demand_gen.get_demand()
    print(f"Updated demand: {new_demand:.2f} MW")
    
    # Test reset method
    demand_gen.reset()
    reset_demand = demand_gen.get_demand()
    print(f"Reset demand: {reset_demand:.2f} MW")
    
    print("DemandGenerator test completed successfully.")

def test_supply_simulator():
    """Test the SupplySimulator class."""
    print("\n=== Testing SupplySimulator ===")
    
    # Create supply simulator
    supply_sim = SupplySimulator()
    
    # Test get_available_supply method
    available_supply = supply_sim.get_available_supply()
    print("Available supply:")
    for source, amount in available_supply.items():
        print(f"  {source}: {amount:.2f} MW")
    
    # Test get_emission_factor method
    print("\nEmission factors:")
    for source in available_supply:
        emission_factor = supply_sim.get_emission_factor(source)
        print(f"  {source}: {emission_factor:.2f} tons CO2/MWh")
    
    # Test update method
    supply_sim.update()
    new_supply = supply_sim.get_available_supply()
    print("\nUpdated supply:")
    for source, amount in new_supply.items():
        print(f"  {source}: {amount:.2f} MW")
    
    # Test reset method
    supply_sim.reset()
    reset_supply = supply_sim.get_available_supply()
    print("\nReset supply:")
    for source, amount in reset_supply.items():
        print(f"  {source}: {amount:.2f} MW")
    
    print("SupplySimulator test completed successfully.")

def test_grid_simulator():
    """Test the GridSimulator class."""
    print("\n=== Testing GridSimulator ===")
    
    # Create grid simulator
    grid_sim = GridSimulator()
    
    # Test get_grid_state method
    grid_state = grid_sim.get_grid_state()
    print("Initial grid state:")
    for key, value in grid_state.items():
        print(f"  {key}: {value:.2f}")
    
    # Test update method
    demand = 1000
    allocation = {
        'solar': 200,
        'wind': 300,
        'hydro': 100,
        'nuclear': 200,
        'gas': 100,
        'coal': 100
    }
    updated_state = grid_sim.update(demand, allocation)
    print("\nUpdated grid state:")
    for key, value in updated_state.items():
        print(f"  {key}: {value:.2f}")
    
    # Test reset method
    grid_sim.reset()
    reset_state = grid_sim.get_grid_state()
    print("\nReset grid state:")
    for key, value in reset_state.items():
        print(f"  {key}: {value:.2f}")
    
    print("GridSimulator test completed successfully.")

def test_pricing_model():
    """Test the PricingModel class."""
    print("\n=== Testing PricingModel ===")
    
    # Create pricing model
    pricing_model = PricingModel()
    
    # Test get_current_prices method
    current_prices = pricing_model.get_current_prices()
    print("Initial prices:")
    for source, price in current_prices.items():
        print(f"  {source}: ${price:.2f}/MWh")
    
    # Test calculate_prices method
    demand = 1000
    allocation = {
        'solar': 200,
        'wind': 300,
        'hydro': 100,
        'nuclear': 200,
        'gas': 100,
        'coal': 100
    }
    grid_state = {
        'stability': 0.8,
        'imbalance': 0.1,
        'renewable_fraction': 0.6,
        'blackout_risk': 0.05
    }
    updated_prices = pricing_model.calculate_prices(demand, allocation, grid_state)
    print("\nUpdated prices:")
    for source, price in updated_prices.items():
        print(f"  {source}: ${price:.2f}/MWh")
    
    # Test calculate_price method
    market_price = pricing_model.calculate_price(
        demand=1000,
        available_capacity=1200,
        timestamp=datetime.now(),
        renewable_fraction=0.6
    )
    print(f"\nMarket price: ${market_price:.2f}/MWh")
    
    # Test reset method
    pricing_model.reset()
    reset_prices = pricing_model.get_current_prices()
    print("\nReset prices:")
    for source, price in reset_prices.items():
        print(f"  {source}: ${price:.2f}/MWh")
    
    print("PricingModel test completed successfully.")

def test_integration():
    """Test the integration of all simulator components."""
    print("\n=== Testing Integration ===")
    
    # Create simulator components
    demand_gen = DemandGenerator()
    supply_sim = SupplySimulator()
    grid_sim = GridSimulator()
    pricing_model = PricingModel()
    
    # Simulate one day
    print("Simulating one day (24 hours):")
    for hour in range(24):
        # Get demand
        demand = demand_gen.get_demand()
        
        # Get available supply
        available_supply = supply_sim.get_available_supply()
        
        # Allocate supply (using merit order)
        allocation = supply_sim._merit_order_dispatch(demand)
        
        # Update grid state
        grid_state = grid_sim.update(demand, allocation)
        
        # Calculate prices
        prices = pricing_model.calculate_prices(demand, allocation, grid_state)
        
        # Print summary for this hour
        print(f"\nHour {hour}:")
        print(f"  Demand: {demand:.2f} MW")
        print(f"  Total supply: {sum(allocation.values()):.2f} MW")
        print(f"  Grid stability: {grid_state['stability']:.2f}")
        print(f"  Average price: ${np.mean(list(prices.values())):.2f}/MWh")
        
        # Update simulators for next hour
        demand_gen.update()
        supply_sim.update()
    
    print("\nIntegration test completed successfully.")

def main():
    """Run all tests."""
    # Create required directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('logs/tensorboard', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        # Run tests
        test_demand_generator()
        test_supply_simulator()
        test_grid_simulator()
        test_pricing_model()
        test_integration()
        
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 