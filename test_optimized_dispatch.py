#!/usr/bin/env python
"""
Test script to demonstrate the optimized dispatch strategy.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from market_simulator.demand_generator import DemandGenerator
from market_simulator.supply_simulator import SupplySimulator
from market_simulator.grid_simulator import GridSimulator
from market_simulator.pricing_model import PricingModel

def test_optimized_dispatch():
    """Test the optimized dispatch strategy with different objectives."""
    print("\n=== Testing Optimized Dispatch Strategy ===")
    
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
    
    # Run simulations with different objectives
    objectives = ['cost', 'emissions', 'renewable', 'balanced']
    results = {}
    
    for objective in objectives:
        print(f"\nRunning optimized dispatch with {objective} objective...")
        
        # Run simulation
        sim_results = grid_simulator.run_simulation(
            dispatch_strategy='optimized',
            optimization_objective=objective,
            include_stability=True,
            save_results=False
        )
        
        # Debug output
        print("\nDebug information:")
        print(f"  First hour outputs: {sim_results['outputs'].iloc[0]}")
        
        # Get renewable source names
        renewable_sources = []
        for source_name in sim_results['outputs'].iloc[0].keys():
            if "Solar" in source_name or "Wind" in source_name or "Hydro" in source_name:
                renewable_sources.append(source_name)
        
        print(f"  Renewable sources: {renewable_sources}")
        renewable_output = sum(sim_results['outputs'].iloc[0].get(source, 0) for source in renewable_sources)
        total_output = sum(sim_results['outputs'].iloc[0].values())
        print(f"  Renewable output: {renewable_output}")
        print(f"  Total output: {total_output}")
        print(f"  Calculated renewable fraction: {renewable_output/total_output if total_output > 0 else 0}")
        
        # Get summary statistics
        stats = grid_simulator.get_summary_statistics()
        
        # Store results
        results[objective] = {
            'total_cost': stats['total_cost'],
            'total_emissions': stats['total_emissions'],
            'average_renewable_fraction': stats['average_renewable_fraction'],
            'average_grid_stability': stats['average_grid_stability']
        }
        
        # Print summary
        print(f"  Total Cost: ${stats['total_cost']:.2f}")
        print(f"  Total Emissions: {stats['total_emissions']:.2f} tons CO2")
        print(f"  Average Renewable Fraction: {stats['average_renewable_fraction']*100:.2f}%")
        print(f"  Average Grid Stability: {stats['average_grid_stability']:.2f}")
    
    # Compare results
    print("\nComparison of Optimization Objectives:")
    metrics = ['total_cost', 'total_emissions', 'average_renewable_fraction', 'average_grid_stability']
    metric_labels = ['Total Cost ($)', 'Total Emissions (tons CO2)', 'Renewable Usage (%)', 'Grid Stability (0-1)']
    
    # Create DataFrame for comparison
    comparison = pd.DataFrame(index=objectives, columns=metric_labels)
    
    for i, objective in enumerate(objectives):
        comparison.iloc[i, 0] = results[objective]['total_cost']
        comparison.iloc[i, 1] = results[objective]['total_emissions']
        comparison.iloc[i, 2] = results[objective]['average_renewable_fraction'] * 100
        comparison.iloc[i, 3] = results[objective]['average_grid_stability']
    
    print(comparison)
    
    # Plot comparison
    plt.figure(figsize=(12, 10))
    
    # Normalize values for plotting
    normalized = comparison.copy()
    for col in normalized.columns:
        if col == 'Renewable Usage (%)' or col == 'Grid Stability (0-1)':
            # Higher is better for these metrics
            normalized[col] = normalized[col] / normalized[col].max()
        else:
            # Lower is better for these metrics
            normalized[col] = 1 - (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min())
    
    # Plot radar chart
    angles = np.linspace(0, 2*np.pi, len(metric_labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    ax = plt.subplot(111, polar=True)
    
    for i, objective in enumerate(objectives):
        values = normalized.loc[objective].tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=objective.capitalize())
        ax.fill(angles, values, alpha=0.1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    
    # Add legend
    plt.legend(loc='upper right')
    
    plt.title('Comparison of Optimization Objectives')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/optimization_comparison.png')
    print("\nComparison plot saved to plots/optimization_comparison.png")
    
    plt.show()

if __name__ == "__main__":
    # Create required directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    test_optimized_dispatch() 