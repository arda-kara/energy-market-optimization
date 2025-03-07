#!/usr/bin/env python
"""
Test script to verify that the SAC agent can train successfully.
"""

import os
import numpy as np
import torch
import logging
import sys
import matplotlib.pyplot as plt
from datetime import datetime

from rl_environment.energy_env import EnergyMarketEnv
from agents.sac_agent import SACAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_sac_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("test_sac_training")

def test_sac_training():
    """Test the SAC agent in a real training scenario."""
    logger.info("=== Testing SAC Agent Training ===")
    
    # Create environment
    env = EnergyMarketEnv(episode_length=24)  # Short episode for testing
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    logger.info(f"Environment state dimension: {state_dim}")
    logger.info(f"Environment action dimension: {action_dim}")
    
    # Initialize agent with debug mode enabled
    agent = SACAgent(
        env=env,
        hidden_dim=64,  # Smaller network for faster testing
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        debug_mode=True
    )
    
    logger.info("SAC agent initialized successfully")
    
    # Train the agent for a small number of steps
    total_timesteps = 1000
    logger.info(f"Training agent for {total_timesteps} timesteps")
    
    try:
        metrics = agent.train(
            env=env,
            total_timesteps=total_timesteps,
            batch_size=64,
            buffer_size=10000,
            update_after=100,
            update_every=10,
            num_eval_episodes=2,
            eval_freq=500
        )
        
        logger.info("Training completed successfully")
        
        # Plot training metrics
        plt.figure(figsize=(12, 8))
        
        # Plot episode returns
        plt.subplot(2, 2, 1)
        plt.plot(metrics['timesteps'], metrics['episode_returns'])
        plt.title('Episode Returns')
        plt.xlabel('Timesteps')
        plt.ylabel('Return')
        
        # Plot critic loss
        plt.subplot(2, 2, 2)
        plt.plot(metrics['critic_losses'])
        plt.title('Critic Loss')
        plt.xlabel('Updates')
        plt.ylabel('Loss')
        
        # Plot actor loss
        plt.subplot(2, 2, 3)
        plt.plot(metrics['actor_losses'])
        plt.title('Actor Loss')
        plt.xlabel('Updates')
        plt.ylabel('Loss')
        
        # Plot alpha
        plt.subplot(2, 2, 4)
        plt.plot(metrics['alpha_values'])
        plt.title('Alpha')
        plt.xlabel('Updates')
        plt.ylabel('Value')
        
        plt.tight_layout()
        plt.savefig('sac_training_metrics.png')
        logger.info("Training metrics saved to sac_training_metrics.png")
        
        # Evaluate the trained agent
        eval_return = agent.evaluate(env, num_episodes=5)
        logger.info(f"Evaluation return: {eval_return:.2f}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info("SAC agent training test completed")

def main():
    """Run the test script."""
    try:
        test_sac_training()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 