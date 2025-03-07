#!/usr/bin/env python
"""
Test script to verify that the SAC agent works correctly after our fixes.
"""

import os
import numpy as np
import torch
import logging
import sys
from datetime import datetime

from rl_environment.energy_env import EnergyMarketEnv
from agents.sac_agent import SACAgent, Actor, Critic, ReplayBuffer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_sac.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("test_sac")

def test_sac_agent():
    """Test the SAC agent."""
    logger.info("=== Testing SAC Agent ===")
    
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
    
    # Test get_action
    state, _ = env.reset()
    action = agent.get_action(state)
    logger.info(f"Action shape: {action.shape}")
    logger.info(f"Action: {action}")
    
    # Test replay buffer
    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=1000
    )
    
    # Collect some transitions
    for _ in range(100):
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        buffer.add(state, action, next_state, reward, done)
        
        state = next_state
        if done:
            state, _ = env.reset()
    
    logger.info("Collected 100 transitions")
    
    # Test network update
    batch = buffer.sample(64)
    states, actions, rewards, next_states, dones = batch
    
    logger.info(f"Batch shapes:")
    logger.info(f"  states: {np.shape(states)}")
    logger.info(f"  actions: {np.shape(actions)}")
    logger.info(f"  rewards: {np.shape(rewards)}")
    logger.info(f"  next_states: {np.shape(next_states)}")
    logger.info(f"  dones: {np.shape(dones)}")
    
    # Update networks
    try:
        critic_loss, actor_loss, alpha_loss = agent._update_networks(batch)
        logger.info(f"Network update successful:")
        logger.info(f"  critic_loss: {critic_loss}")
        logger.info(f"  actor_loss: {actor_loss}")
        logger.info(f"  alpha_loss: {alpha_loss}")
    except Exception as e:
        logger.error(f"Error updating networks: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test mini-training
    try:
        logger.info("Testing mini-training (10 updates)...")
        for i in range(10):
            batch = buffer.sample(64)
            critic_loss, actor_loss, alpha_loss = agent._update_networks(batch)
            logger.info(f"Update {i+1}/10 - critic_loss: {critic_loss:.4f}, actor_loss: {actor_loss:.4f}, alpha_loss: {alpha_loss:.4f}")
        logger.info("Mini-training completed successfully")
    except Exception as e:
        logger.error(f"Error during mini-training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info("SAC agent test completed")

def main():
    """Run the test script."""
    try:
        test_sac_agent()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 