#!/usr/bin/env python
"""
Debug script to identify dimension mismatches in the SAC agent.
This script adds detailed logging to track tensor shapes throughout the code.
"""

import os
import sys
import numpy as np
import torch
import logging
from datetime import datetime

from rl_environment.energy_env import EnergyMarketEnv
from agents.sac_agent import SACAgent, Actor, Critic, ReplayBuffer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_sac.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("debug_sac")

class DebugSACAgent(SACAgent):
    """
    Extended SAC agent with additional debugging capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the debug SAC agent."""
        logger.info("Initializing Debug SAC Agent")
        super().__init__(*args, **kwargs)
        
        # Log network architectures
        self._log_network_architecture()
    
    def _log_network_architecture(self):
        """Log the architecture of all networks."""
        logger.info(f"State dimension: {self.state_dim}")
        logger.info(f"Action dimension: {self.action_dim}")
        
        # Log actor architecture
        logger.info("Actor Network Architecture:")
        for name, param in self.actor.named_parameters():
            logger.info(f"  {name}: {param.shape}")
        
        # Log critic architectures
        logger.info("Critic 1 Network Architecture:")
        for name, param in self.critic1.named_parameters():
            logger.info(f"  {name}: {param.shape}")
        
        logger.info("Critic 2 Network Architecture:")
        for name, param in self.critic2.named_parameters():
            logger.info(f"  {name}: {param.shape}")
    
    def get_action(self, state, deterministic=False):
        """
        Override get_action to add debugging.
        """
        logger.debug(f"get_action input state shape: {np.shape(state)}")
        
        # Convert state to tensor if it's a numpy array
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).to(self.device)
        else:
            state_tensor = state
            
        logger.debug(f"get_action state_tensor shape: {state_tensor.shape}")
        
        # Get action from parent method
        try:
            action = super().get_action(state, deterministic)
            logger.debug(f"get_action output action shape: {np.shape(action)}")
            return action
        except Exception as e:
            logger.error(f"Error in get_action: {str(e)}")
            logger.error(f"State shape: {np.shape(state)}")
            logger.error(f"State tensor shape: {state_tensor.shape}")
            logger.error(f"Expected state dim: {self.state_dim}")
            raise
    
    def _update_networks(self, batch):
        """
        Override _update_networks to add debugging.
        """
        states, actions, rewards, next_states, dones = batch
        
        logger.debug(f"_update_networks batch shapes:")
        logger.debug(f"  states: {np.shape(states)}")
        logger.debug(f"  actions: {np.shape(actions)}")
        logger.debug(f"  rewards: {np.shape(rewards)}")
        logger.debug(f"  next_states: {np.shape(next_states)}")
        logger.debug(f"  dones: {np.shape(dones)}")
        
        try:
            return super()._update_networks(batch)
        except Exception as e:
            logger.error(f"Error in _update_networks: {str(e)}")
            
            # Convert to tensors for more detailed debugging
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.FloatTensor(actions).to(self.device)
            next_states_tensor = torch.FloatTensor(next_states).to(self.device)
            
            logger.error(f"Tensor shapes:")
            logger.error(f"  states_tensor: {states_tensor.shape}")
            logger.error(f"  actions_tensor: {actions_tensor.shape}")
            logger.error(f"  next_states_tensor: {next_states_tensor.shape}")
            
            # Try to identify where the error occurs
            try:
                logger.debug("Testing actor forward pass...")
                mu, log_std = self.actor(states_tensor)
                logger.debug(f"  mu shape: {mu.shape}")
                logger.debug(f"  log_std shape: {log_std.shape}")
            except Exception as e2:
                logger.error(f"Error in actor forward pass: {str(e2)}")
            
            try:
                logger.debug("Testing critic forward pass...")
                q1 = self.critic1(states_tensor, actions_tensor)
                logger.debug(f"  q1 shape: {q1.shape}")
            except Exception as e2:
                logger.error(f"Error in critic forward pass: {str(e2)}")
            
            raise

# Monkey patch the Actor class to add debugging
original_actor_forward = Actor.forward
def debug_actor_forward(self, state):
    """Add debugging to Actor.forward."""
    logger.debug(f"Actor.forward input state shape: {state.shape}")
    try:
        mu, log_std = original_actor_forward(self, state)
        logger.debug(f"Actor.forward output shapes: mu {mu.shape}, log_std {log_std.shape}")
        return mu, log_std
    except Exception as e:
        logger.error(f"Error in Actor.forward: {str(e)}")
        logger.error(f"Input state shape: {state.shape}")
        logger.error(f"Expected state_dim: {self.state_dim}")
        raise

Actor.forward = debug_actor_forward

# Monkey patch the Critic class to add debugging
original_critic_forward = Critic.forward
def debug_critic_forward(self, state, action):
    """Add debugging to Critic.forward."""
    logger.debug(f"Critic.forward input shapes: state {state.shape}, action {action.shape}")
    try:
        q = original_critic_forward(self, state, action)
        logger.debug(f"Critic.forward output shape: {q.shape}")
        return q
    except Exception as e:
        logger.error(f"Error in Critic.forward: {str(e)}")
        logger.error(f"Input shapes: state {state.shape}, action {action.shape}")
        raise

Critic.forward = debug_critic_forward

def debug_env_step(env):
    """Debug the environment step function."""
    logger.info("Debugging environment step function")
    
    # Reset environment
    state, _ = env.reset()
    logger.info(f"Initial state shape: {state.shape}")
    
    # Take a random action
    action = env.action_space.sample()
    logger.info(f"Random action shape: {action.shape}")
    
    # Step environment
    next_state, reward, terminated, truncated, info = env.step(action)
    logger.info(f"Next state shape: {next_state.shape}")
    logger.info(f"Reward: {reward}")
    logger.info(f"Info: {info}")
    
    return state, action, next_state

def debug_agent_initialization(env):
    """Debug agent initialization."""
    logger.info("Debugging agent initialization")
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    logger.info(f"Environment state dimension: {state_dim}")
    logger.info(f"Environment action dimension: {action_dim}")
    
    # Initialize agent
    agent = DebugSACAgent(
        env=env,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True
    )
    
    return agent

def debug_agent_action(agent, env):
    """Debug agent action selection."""
    logger.info("Debugging agent action selection")
    
    # Reset environment
    state, _ = env.reset()
    logger.info(f"State shape: {state.shape}")
    
    # Get action
    try:
        action = agent.get_action(state)
        logger.info(f"Action shape: {action.shape}")
        logger.info(f"Action: {action}")
    except Exception as e:
        logger.error(f"Error getting action: {str(e)}")
    
    return state, action

def debug_replay_buffer(agent, env):
    """Debug replay buffer operations."""
    logger.info("Debugging replay buffer")
    
    # Create replay buffer
    buffer = ReplayBuffer(
        state_dim=agent.state_dim,
        action_dim=agent.action_dim,
        max_size=1000
    )
    
    # Collect some transitions
    state, _ = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Add to buffer
        buffer.add(state, action, next_state, reward, done)
        
        state = next_state
        if done:
            state, _ = env.reset()
    
    # Sample from buffer
    try:
        batch = buffer.sample(5)
        states, actions, rewards, next_states, dones = batch
        
        logger.info(f"Sampled batch shapes:")
        logger.info(f"  states: {np.shape(states)}")
        logger.info(f"  actions: {np.shape(actions)}")
        logger.info(f"  rewards: {np.shape(rewards)}")
        logger.info(f"  next_states: {np.shape(next_states)}")
        logger.info(f"  dones: {np.shape(dones)}")
    except Exception as e:
        logger.error(f"Error sampling from buffer: {str(e)}")
    
    return buffer

def debug_network_update(agent, buffer):
    """Debug network update."""
    logger.info("Debugging network update")
    
    # Sample batch
    try:
        batch = buffer.sample(64)
        
        # Update networks
        critic_loss, actor_loss, alpha_loss = agent._update_networks(batch)
        
        logger.info(f"Update successful:")
        logger.info(f"  critic_loss: {critic_loss}")
        logger.info(f"  actor_loss: {actor_loss}")
        logger.info(f"  alpha_loss: {alpha_loss}")
    except Exception as e:
        logger.error(f"Error updating networks: {str(e)}")

def main():
    """Run the debugging script."""
    logger.info("Starting SAC agent debugging")
    
    # Create environment
    env = EnergyMarketEnv(episode_length=24)  # Short episode for debugging
    
    # Debug environment
    state, action, next_state = debug_env_step(env)
    
    # Debug agent initialization
    agent = debug_agent_initialization(env)
    
    # Debug agent action
    state, action = debug_agent_action(agent, env)
    
    # Debug replay buffer
    buffer = debug_replay_buffer(agent, env)
    
    # Fill buffer with more transitions for network update
    state, _ = env.reset()
    for _ in range(100):  # Collect more transitions
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        buffer.add(state, action, next_state, reward, done)
        
        state = next_state
        if done:
            state, _ = env.reset()
    
    # Debug network update
    debug_network_update(agent, buffer)
    
    logger.info("Debugging completed")

if __name__ == "__main__":
    main() 