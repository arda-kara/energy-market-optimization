#!/usr/bin/env python
"""
Debug script to diagnose tensor shape mismatches in the RL agents.
This script provides utilities to track tensor shapes throughout the code.
"""

import os
import sys
import numpy as np
import torch
import logging
from datetime import datetime
import argparse

from rl_environment.energy_env import EnergyMarketEnv
from agents.sac_agent import SACAgent
from agents.ppo_agent import PPOAgent

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_tensor_shapes.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("debug_tensor_shapes")

def debug_environment(env_name="energy", episode_length=24):
    """Debug the environment's state and action spaces."""
    logger.info(f"Debugging environment: {env_name}")
    
    # Create environment
    if env_name == "energy":
        env = EnergyMarketEnv(episode_length=episode_length)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    logger.info(f"Environment state dimension: {state_dim}")
    logger.info(f"Environment action dimension: {action_dim}")
    
    # Reset environment
    state, _ = env.reset()
    logger.info(f"Initial state shape: {state.shape}")
    logger.info(f"Initial state: {state}")
    
    # Take a random action
    action = env.action_space.sample()
    logger.info(f"Random action shape: {action.shape}")
    logger.info(f"Random action: {action}")
    
    # Step environment
    next_state, reward, terminated, truncated, info = env.step(action)
    logger.info(f"Next state shape: {next_state.shape}")
    logger.info(f"Reward: {reward}")
    logger.info(f"Info: {info}")
    
    return env

def debug_agent_initialization(env, agent_type="sac", hidden_dim=64):
    """Debug agent initialization."""
    logger.info(f"Debugging {agent_type.upper()} agent initialization")
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    if agent_type.lower() == "sac":
        agent = SACAgent(
            env=env,
            hidden_dim=hidden_dim,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            auto_alpha=True,
            debug_mode=True
        )
    elif agent_type.lower() == "ppo":
        agent = PPOAgent(
            env=env,
            hidden_dim=hidden_dim,
            lr=3e-4,
            gamma=0.99,
            lam=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    logger.info(f"{agent_type.upper()} agent initialized successfully")
    
    # Log network architectures
    if agent_type.lower() == "sac":
        logger.info("Actor Network Architecture:")
        for name, param in agent.actor.named_parameters():
            logger.info(f"  {name}: {param.shape}")
        
        logger.info("Critic 1 Network Architecture:")
        for name, param in agent.critic1.named_parameters():
            logger.info(f"  {name}: {param.shape}")
    elif agent_type.lower() == "ppo":
        logger.info("Actor-Critic Network Architecture:")
        for name, param in agent.actor_critic.named_parameters():
            logger.info(f"  {name}: {param.shape}")
    
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
        logger.info(f"Action shape: {np.shape(action)}")
        logger.info(f"Action: {action}")
    except Exception as e:
        logger.error(f"Error getting action: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return state, action

def debug_agent_step(agent, env, state):
    """Debug agent step."""
    logger.info("Debugging agent step")
    
    # Get action
    action = agent.get_action(state)
    
    # Step environment
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    logger.info(f"Next state shape: {np.shape(next_state)}")
    logger.info(f"Reward: {reward}")
    logger.info(f"Done: {done}")
    logger.info(f"Info: {info}")
    
    return next_state, reward, done, info

def debug_agent_training(agent, env, num_steps=100):
    """Debug agent training."""
    logger.info(f"Debugging agent training for {num_steps} steps")
    
    # Train the agent for a small number of steps
    try:
        if isinstance(agent, SACAgent):
            metrics = agent.train(
                env=env,
                total_timesteps=num_steps,
                batch_size=64,
                buffer_size=1000,
                update_after=50,
                update_every=10,
                num_eval_episodes=2,
                eval_freq=50
            )
        elif isinstance(agent, PPOAgent):
            metrics = agent.train(
                env=env,
                total_timesteps=num_steps,
                buffer_size=200,
                batch_size=64,
                update_epochs=5,
                num_eval_episodes=2,
                eval_freq=50
            )
        else:
            raise ValueError(f"Unknown agent type: {type(agent)}")
        
        logger.info("Training completed successfully")
        
        # Log training metrics
        logger.info("Training metrics:")
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0:
                logger.info(f"  {key}: {value[-1]}")
            elif not isinstance(value, list):
                logger.info(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Run the debugging script."""
    parser = argparse.ArgumentParser(description="Debug tensor shapes in RL agents")
    parser.add_argument("--agent", type=str, default="sac", choices=["sac", "ppo"], help="Agent type to debug")
    parser.add_argument("--env", type=str, default="energy", help="Environment to debug")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension of the networks")
    parser.add_argument("--episode-length", type=int, default=24, help="Episode length")
    
    args = parser.parse_args()
    
    logger.info("Starting tensor shape debugging")
    logger.info(f"Agent: {args.agent}")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Training steps: {args.steps}")
    logger.info(f"Hidden dimension: {args.hidden_dim}")
    logger.info(f"Episode length: {args.episode_length}")
    
    # Debug environment
    env = debug_environment(env_name=args.env, episode_length=args.episode_length)
    
    # Debug agent initialization
    agent = debug_agent_initialization(env, agent_type=args.agent, hidden_dim=args.hidden_dim)
    
    # Debug agent action
    state, action = debug_agent_action(agent, env)
    
    # Debug agent step
    next_state, reward, done, info = debug_agent_step(agent, env, state)
    
    # Debug agent training
    debug_agent_training(agent, env, num_steps=args.steps)
    
    logger.info("Debugging completed")

if __name__ == "__main__":
    main() 