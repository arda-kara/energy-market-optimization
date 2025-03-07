import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    The actor outputs the mean and log standard deviation of a Gaussian distribution for each action.
    The critic outputs the value function estimate.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the Actor-Critic network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Dimension of hidden layers
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize network weights.
        
        Args:
            module (nn.Module): Module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: (action_mean, action_logstd, value)
        """
        features = self.feature_extractor(state)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        
        return action_mean, self.actor_logstd, value
    
    def actor(self, state):
        """
        Forward pass through the actor network.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: (action_mean, action_logstd)
        """
        features = self.feature_extractor(state)
        action_mean = self.actor_mean(features)
        
        return action_mean, self.actor_logstd
    
    def get_value(self, state):
        """
        Forward pass through the critic network.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            torch.Tensor: Value estimate
        """
        features = self.feature_extractor(state)
        value = self.critic(features)
        
        return value
    
    def get_action(self, state, deterministic=False):
        """
        Sample an action from the policy.
        
        Args:
            state (torch.Tensor): State tensor
            deterministic (bool): If True, return the mean action
            
        Returns:
            tuple: (action, log_prob, value)
        """
        action_mean, action_logstd, value = self(state)
        
        if deterministic:
            action = action_mean
            log_prob = None
        else:
            action_std = torch.exp(action_logstd)
            distribution = Normal(action_mean, action_std)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, state, action):
        """
        Evaluate actions given states.
        
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            tuple: (log_prob, value, entropy)
        """
        action_mean, action_logstd, value = self(state)
        action_std = torch.exp(action_logstd)
        
        distribution = Normal(action_mean, action_std)
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        
        return log_prob, value, entropy


class PPOBuffer:
    """
    Buffer for storing trajectories collected from the environment.
    
    This buffer supports computing advantages using Generalized Advantage Estimation (GAE).
    """
    
    def __init__(self, state_dim, action_dim, size, gamma=0.99, lam=0.95):
        """
        Initialize the buffer.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            size (int): Maximum size of the buffer
            gamma (float): Discount factor
            lam (float): GAE lambda parameter
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.size = size
        self.gamma = gamma
        self.lam = lam
        
        # Initialize buffers
        self.reset()
    
    def reset(self):
        """Reset the buffer."""
        self.state_buf = np.zeros((self.size, self.state_dim), dtype=np.float32)
        self.action_buf = np.zeros((self.size, self.action_dim), dtype=np.float32)
        self.reward_buf = np.zeros(self.size, dtype=np.float32)
        self.value_buf = np.zeros(self.size, dtype=np.float32)
        self.log_prob_buf = np.zeros(self.size, dtype=np.float32)
        self.done_buf = np.zeros(self.size, dtype=np.float32)
        
        self.ret_buf = np.zeros(self.size, dtype=np.float32)
        self.adv_buf = np.zeros(self.size, dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = self.size
    
    def store(self, state, action, reward, value, log_prob, done):
        """
        Store a transition in the buffer.
        
        Args:
            state (np.ndarray): State
            action (np.ndarray): Action
            reward (float): Reward
            value (float): Value estimate
            log_prob (float): Log probability of the action
            done (bool): Whether the episode is done
        """
        assert self.ptr < self.max_size
        
        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.value_buf[self.ptr] = value
        self.log_prob_buf[self.ptr] = log_prob
        self.done_buf[self.ptr] = done
        
        self.ptr += 1
    
    def finish_path(self, last_value=0):
        """
        Compute returns and advantages for the current path.
        
        Args:
            last_value (float): Value estimate for the last state
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.reward_buf[path_slice], last_value)
        values = np.append(self.value_buf[path_slice], last_value)
        dones = np.append(self.done_buf[path_slice], 0)
        
        # Compute GAE
        deltas = rewards[:-1] + self.gamma * values[1:] * (1 - dones[:-1]) - values[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        
        # Compute returns
        self.ret_buf[path_slice] = self._discount_cumsum(rewards[:-1], self.gamma)
        
        self.path_start_idx = self.ptr
    
    def get(self):
        """
        Get all data from the buffer and normalize advantages.
        
        Returns:
            dict: Dictionary containing all data
        """
        # Make sure the buffer has data
        assert self.ptr > 0, "Buffer is empty"
        
        # Normalize advantages
        adv_mean = np.mean(self.adv_buf[:self.ptr])
        adv_std = np.std(self.adv_buf[:self.ptr]) + 1e-8
        self.adv_buf[:self.ptr] = (self.adv_buf[:self.ptr] - adv_mean) / adv_std
        
        # Convert to tensors
        data = {
            "states": torch.FloatTensor(self.state_buf[:self.ptr]),
            "actions": torch.FloatTensor(self.action_buf[:self.ptr]),
            "advantages": torch.FloatTensor(self.adv_buf[:self.ptr]),
            "returns": torch.FloatTensor(self.ret_buf[:self.ptr]),
            "log_probs": torch.FloatTensor(self.log_prob_buf[:self.ptr]),
            "values": torch.FloatTensor(self.value_buf[:self.ptr])
        }
        
        return data
    
    def _discount_cumsum(self, x, discount):
        """
        Compute discounted cumulative sum.
        
        Args:
            x (np.ndarray): Array to discount
            discount (float): Discount factor
            
        Returns:
            np.ndarray: Discounted cumulative sum
        """
        n = len(x)
        y = np.zeros_like(x)
        y[-1] = x[-1]
        for t in range(n-2, -1, -1):
            y[t] = x[t] + discount * y[t+1]
        return y


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    
    This implementation uses the clipped surrogate objective and
    generalized advantage estimation (GAE).
    """
    
    def __init__(self, env=None, state_dim=None, action_dim=None, hidden_dim=128, 
                 lr=3e-4, gamma=0.99, lam=0.95, clip_ratio=0.2, target_kl=0.01,
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5):
        """
        Initialize the PPO agent.
        
        Args:
            env (gym.Env, optional): Environment (used to get state/action dims if not provided)
            state_dim (int, optional): Dimension of the state space
            action_dim (int, optional): Dimension of the action space
            hidden_dim (int): Dimension of hidden layers
            lr (float): Learning rate
            gamma (float): Discount factor
            lam (float): GAE lambda parameter
            clip_ratio (float): PPO clip ratio
            target_kl (float): Target KL divergence
            entropy_coef (float): Entropy coefficient
            value_coef (float): Value loss coefficient
            max_grad_norm (float): Maximum gradient norm
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # If environment is provided, get state and action dimensions
        if env is not None:
            # Get the observation space shape
            if hasattr(env.observation_space, 'shape'):
                if len(env.observation_space.shape) == 1:
                    state_dim = env.observation_space.shape[0]
                else:
                    # Flatten the observation space if it's not 1D
                    state_dim = np.prod(env.observation_space.shape)
            else:
                raise ValueError("Unsupported observation space type")
            
            # Get the action space shape
            if hasattr(env.action_space, 'shape'):
                if len(env.action_space.shape) == 1:
                    action_dim = env.action_space.shape[0]
                else:
                    # Flatten the action space if it's not 1D
                    action_dim = np.prod(env.action_space.shape)
            else:
                raise ValueError("Unsupported action space type")
        
        # Ensure state_dim and action_dim are provided
        assert state_dim is not None and action_dim is not None, "Must provide env or state_dim and action_dim"
        
        # Store dimensions for later use
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create actor-critic network
        self.ac = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        
        # Store hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize metrics
        self.metrics = {
            'episode_returns': [],
            'episode_lengths': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl': [],
            'explained_variance': []
        }
    
    def get_action(self, state, deterministic=False):
        """
        Get action from policy.
        
        Args:
            state (np.ndarray): State array
            deterministic (bool): If True, return the mean action
            
        Returns:
            tuple: (action, log_prob, value)
        """
        # Convert state to tensor and ensure correct shape
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        # Ensure state has correct shape (batch_size, state_dim)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        
        # Ensure state has the correct size
        if state.shape[1] != self.state_dim:
            raise ValueError(f"State dimension mismatch: expected {self.state_dim}, got {state.shape[1]}")
        
        with torch.no_grad():
            action_mean, action_logstd = self.ac.actor(state)
            value = self.ac.get_value(state)
            
            # Sample action
            if deterministic:
                action = action_mean
                log_prob = None
            else:
                action_std = torch.exp(action_logstd)
                action_dist = Normal(action_mean, action_std)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(dim=-1)
        
        # Convert to numpy
        action_np = action.cpu().numpy()
        log_prob_np = log_prob.cpu().numpy() if log_prob is not None else None
        value_np = value.cpu().numpy().flatten()
        
        return action_np[0], log_prob_np[0] if log_prob_np is not None else None, value_np[0]
    
    def train(self, env, total_timesteps, buffer_size=2048, batch_size=64, 
              update_epochs=10, num_eval_episodes=5, eval_freq=10000, 
              save_freq=50000, save_path=None, tb_writer=None):
        """
        Train the agent.
        
        Args:
            env (gym.Env): Environment to train on
            total_timesteps (int): Total number of timesteps to train for
            buffer_size (int): Size of the buffer
            batch_size (int): Batch size for updates
            update_epochs (int): Number of epochs to update for
            num_eval_episodes (int): Number of episodes to evaluate for
            eval_freq (int): Frequency of evaluation
            save_freq (int): Frequency of saving
            save_path (str): Path to save the model
            tb_writer: TensorBoard writer
            
        Returns:
            dict: Training metrics
        """
        # Create buffer
        buffer = PPOBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            size=buffer_size,
            gamma=self.gamma,
            lam=self.lam
        )
        
        # Initialize metrics
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'kls': [],
            'explained_variances': [],
            'eval_returns': [],
            'eval_timesteps': []
        }
        
        # Initialize environment
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Training loop
        for t in range(total_timesteps):
            # Get action
            action, log_prob, value = self.get_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition in buffer
            buffer.store(state, action, reward, value, log_prob, done)
            
            # Update state
            state = next_state
            
            # Update episode metrics
            episode_reward += reward
            episode_length += 1
            
            # If episode is done, reset environment
            if done:
                # Store episode metrics
                metrics['episode_rewards'].append(episode_reward)
                metrics['episode_lengths'].append(episode_length)
                
                # Reset environment
                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0
            
            # Update policy if buffer is full
            if buffer.ptr == buffer_size:
                policy_loss, value_loss, entropy, kl, explained_var = self._update_policy(buffer, batch_size, update_epochs)
                
                # Store update metrics
                metrics['policy_losses'].append(policy_loss)
                metrics['value_losses'].append(value_loss)
                metrics['entropies'].append(entropy)
                metrics['kls'].append(kl)
                metrics['explained_variances'].append(explained_var)
                
                # Reset buffer
                buffer.reset()
            
            # Evaluate agent
            if (t + 1) % eval_freq == 0 and num_eval_episodes > 0:
                eval_return = self.evaluate(env, num_episodes=num_eval_episodes)
                print(f"Timestep {t+1}/{total_timesteps} | Eval return: {eval_return:.2f}")
                
                # Store evaluation metrics
                metrics['eval_returns'].append(eval_return)
                metrics['eval_timesteps'].append(t + 1)
                
                # Log to TensorBoard
                if tb_writer is not None:
                    tb_writer.add_scalar("eval/return", eval_return, t + 1)
            
            # Save model
            if save_path is not None and (t + 1) % save_freq == 0:
                os.makedirs(save_path, exist_ok=True)
                self.save(os.path.join(save_path, f"ppo_agent_{t+1}.pt"))
        
        # Final evaluation
        if num_eval_episodes > 0:
            eval_return = self.evaluate(env, num_episodes=num_eval_episodes)
            print(f"Final evaluation: {eval_return:.2f}")
            
            # Store evaluation metrics
            metrics['eval_returns'].append(eval_return)
            metrics['eval_timesteps'].append(total_timesteps)
            
            # Log to TensorBoard
            if tb_writer is not None:
                tb_writer.add_scalar("eval/return", eval_return, total_timesteps)
        
        # Save final model
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            self.save(os.path.join(save_path, "ppo_agent_final.pt"))
        
        return metrics
    
    def evaluate(self, env, num_episodes=10, render=False):
        """
        Evaluate the agent on the environment.
        
        Args:
            env (gym.Env): Environment to evaluate on
            num_episodes (int): Number of episodes to evaluate
            render (bool): Whether to render the environment
            
        Returns:
            float: Mean episode return
        """
        episode_returns = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_return = 0
            
            while not done:
                # Select action deterministically
                action, _, _ = self.get_action(state, deterministic=True)
                
                # Take step in environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Update state and return
                state = next_state
                episode_return += reward
                
                # Render if specified
                if render:
                    env.render()
            
            episode_returns.append(episode_return)
        
        # Calculate mean return
        mean_return = np.mean(episode_returns)
        
        return mean_return
    
    def save(self, path):
        """
        Save the agent.
        
        Args:
            path (str): Path to save the agent
        """
        torch.save({
            'model_state_dict': self.ac.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics
        }, path)
    
    def load(self, path):
        """
        Load the agent.
        
        Args:
            path (str): Path to load the agent from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint['metrics']
    
    def plot_metrics(self, save_path=None):
        """
        Plot training metrics.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        
        # Plot episode returns
        axs[0, 0].plot(self.metrics['episode_returns'])
        axs[0, 0].set_title('Episode Returns')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Return')
        axs[0, 0].grid(True)
        
        # Plot episode lengths
        axs[0, 1].plot(self.metrics['episode_lengths'])
        axs[0, 1].set_title('Episode Lengths')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Length')
        axs[0, 1].grid(True)
        
        # Plot policy and value losses
        axs[1, 0].plot(self.metrics['policy_loss'], label='Policy Loss')
        axs[1, 0].plot(self.metrics['value_loss'], label='Value Loss')
        axs[1, 0].set_title('Losses')
        axs[1, 0].set_xlabel('Update')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # Plot entropy
        axs[1, 1].plot(self.metrics['entropy'])
        axs[1, 1].set_title('Entropy')
        axs[1, 1].set_xlabel('Update')
        axs[1, 1].set_ylabel('Entropy')
        axs[1, 1].grid(True)
        
        # Plot KL divergence
        axs[2, 0].plot(self.metrics['kl'])
        axs[2, 0].set_title('KL Divergence')
        axs[2, 0].set_xlabel('Update')
        axs[2, 0].set_ylabel('KL')
        axs[2, 0].grid(True)
        
        # Plot explained variance
        axs[2, 1].plot(self.metrics['explained_variance'])
        axs[2, 1].set_title('Explained Variance')
        axs[2, 1].set_xlabel('Update')
        axs[2, 1].set_ylabel('Explained Var')
        axs[2, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()

    def _update_policy(self, buffer, batch_size, epochs):
        """
        Update policy using the PPO algorithm.
        
        Args:
            buffer (PPOBuffer): Buffer containing collected trajectories
            batch_size (int): Batch size for updates
            epochs (int): Number of epochs to update for
            
        Returns:
            tuple: (policy_loss, value_loss, entropy, kl, explained_variance)
        """
        # Get data from buffer
        data = buffer.get()
        
        # Get tensors
        states = data["states"].to(self.device)
        actions = data["actions"].to(self.device)
        advantages = data["advantages"].to(self.device)
        returns = data["returns"].to(self.device)
        old_log_probs = data["log_probs"].to(self.device)
        
        # Track metrics
        policy_losses = []
        value_losses = []
        entropies = []
        kls = []
        
        # Update policy for several epochs
        for _ in range(epochs):
            # Generate random indices
            indices = torch.randperm(len(states))
            
            # Update in batches
            for start in range(0, len(states), batch_size):
                # Get batch indices
                end = start + batch_size
                if end > len(states):
                    end = len(states)
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Get current log probs and values
                log_probs, values, entropy = self.ac.evaluate_actions(batch_states, batch_actions)
                
                # Calculate ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Calculate surrogate losses
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                
                # Calculate policy loss
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Calculate value loss
                value_loss = 0.5 * ((values - batch_returns) ** 2).mean()
                
                # Calculate total loss
                loss = policy_loss - self.entropy_coef * entropy.mean() + self.value_coef * value_loss
                
                # Calculate approximate KL divergence
                with torch.no_grad():
                    log_ratio = log_probs - batch_old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    
                    # Early stopping if KL divergence is too high
                    if approx_kl > 1.5 * self.target_kl:
                        break
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Store metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                kls.append(approx_kl)
            
            # Early stopping if KL divergence is too high
            if np.mean(kls) > 1.5 * self.target_kl:
                break
        
        # Calculate explained variance
        with torch.no_grad():
            values = self.ac.get_value(states).squeeze().cpu().numpy()
            explained_var = 1 - np.var(returns.cpu().numpy() - values) / (np.var(returns.cpu().numpy()) + 1e-8)
        
        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies), np.mean(kls), explained_var


def main():
    """Test the PPO agent with a simple environment."""
    from rl_environment.energy_env import EnergyMarketEnv
    
    # Create environment
    env = EnergyMarketEnv(episode_length=24*3)  # 3 days
    
    # Create agent
    agent = PPOAgent(env=env, hidden_dim=128)
    
    # Train agent (short training for testing)
    metrics = agent.train(
        env=env,
        total_timesteps=10000,
        buffer_size=2048,
        batch_size=64,
        update_epochs=10,
        num_eval_episodes=5,
        eval_freq=2000,
        save_freq=5000,
        save_path='models'
    )
    
    # Plot metrics
    agent.plot_metrics(save_path='plots/ppo_metrics.png')
    
    # Evaluate agent
    eval_return = agent.evaluate(env, num_episodes=10)
    print(f"Evaluation return: {eval_return:.2f}")

if __name__ == "__main__":
    main() 