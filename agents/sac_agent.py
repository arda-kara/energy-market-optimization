import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

class ReplayBuffer:
    """
    Replay buffer for storing transitions.
    """
    
    def __init__(self, state_dim, action_dim, max_size=1000000):
        """
        Initialize the replay buffer.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            max_size (int): Maximum size of the buffer
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
    
    def add(self, state, action, next_state, reward, done):
        """
        Add a transition to the buffer.
        
        Args:
            state (np.ndarray): State
            action (np.ndarray): Action
            next_state (np.ndarray): Next state
            reward (float): Reward
            done (bool): Whether the episode is done
        """
        # Ensure state and next_state have the correct shape
        if len(np.shape(state)) == 1:
            state = state.reshape(1, -1)[0]  # Ensure it's a flat array with correct shape
            
        if len(np.shape(next_state)) == 1:
            next_state = next_state.reshape(1, -1)[0]  # Ensure it's a flat array with correct shape
            
        # Store transition
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size (int): Batch size
            
        Returns:
            tuple: (states, actions, next_states, rewards, dones)
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        
        states = self.states[ind]
        actions = self.actions[ind]
        rewards = self.rewards[ind]
        next_states = self.next_states[ind]
        dones = self.dones[ind]
        
        # Ensure rewards has the correct shape (batch_size,)
        if len(rewards.shape) > 1 and rewards.shape[1] > 1:
            rewards = rewards[:, 0]
            
        # Ensure next_states has the correct shape (batch_size, state_dim)
        if len(next_states.shape) == 1:
            # If next_states is 1D, reshape it to match states shape
            next_states = np.reshape(next_states, states.shape)
        
        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )


class Actor(nn.Module):
    """
    Actor network for SAC.
    
    Outputs the mean and log standard deviation of a Gaussian policy.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        """
        Initialize the actor network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Dimension of hidden layers
            log_std_min (float): Minimum log standard deviation
            log_std_max (float): Maximum log standard deviation
        """
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.state_dim = state_dim
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: (mean, log_std)
        """
        # Handle different state shapes
        if state.dim() == 0:  # Scalar tensor
            state = state.unsqueeze(0).unsqueeze(0)  # Add batch and feature dimensions
        elif state.dim() == 1:  # 1D tensor
            # Check if this is a batch of scalars or a single state vector
            if state.shape[0] == self.state_dim:
                # This is a single state vector, add batch dimension
                state = state.unsqueeze(0)
            else:
                # This is likely a batch of scalars, reshape to [batch_size, state_dim]
                # This is a critical error, but we'll try to recover
                print(f"WARNING: Unexpected state shape in Actor.forward: {state.shape}, expected dim {self.state_dim}")
                # Try to reshape if the total elements match
                if state.shape[0] % self.state_dim == 0:
                    batch_size = state.shape[0] // self.state_dim
                    state = state.reshape(batch_size, self.state_dim)
                else:
                    # If we can't reshape, raise an error
                    raise ValueError(f"State shape {state.shape} incompatible with expected state_dim {self.state_dim}")
        
        # Check if the state has the correct shape
        if state.shape[1] != self.state_dim:
            raise ValueError(f"State shape {state.shape} incompatible with expected state_dim {self.state_dim}")
            
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mu, log_std
    
    def sample(self, state):
        """
        Sample an action from the policy.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: (action, log_prob)
        """
        # Handle different state shapes
        if state.dim() == 0:  # Scalar tensor
            state = state.unsqueeze(0).unsqueeze(0)  # Add batch and feature dimensions
        elif state.dim() == 1:  # 1D tensor
            # Check if this is a batch of scalars or a single state vector
            if state.shape[0] == self.state_dim:
                # This is a single state vector, add batch dimension
                state = state.unsqueeze(0)
            else:
                # This is likely a batch of scalars, reshape to [batch_size, state_dim]
                # This is a critical error, but we'll try to recover
                print(f"WARNING: Unexpected state shape in Actor.sample: {state.shape}, expected dim {self.state_dim}")
                # Try to reshape if the total elements match
                if state.shape[0] % self.state_dim == 0:
                    batch_size = state.shape[0] // self.state_dim
                    state = state.reshape(batch_size, self.state_dim)
                else:
                    # If we can't reshape, duplicate the first element to create a valid batch
                    dummy_state = torch.zeros(1, self.state_dim, device=state.device)
                    print(f"ERROR: Cannot reshape state with shape {state.shape} to match state_dim {self.state_dim}")
                    print(f"Using dummy state as fallback")
                    state = dummy_state
        
        # Get mean and log_std from the policy network
        mu, log_std = self(state)
        std = log_std.exp()
        
        # Create normal distribution
        normal = Normal(mu, std)
        
        # Sample action from the distribution
        x_t = normal.rsample()  # Reparameterization trick
        
        # Calculate log probability
        log_prob = normal.log_prob(x_t).sum(dim=1, keepdim=True)
        
        # Apply tanh squashing
        y_t = torch.tanh(x_t)
        
        # Scale action to the action space
        action = y_t
        
        # Compute log probability with the squashing correction
        # log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(dim=1, keepdim=True)
        
        return action, log_prob
    
    def deterministic_action(self, state):
        """
        Get deterministic action from the policy.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            torch.Tensor: Deterministic action
        """
        mu, _ = self(state)
        
        # Apply tanh squashing
        y_t = torch.tanh(mu)
        
        # Scale action to [0, 1]
        action = (y_t + 1) / 2
        
        return action


class Critic(nn.Module):
    """
    Critic network for SAC.
    
    Outputs the Q-value for a state-action pair.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Initialize the critic network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Dimension of hidden layers
        """
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
        # Store dimensions for validation
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def forward(self, state, action):
        """
        Forward pass through the network.
        
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            tuple: (q1, q2)
        """
        # Ensure state and action have batch dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        # Validate dimensions
        if state.shape[1] != self.state_dim:
            raise ValueError(f"State shape {state.shape} incompatible with expected state_dim {self.state_dim}")
        if action.shape[1] != self.action_dim:
            raise ValueError(f"Action shape {action.shape} incompatible with expected action_dim {self.action_dim}")
            
        # Concatenate state and action
        sa = torch.cat([state, action], dim=1)
        
        # Q1
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)
        
        # Q2
        q2 = F.relu(self.fc3(sa))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2(q2)
        
        return q1, q2
    
    def q1_forward(self, state, action):
        """
        Forward pass through the first Q-network only.
        
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            torch.Tensor: Q1 value
        """
        # Ensure state and action have batch dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        # Concatenate state and action
        sa = torch.cat([state, action], dim=1)
        
        # Q1
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)
        
        return q1  # Return only q1, not a tuple


class SACAgent:
    """
    Soft Actor-Critic (SAC) agent.
    
    This implementation uses automatic entropy tuning and twin Q-networks.
    """
    
    def __init__(self, env=None, state_dim=None, action_dim=None, hidden_dim=256, 
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, auto_alpha=True, debug_mode=False):
        """
        Initialize the SAC agent.
        
        Args:
            env (gym.Env, optional): Environment
            state_dim (int, optional): Dimension of the state space
            action_dim (int, optional): Dimension of the action space
            hidden_dim (int): Dimension of hidden layers
            lr (float): Learning rate
            gamma (float): Discount factor
            tau (float): Target network update rate
            alpha (float): Temperature parameter for entropy
            auto_alpha (bool): Whether to automatically tune alpha
            debug_mode (bool): Whether to enable debug logging
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get state and action dimensions from environment if provided
        if env is not None:
            self.state_dim = env.observation_space.shape[0]
            self.action_dim = env.action_space.shape[0]
        else:
            self.state_dim = state_dim
            self.action_dim = action_dim
            
        # Validate state and action dimensions
        if self.state_dim is None or self.action_dim is None:
            raise ValueError("Either env or state_dim and action_dim must be provided")
            
        # Store hyperparameters
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.debug_mode = debug_mode
        
        # Initialize actor network
        self.actor = Actor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Create critic networks
        self.critic1 = Critic(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.critic2 = Critic(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        
        # Create target critic networks
        self.critic1_target = Critic(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.critic2_target = Critic(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        
        # Copy parameters to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # If auto_alpha is enabled, create log_alpha parameter and optimizer
        if auto_alpha:
            # Target entropy is -|A|
            self.target_entropy = -self.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = torch.exp(self.log_alpha).item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # Initialize metrics
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_loss': [],
            'critic_loss': [],
            'alpha_loss': [],
            'alpha': []
        }
    
    def get_action(self, state, deterministic=False):
        """
        Get action from policy.
        
        Args:
            state (np.ndarray): State array
            deterministic (bool): If True, return the mean action
            
        Returns:
            np.ndarray: Action array
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
            if deterministic:
                action = self.actor.deterministic_action(state)
            else:
                action, _ = self.actor.sample(state)
        
        return action.cpu().numpy()[0]
    
    def train(self, env, total_timesteps, batch_size=256, buffer_size=1000000, 
              update_after=1000, update_every=50, num_eval_episodes=10, 
              eval_freq=10000, save_freq=50000, save_path=None, tb_writer=None):
        """
        Train the agent.
        
        Args:
            env (gym.Env): Environment
            total_timesteps (int): Total number of timesteps to train for
            batch_size (int): Batch size for updates
            buffer_size (int): Size of the replay buffer
            update_after (int): Number of timesteps to collect before starting updates
            update_every (int): Number of timesteps between updates
            num_eval_episodes (int): Number of episodes to evaluate for
            eval_freq (int): Frequency of evaluation (in timesteps)
            save_freq (int): Frequency of saving (in timesteps)
            save_path (str, optional): Path to save the agent
            tb_writer (SummaryWriter, optional): TensorBoard writer
            
        Returns:
            dict: Training metrics
        """
        # Create replay buffer
        buffer = ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_size=buffer_size
        )
        
        # Initialize metrics
        metrics = {
            'timesteps': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'critic_losses': [],
            'actor_losses': [],
            'alpha_losses': [],
            'alpha_values': [],
            'eval_returns': [],
            'eval_timesteps': []
        }
        
        # Initialize variables
        state, _ = env.reset()
        episode_return = 0
        episode_length = 0
        episode_num = 0
        
        print(f"Starting training for {total_timesteps} timesteps")
        print(f"Buffer size: {buffer_size}, Batch size: {batch_size}")
        print(f"Update after: {update_after}, Update every: {update_every}")
        print(f"Evaluation frequency: {eval_freq}, Save frequency: {save_freq}")
        
        # Training loop
        for t in range(1, total_timesteps + 1):
            # Select action
            if t < update_after:
                # Random action for exploration
                action = env.action_space.sample()
            else:
                # Select action according to policy
                action = self.get_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition in replay buffer
            buffer.add(state, action, next_state, reward, done)
            
            # Update state, return, and length
            state = next_state
            episode_return += reward
            episode_length += 1
            
            # Update networks
            if t >= update_after and t % update_every == 0:
                for _ in range(update_every):
                    batch = buffer.sample(batch_size)
                    critic_loss, actor_loss, alpha_loss = self._update_networks(batch)
                    
                    # Store losses
                    metrics['critic_losses'].append(critic_loss)
                    metrics['actor_losses'].append(actor_loss)
                    if alpha_loss is not None:
                        metrics['alpha_losses'].append(alpha_loss)
                    metrics['alpha_values'].append(self.alpha)
            
            # End of episode
            if done:
                # Store episode metrics
                metrics['timesteps'].append(t)
                metrics['episode_rewards'].append(episode_return)
                metrics['episode_lengths'].append(episode_length)
                
                # Log episode results
                episode_num += 1
                if episode_num % 10 == 0:  # Log every 10 episodes
                    print(f"Episode {episode_num} | Timestep {t}/{total_timesteps} | Return: {episode_return:.2f} | Length: {episode_length}")
                
                # Reset environment
                state, _ = env.reset()
                episode_return = 0
                episode_length = 0
                
                # TensorBoard logging
                if tb_writer is not None:
                    tb_writer.add_scalar('train/episode_reward', metrics['episode_rewards'][-1], t)
                    tb_writer.add_scalar('train/episode_length', metrics['episode_lengths'][-1], t)
            
            # Evaluate agent
            if t % eval_freq == 0:
                print(f"\nEvaluating at timestep {t}/{total_timesteps}...")
                eval_return = self.evaluate(env, num_episodes=num_eval_episodes)
                print(f"Eval return: {eval_return:.2f}")
                
                # Store evaluation metrics
                metrics['eval_returns'].append(eval_return)
                metrics['eval_timesteps'].append(t)
                
                # TensorBoard logging
                if tb_writer is not None:
                    tb_writer.add_scalar('eval/return', eval_return, t)
            
            # Save agent
            if save_path is not None and t % save_freq == 0:
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                self.save(f"{save_path}_step{t}.pt")
                print(f"Agent saved to {save_path}_step{t}.pt")
                
            # Print progress
            if t % (total_timesteps // 10) == 0:  # Print 10 progress updates
                print(f"Timestep {t}/{total_timesteps} | {t/total_timesteps*100:.1f}% complete")
        
        # Final evaluation
        print("\nFinal evaluation...")
        final_return = self.evaluate(env, num_episodes=num_eval_episodes)
        print(f"Final evaluation return: {final_return:.2f}")
        
        # Save final model
        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save(f"{save_path}_final.pt")
            print(f"Final agent saved to {save_path}_final.pt")
        
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
                action = self.get_action(state, deterministic=True)
                
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
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_alpha else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_alpha else None,
            'metrics': self.metrics
        }, path)
    
    def load(self, path):
        """
        Load the agent.
        
        Args:
            path (str): Path to load the agent from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        if self.auto_alpha and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = torch.exp(self.log_alpha).item()
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.metrics = checkpoint['metrics']
    
    def plot_metrics(self, save_path=None):
        """
        Plot training metrics.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        
        # Plot episode rewards
        axs[0, 0].plot(self.metrics['episode_rewards'])
        axs[0, 0].set_title('Episode Rewards')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Return')
        axs[0, 0].grid(True)
        
        # Plot episode lengths
        axs[0, 1].plot(self.metrics['episode_lengths'])
        axs[0, 1].set_title('Episode Lengths')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Length')
        axs[0, 1].grid(True)
        
        # Plot actor and critic losses
        axs[1, 0].plot(self.metrics['actor_loss'], label='Actor Loss')
        axs[1, 0].plot(self.metrics['critic_loss'], label='Critic Loss')
        axs[1, 0].set_title('Losses')
        axs[1, 0].set_xlabel('Update')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # Plot alpha
        axs[1, 1].plot(self.metrics['alpha'])
        axs[1, 1].set_title('Alpha')
        axs[1, 1].set_xlabel('Update')
        axs[1, 1].set_ylabel('Alpha')
        axs[1, 1].grid(True)
        
        # Plot alpha loss
        axs[2, 0].plot(self.metrics['alpha_loss'])
        axs[2, 0].set_title('Alpha Loss')
        axs[2, 0].set_xlabel('Update')
        axs[2, 0].set_ylabel('Loss')
        axs[2, 0].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()

    def _update_networks(self, batch):
        """
        Update networks using a batch of transitions.
        
        Args:
            batch (tuple): Batch of transitions (state, action, reward, next_state, done)
            
        Returns:
            tuple: (critic_loss, actor_loss, alpha_loss)
        """
        states, actions, rewards, next_states, dones = batch
        
        # Debug shapes before conversion
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"Batch shapes before conversion:")
            print(f"  states: {np.shape(states)}")
            print(f"  actions: {np.shape(actions)}")
            print(f"  rewards: {np.shape(rewards)}")
            print(f"  next_states: {np.shape(next_states)}")
            print(f"  dones: {np.shape(dones)}")
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        
        # Check if rewards is already 2D and reshape if needed
        if len(np.shape(rewards)) > 1:
            # If rewards has shape (batch_size, state_dim), take only the first column
            rewards = rewards[:, 0] if np.shape(rewards)[1] > 1 else rewards.flatten()
        
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        
        # Handle next_states based on its shape
        next_states_shape = np.shape(next_states)
        
        # If next_states is 1D, we need to reshape it
        if len(next_states_shape) == 1:
            batch_size = len(next_states)
            
            # Check if each element in next_states is actually a state vector
            if isinstance(next_states[0], np.ndarray):
                # If elements are arrays, stack them
                next_states = np.stack(next_states)
            else:
                # If elements are scalars, try to reshape based on state_dim
                if batch_size % self.state_dim == 0:
                    # If batch_size is divisible by state_dim, reshape to [batch_size/state_dim, state_dim]
                    next_states = np.reshape(next_states, (-1, self.state_dim))
                else:
                    # This is a critical error - try to recover by using states
                    print(f"ERROR: Cannot reshape next_states with shape {next_states_shape}")
                    print(f"Using states as fallback for next_states")
                    next_states = np.copy(states.cpu().numpy())
        
        # If next_states is already 2D but has wrong second dimension
        elif len(next_states_shape) == 2 and next_states_shape[1] != self.state_dim:
            print(f"WARNING: next_states has shape {next_states_shape} but expected second dim to be {self.state_dim}")
            # Try to reshape if possible
            if next_states_shape[0] * next_states_shape[1] % self.state_dim == 0:
                new_batch_size = (next_states_shape[0] * next_states_shape[1]) // self.state_dim
                next_states = np.reshape(next_states, (new_batch_size, self.state_dim))
            else:
                # Use states as fallback
                print(f"ERROR: Cannot reshape next_states with shape {next_states_shape}")
                print(f"Using states as fallback for next_states")
                next_states = np.copy(states.cpu().numpy())
        
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Debug shapes after conversion
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"Tensor shapes after conversion:")
            print(f"  states: {states.shape}")
            print(f"  actions: {actions.shape}")
            print(f"  rewards: {rewards.shape}")
            print(f"  next_states: {next_states.shape}")
            print(f"  dones: {dones.shape}")
        
        # Update critic
        with torch.no_grad():
            # Sample next actions and log probs from current policy
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Target Q-values
            next_q1_tuple = self.critic1_target(next_states, next_actions)
            next_q2_tuple = self.critic2_target(next_states, next_actions)
            
            # Handle the case where critic returns a tuple or a single tensor
            if isinstance(next_q1_tuple, tuple):
                next_q1, _ = next_q1_tuple  # Unpack tuple, ignore second value
            else:
                next_q1 = next_q1_tuple
                
            if isinstance(next_q2_tuple, tuple):
                next_q2, _ = next_q2_tuple  # Unpack tuple, ignore second value
            else:
                next_q2 = next_q2_tuple
            
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Current Q-values
        current_q1_tuple = self.critic1(states, actions)
        current_q2_tuple = self.critic2(states, actions)
        
        # Handle the case where critic returns a tuple or a single tensor
        if isinstance(current_q1_tuple, tuple):
            current_q1, _ = current_q1_tuple  # Unpack tuple, ignore second value
        else:
            current_q1 = current_q1_tuple
            
        if isinstance(current_q2_tuple, tuple):
            current_q2, _ = current_q2_tuple  # Unpack tuple, ignore second value
        else:
            current_q2 = current_q2_tuple
        
        # Critic loss
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        # Sample actions and log probs from current policy
        new_actions, log_probs = self.actor.sample(states)
        
        # Actor loss
        q1_tuple = self.critic1(states, new_actions)
        q2_tuple = self.critic2(states, new_actions)
        
        # Handle the case where critic returns a tuple or a single tensor
        if isinstance(q1_tuple, tuple):
            q1, _ = q1_tuple  # Unpack tuple, ignore second value
        else:
            q1 = q1_tuple
            
        if isinstance(q2_tuple, tuple):
            q2, _ = q2_tuple  # Unpack tuple, ignore second value
        else:
            q2 = q2_tuple
        
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha if automatic entropy tuning is enabled
        alpha_loss = torch.tensor(0.0).to(self.device)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = torch.exp(self.log_alpha).item()
        
        # Update target networks
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Store metrics
        self.metrics['critic_loss'].append(critic_loss.item())
        self.metrics['actor_loss'].append(actor_loss.item())
        self.metrics['alpha_loss'].append(alpha_loss.item())
        self.metrics['alpha'].append(self.alpha)
        
        return critic_loss.item(), actor_loss.item(), alpha_loss.item()


def main():
    """Test the SAC agent with a simple environment."""
    from rl_environment.energy_env import EnergyMarketEnv
    
    # Create environment
    env = EnergyMarketEnv(episode_length=24*3)  # 3 days
    
    # Create agent
    agent = SACAgent(env=env, hidden_dim=256)
    
    # Train agent (short training for testing)
    metrics = agent.train(
        env=env,
        total_timesteps=10000,
        batch_size=256,
        buffer_size=100000,
        update_after=1000,
        update_every=50,
        num_eval_episodes=5,
        eval_freq=2000,
        save_freq=5000,
        save_path='models'
    )
    
    # Plot metrics
    agent.plot_metrics(save_path='plots/sac_metrics.png')
    
    # Evaluate agent
    eval_return = agent.evaluate(env, num_episodes=10)
    print(f"Evaluation return: {eval_return:.2f}")

if __name__ == "__main__":
    main() 