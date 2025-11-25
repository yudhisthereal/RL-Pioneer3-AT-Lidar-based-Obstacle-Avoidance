import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import sys
import os
from collections import deque

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from base_env import BaseEnv
    print("Successfully imported BaseEnv")
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback - define minimal BaseEnv if not found
    class BaseEnv:
        def __init__(self, robot, max_steps=1000):
            self.robot = robot
            self.max_steps = max_steps
            self.lidar_rays = 16  # Default value
        
        def get_lidar_data(self):
            return np.random.random(self.lidar_rays)  # Dummy data
        
        def get_robot_velocity(self):
            return 0.0
        
        def get_robot_angular_velocity(self):
            return 0.0
        
        def get_robot_position(self):
            return np.array([0, 0, 0])
        
        def apply_action(self, action):
            pass
        
        def is_done(self):
            return False
        
        def reset(self):
            pass
        
        def compute_reward(self, action):
            return 0.0

class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(PPONetwork, self).__init__()
        
        # Actor network (policy)
        self.actor_fc1 = nn.Linear(state_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_size))
        
        # Critic network (value function)
        self.critic_fc1 = nn.Linear(state_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size)
        self.critic_value = nn.Linear(hidden_size, 1)
        
        self.activation = nn.ReLU()
        
    def forward(self, state):
        return self.actor(state), self.critic(state)
    
    def actor(self, state):
        x = self.activation(self.actor_fc1(state))
        x = self.activation(self.actor_fc2(x))
        mean = torch.tanh(self.actor_mean(x))
        return mean
    
    def critic(self, state):
        x = self.activation(self.critic_fc1(state))
        x = self.activation(self.critic_fc2(x))
        value = self.critic_value(x)
        return value

class PPO:
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparameters
        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.ppo_epochs = 10
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.batch_size = 64
        
        # Networks
        self.network = PPONetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
        # Memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def get_action(self, state, training=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean = self.network.actor(state_tensor)
            action_std = torch.exp(self.network.actor_log_std)
            
            if training:
                distribution = Normal(action_mean, action_std)
                action = distribution.sample()
                log_prob = distribution.log_prob(action).sum(-1)
                action = action.squeeze(0).cpu().numpy()
            else:
                action = action_mean.squeeze(0).cpu().numpy()
                log_prob = None
            
            value = self.network.critic(state_tensor)
        
        return action, log_prob.item() if log_prob is not None else None, value.item()
    
    def store_transition(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_advantages(self, last_value=0):
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])
        
        # Compute returns and advantages
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(advantages),
            torch.FloatTensor(returns),
            torch.FloatTensor(np.array(self.log_probs))
        )
    
    def update(self, next_state=None):
        if len(self.states) < self.batch_size:
            return
        
        if next_state is not None:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                last_value = self.network.critic(next_state_tensor).item()
        else:
            last_value = 0
        
        # Compute advantages and returns
        states, actions, advantages, returns, old_log_probs = self.compute_advantages(last_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.ppo_epochs):
            # Get new policy
            action_means = self.network.actor(states)
            action_stds = torch.exp(self.network.actor_log_std).expand_as(action_means)
            distribution = Normal(action_means, action_stds)
            
            new_log_probs = distribution.log_prob(actions).sum(dim=-1)
            entropy = distribution.entropy().mean()
            
            # Get new values
            values = self.network.critic(states).squeeze()
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Policy loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # Clear memory
        self.clear_memory()
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def save_model(self, filepath):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class PPOEnv(BaseEnv):
    def __init__(self, robot, max_steps=1000):
        super().__init__(robot, max_steps)
        
        # PPO-specific parameters
        self.lidar_hist_len = 3
        self.lidar_history = deque(maxlen=self.lidar_hist_len)
        
        # Initialize lidar history
        for _ in range(self.lidar_hist_len):
            self.lidar_history.append(np.zeros(self.lidar_rays))
        
        # Training variables
        self.episode = 0
        self.max_episodes = 10000
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_steps = 0
        
        # Initialize PPO agent
        state_size = self.get_state_size()
        action_size = self.get_action_size()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.agent = PPO(state_size, action_size, device=self.device)
        
        # Logging
        self.episode_rewards = []
        self.episode_lengths = []
        
    def get_state(self):
        """Get state for PPO algorithm"""
        # Get current lidar data
        current_lidar = super().get_lidar_data()
        
        # Update history
        self.lidar_history.append(current_lidar)
        
        # Create state with history
        state = np.concatenate(list(self.lidar_history))
        
        # Add robot velocity information
        velocity_state = np.array([
            self.get_robot_velocity(),
            self.get_robot_angular_velocity()
        ])
        
        state = np.concatenate([state, velocity_state])
        return state
    
    def compute_reward(self, action):
        """Compute reward for PPO"""
        base_reward = super().compute_reward(action)
        
        # PPO-specific reward components
        progress_reward = self.get_progress_reward()
        safety_reward = self.get_safety_reward()
        action_penalty = self.get_action_penalty(action)
        
        total_reward = (
            base_reward +
            progress_reward +
            safety_reward +
            action_penalty
        )
        
        return total_reward
    
    def get_progress_reward(self):
        """Reward for making progress"""
        if hasattr(self, 'last_position'):
            current_pos = self.get_robot_position()
            distance_traveled = np.linalg.norm(current_pos - self.last_position)
            self.last_position = current_pos
            return distance_traveled * 10.0
        else:
            self.last_position = self.get_robot_position()
            return 0.0
    
    def get_safety_reward(self):
        """Reward for maintaining safe distance from obstacles"""
        lidar_data = self.get_lidar_data()
        min_distance = np.min(lidar_data)
        
        if min_distance < 0.3:  # Too close to obstacle
            return -5.0
        elif min_distance < 0.5:  # Warning zone
            return -1.0
        else:  # Safe zone
            return 0.5
    
    def get_action_penalty(self, action):
        """Penalty for erratic actions"""
        if hasattr(self, 'last_action'):
            action_diff = np.linalg.norm(action - self.last_action)
            self.last_action = action
            return -0.1 * action_diff  # Small penalty for large action changes
        else:
            self.last_action = action
            return 0.0
    
    def get_state_size(self):
        """Get state size for PPO"""
        lidar_size = self.lidar_rays * self.lidar_hist_len
        velocity_size = 2  # linear and angular velocity
        return lidar_size + velocity_size
    
    def get_action_size(self):
        """Get action size for PPO"""
        return 2  # [left_motor_speed, right_motor_speed]
    
    def step(self, action):
        """Execute one step with training"""
        # Apply action
        self.apply_action(action)
        
        # Get next state
        next_state = self.get_state()
        
        # Compute reward
        reward = self.compute_reward(action)
        
        # Check if episode is done
        done = self.is_done()
        
        # Store experience for training
        if hasattr(self, 'current_state') and hasattr(self, 'last_log_prob') and hasattr(self, 'last_value'):
            self.agent.store_transition(
                self.current_state, 
                self.last_action, 
                self.last_log_prob, 
                self.last_value, 
                reward, 
                done
            )
        
        # Update training variables
        self.episode_reward += reward
        self.episode_steps += 1
        self.total_steps += 1
        
        # Get action for next step
        next_action, next_log_prob, next_value = self.agent.get_action(next_state)
        
        # Store for next step
        self.current_state = next_state
        self.last_action = next_action
        self.last_log_prob = next_log_prob
        self.last_value = next_value
        
        # Update PPO agent periodically
        if self.total_steps % 2048 == 0:  # Update every 2048 steps
            self.agent.update(next_state if not done else None)
        
        return next_state, reward, done, {}
    
    def reset(self):
        """Reset environment for new episode"""
        super().reset()
        
        # Reset training variables
        self.episode_steps = 0
        self.episode_reward = 0
        
        # Reset lidar history
        self.lidar_history.clear()
        for _ in range(self.lidar_hist_len):
            self.lidar_history.append(np.zeros(self.lidar_rays))
        
        # Get initial state
        initial_state = self.get_state()
        
        # Get initial action
        initial_action, initial_log_prob, initial_value = self.agent.get_action(initial_state)
        
        # Store initial values
        self.current_state = initial_state
        self.last_action = initial_action
        self.last_log_prob = initial_log_prob
        self.last_value = initial_value
        
        # Log episode results
        if self.episode > 0:
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_steps)
            
            # Print training progress
            if self.episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"Episode {self.episode}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
        
        self.episode += 1
        
        return initial_state
    
    def run_training_episode(self):
        """Run one complete training episode"""
        state = self.reset()
        done = False
        
        while not done:
            state, reward, done, _ = self.step(self.last_action)
        
        # Final update at end of episode
        self.agent.update()
        
        return self.episode_reward, self.episode_steps
    
    def save_model(self, filepath):
        """Save trained model"""
        self.agent.save_model(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        self.agent.load_model(filepath)
        print(f"Model loaded from {filepath}")

if __name__ == "__main__":
    print("PPO Environment module loaded successfully!")
    print("This file should be used as a Webots controller, not run directly.")