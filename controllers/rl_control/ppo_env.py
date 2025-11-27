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

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from base_env import BaseEnv
    print("✅ Successfully imported BaseEnv")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(PPONetwork, self).__init__()
        
        # Shared layers with batch normalization
        self.shared_fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # Actor head - untuk output [0, 1] (hanya maju)
        self.actor_mean = nn.Linear(hidden_size, action_size)
        # Start with higher std for more exploration
        self.actor_log_std = nn.Parameter(torch.ones(1, action_size) * -1.0)
        
        # Critic head
        self.critic_value = nn.Linear(hidden_size, 1)
        
        self.activation = nn.ReLU()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, state):
        # Shared features with batch norm
        x = self.shared_fc1(state)
        if x.shape[0] > 1:  # Only use batch norm if batch size > 1
            x = self.bn1(x)
        x = self.activation(x)
        
        x = self.shared_fc2(x)
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = self.activation(x)
        
        # Actor - output sigmoid untuk [0, 1] range (hanya maju)
        # Add bias towards forward motion
        action_mean = torch.sigmoid(self.actor_mean(x)) * 0.9 + 0.1  # Range [0.1, 1.0]
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        action_std = torch.clamp(action_std, 0.05, 0.4)  # Limit std
        
        # Critic
        value = self.critic_value(x)
        
        return action_mean, action_std, value

class PPOAgent:
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # HYPERPARAMETERS - Disesuaikan untuk collision avoidance
        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.ppo_epochs = 10
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.02  # Increased untuk eksplorasi lebih
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.batch_size = 64
        self.update_frequency = 2048
        
        # Network
        self.network = PPONetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, eps=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.95)
        
        # Memory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Statistics
        self.total_steps = 0
        self.update_count = 0
        
        # Use rule-based policy for first N episodes
        self.use_rule_based = True
        self.rule_based_episodes = 50
    
    def get_rule_based_action(self, state):
        """Simple rule-based policy for bootstrapping"""
        # Extract lidar data (first 16 values in first frame)
        lidar = state[:16]
        
        # Find distances
        front = np.min(lidar[6:10])
        left = np.mean(lidar[0:4])
        right = np.mean(lidar[12:16])
        
        # Simple rules
        if front < 0.5:
            # Too close front - turn
            if left > right:
                return np.array([0.2, 0.8])  # Turn left
            else:
                return np.array([0.8, 0.2])  # Turn right
        elif front < 1.0:
            # Moderate distance - slow and steer
            if left > right:
                return np.array([0.4, 0.7])
            else:
                return np.array([0.7, 0.4])
        else:
            # Safe - go forward with slight correction
            balance = left - right
            if abs(balance) < 0.5:
                return np.array([0.6, 0.6])  # Straight
            elif balance > 0:
                return np.array([0.5, 0.7])  # Slight right
            else:
                return np.array([0.7, 0.5])  # Slight left
    
    def get_action(self, state, training=True, episode=0):
        """Get action from policy"""
        # Use rule-based for first episodes
        if training and self.use_rule_based and episode < self.rule_based_episodes:
            action = self.get_rule_based_action(state)
            
            # Still get value for training
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, value = self.network(state_tensor)
            
            # Fake log_prob for consistency
            log_prob = 0.0
            return action, log_prob, value.item()
        
        # Normal PPO policy
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean, action_std, value = self.network(state_tensor)
            
            if training:
                # Sample action during training
                distribution = Normal(action_mean, action_std)
                action = distribution.sample()
                action = torch.clamp(action, 0.0, 1.0)  # Clip ke [0, 1]
                log_prob = distribution.log_prob(action).sum(-1)
            else:
                # Use mean during evaluation
                action = action_mean
                log_prob = None
            
            action = action.squeeze(0).cpu().numpy()
            value = value.squeeze(0).cpu().item()
        
        return action, log_prob.item() if log_prob is not None else None, value
    
    def store_transition(self, state, action, log_prob, value, reward, done):
        """Store transition in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.total_steps += 1
    
    def compute_advantages(self, last_value=0):
        """Compute GAE advantages"""
        advantages = []
        returns = []
        gae = 0
        
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(np.array(advantages)),
            torch.FloatTensor(np.array(returns)),
            torch.FloatTensor(np.array(self.log_probs)),
            torch.FloatTensor(np.array(self.values))
        )
    
    def update(self):
        """Update policy using PPO"""
        if len(self.states) < self.batch_size:
            return {}
        
        # Get last value
        if self.dones[-1]:
            last_value = 0
        else:
            last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, last_value_tensor = self.network(last_state)
                last_value = last_value_tensor.item()
        
        # Compute advantages
        states, actions, advantages, returns, old_log_probs, old_values = self.compute_advantages(last_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        # PPO update epochs
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices].to(self.device)
                batch_actions = actions[batch_indices].to(self.device)
                batch_advantages = advantages[batch_indices].to(self.device)
                batch_returns = returns[batch_indices].to(self.device)
                batch_old_log_probs = old_log_probs[batch_indices].to(self.device)
                
                # Forward pass
                action_means, action_stds, values = self.network(batch_states)
                distribution = Normal(action_means, action_stds)
                
                # New log probs
                new_log_probs = distribution.log_prob(batch_actions).sum(dim=-1)
                entropy = distribution.entropy().mean()
                
                # PPO objective
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                total_loss = (policy_loss + 
                            self.value_coef * value_loss - 
                            self.entropy_coef * entropy)
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
        
        # Update learning rate
        self.scheduler.step()
        self.update_count += 1
        
        # Clear memory
        self._clear_memory()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def _clear_memory(self):
        """Clear memory buffers"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.update_count = checkpoint.get('update_count', 0)

class PPOEnv(BaseEnv):
    def __init__(self, robot, max_steps=1000):
        super().__init__(robot, max_steps)
        
        # Lidar history untuk temporal info
        self.lidar_history = deque(maxlen=3)
        for _ in range(3):
            self.lidar_history.append(np.zeros(self.lidar_rays))
        
        # Episode tracking
        self.episode_count = 0
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_length = 0
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.collision_count = 0
        self.successful_episodes = 0
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔥 PPO Device: {self.device}")
        
        # Create agent
        state_size = self.get_state_size()
        action_size = self.get_action_size()
        self.agent = PPOAgent(state_size, action_size, self.device)
        
        # Curriculum learning - start with easier task
        self.curriculum_stage = 0
        self.collision_threshold = 0.2  # Will increase as training improves
        
    def get_state_size(self):
        """State: 3 lidar frames + velocity + angular_velocity"""
        return self.lidar_rays * 3 + 2
    
    def get_state(self):
        """Get enhanced state with history"""
        current_lidar = super().get_lidar_data()
        self.lidar_history.append(current_lidar)
        
        # Concatenate lidar history
        lidar_state = np.concatenate(list(self.lidar_history))
        
        # Add velocity info (normalized)
        velocity = self.get_robot_velocity()
        angular_velocity = self.get_robot_angular_velocity()
        velocity_state = np.array([
            np.clip(velocity / self.max_velocity, 0, 1),
            np.clip(angular_velocity / 2.0, -1, 1)
        ])
        
        state = np.concatenate([lidar_state, velocity_state])
        return state
    
    def step(self, action):
        """Execute step with PPO logic"""
        # Store previous transition
        if hasattr(self, 'last_state') and hasattr(self, 'last_action'):
            self.agent.store_transition(
                self.last_state,
                self.last_action,
                self.last_log_prob,
                self.last_value,
                self.episode_reward_step,  # Reward dari step sebelumnya
                False  # done akan di-update di reset
            )
        
        # Execute action in environment
        next_state, reward, done, info = super().step(action)
        
        # Get next action - PASS EPISODE NUMBER
        next_action, next_log_prob, next_value = self.agent.get_action(
            next_state, 
            training=True, 
            episode=self.episode_count
        )
        
        # Store for next iteration
        self.last_state = next_state
        self.last_action = next_action
        self.last_log_prob = next_log_prob
        self.last_value = next_value
        self.episode_reward_step = reward
        
        # Update episode stats
        self.episode_reward += reward
        self.episode_length += 1
        self.total_steps += 1
        
        # Update policy when buffer is full
        if len(self.agent.states) >= self.agent.update_frequency:
            stats = self.agent.update()
            if stats and self.episode_count % 10 == 0:
                print(f"🔄 Update #{stats['update_count']} | "
                      f"PL: {stats['policy_loss']:.4f} | "
                      f"VL: {stats['value_loss']:.4f} | "
                      f"Ent: {stats['entropy']:.4f}")
        
        # Episode end handling
        if done:
            # Check if collision
            is_collision = np.min(self.get_lidar_data()) < self.collision_threshold
            
            if is_collision:
                self.collision_count += 1
            else:
                self.successful_episodes += 1
            
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length,
                'collision': is_collision,
                'success': not is_collision
            }
            
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_length)
            
            # Curriculum learning adjustment
            if self.episode_count > 0 and self.episode_count % 100 == 0:
                success_rate = self.successful_episodes / 100
                if success_rate > 0.7 and self.curriculum_stage < 3:
                    self.curriculum_stage += 1
                    self.max_velocity *= 1.1  # Increase difficulty
                    print(f"📈 CURRICULUM UP! Stage {self.curriculum_stage}, Success: {success_rate*100:.1f}%")
                self.successful_episodes = 0
            
            # Print episode summary
            avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else self.episode_reward
            status = "💥 COLLISION" if is_collision else "✅ SUCCESS"
            
            # Show if using rule-based
            policy_type = "🤖 Rule" if self.episode_count < self.agent.rule_based_episodes else "🧠 PPO"
            
            print(f"{policy_type} | Ep {self.episode_count:4d} | "
                  f"R: {self.episode_reward:7.2f} | "
                  f"Len: {self.episode_length:4d} | "
                  f"Avg10: {avg_reward:6.2f} | "
                  f"Collisions: {self.collision_count} | "
                  f"{status}")
        
        return next_state, reward, done, info
    
    def reset(self):
        """Reset environment with IMPROVED safety check"""
        # Store final transition if exists
        if hasattr(self, 'last_state'):
            self.agent.store_transition(
                self.last_state,
                self.last_action,
                self.last_log_prob,
                self.last_value,
                self.episode_reward_step,
                True  # Episode ended
            )
        
        # Reset episode stats
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_reward_step = 0
        
        # Clear lidar history
        self.lidar_history.clear()
        for _ in range(3):
            self.lidar_history.append(np.zeros(self.lidar_rays))
        
        # Call parent reset
        state = super().reset()
        
        # CRITICAL: Wait for simulation to stabilize
        for _ in range(15):
            self.robot.step(self.timestep)
        
        # Check spawn safety with multiple attempts
        max_escape_attempts = 5
        for attempt in range(max_escape_attempts):
            lidar_data = self.get_lidar_data()
            min_spawn_dist = np.min(lidar_data)
            
            if min_spawn_dist > 0.4:
                print(f"✅ Safe spawn! MinDist: {min_spawn_dist:.2f}m")
                break
            
            print(f"⚠️ UNSAFE SPAWN #{attempt+1}! MinDist: {min_spawn_dist:.3f}m")
            
            # Analyze all directions
            front_dist = np.min(lidar_data[6:10]) if len(lidar_data) >= 16 else min_spawn_dist
            left_dist = np.mean(lidar_data[0:4]) if len(lidar_data) >= 16 else min_spawn_dist
            right_dist = np.mean(lidar_data[12:16]) if len(lidar_data) >= 16 else min_spawn_dist
            back_dist = np.min(lidar_data[13:16] + lidar_data[0:3]) if len(lidar_data) >= 16 else min_spawn_dist
            
            print(f"   Distances - F:{front_dist:.2f} L:{left_dist:.2f} R:{right_dist:.2f} B:{back_dist:.2f}")
            
            # Find best escape route
            max_dist = max(front_dist, left_dist, right_dist, back_dist)
            
            if max_dist < 0.4:
                # VERY STUCK - do aggressive continuous spin
                print(f"   🆘 CRITICAL: All sides blocked! Spinning...")
                spin_direction = 1 if left_dist > right_dist else -1
                
                for _ in range(40):  # Longer spin
                    if spin_direction > 0:
                        self.apply_action(np.array([0.1, 0.7]))
                    else:
                        self.apply_action(np.array([0.7, 0.1]))
                    self.robot.step(self.timestep)
                
                # Try moving forward after spin
                for _ in range(15):
                    self.apply_action(np.array([0.4, 0.4]))
                    self.robot.step(self.timestep)
            
            elif max_dist == left_dist:
                print(f"   ↺ Best escape: LEFT")
                # Turn left and move
                for _ in range(25):
                    self.apply_action(np.array([0.2, 0.8]))
                    self.robot.step(self.timestep)
                for _ in range(15):
                    self.apply_action(np.array([0.5, 0.5]))
                    self.robot.step(self.timestep)
            
            elif max_dist == right_dist:
                print(f"   ↻ Best escape: RIGHT")
                # Turn right and move
                for _ in range(25):
                    self.apply_action(np.array([0.8, 0.2]))
                    self.robot.step(self.timestep)
                for _ in range(15):
                    self.apply_action(np.array([0.5, 0.5]))
                    self.robot.step(self.timestep)
            
            elif max_dist == front_dist and front_dist > 0.3:
                print(f"   → Best escape: FORWARD")
                # Move forward slowly
                for _ in range(30):
                    self.apply_action(np.array([0.4, 0.4]))
                    self.robot.step(self.timestep)
            
            else:  # back_dist is max but we can't go backwards
                print(f"   🔄 Rotating to face open space")
                # Do 180 degree turn
                for _ in range(35):
                    self.apply_action(np.array([0.1, 0.8]))
                    self.robot.step(self.timestep)
                for _ in range(20):
                    self.apply_action(np.array([0.4, 0.4]))
                    self.robot.step(self.timestep)
            
            # Wait for movement to complete
            for _ in range(10):
                self.robot.step(self.timestep)
        
        # Final check
        final_lidar = self.get_lidar_data()
        final_min = np.min(final_lidar)
        if final_min < 0.3:
            print(f"⚠️ WARNING: Still unsafe after {max_escape_attempts} attempts! MinDist: {final_min:.2f}m")
            print(f"   Starting episode anyway - good luck! 🍀")
        
        # Final stabilization
        for _ in range(5):
            self.apply_action(np.array([0.0, 0.0]))  # Stop
            self.robot.step(self.timestep)
        
        # Get initial state and action - PASS EPISODE NUMBER
        initial_state = self.get_state()
        initial_action, initial_log_prob, initial_value = self.agent.get_action(
            initial_state,
            training=True,
            episode=self.episode_count
        )
        
        self.last_state = initial_state
        self.last_action = initial_action
        self.last_log_prob = initial_log_prob
        self.last_value = initial_value
        
        self.episode_count += 1
        
        return initial_state
    
    def get_stats(self):
        """Get training statistics"""
        if len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
        else:
            avg_reward = 0
            avg_length = 0
        
        return {
            'episode': self.episode_count,
            'total_steps': self.total_steps,
            'average_reward': avg_reward,
            'average_length': avg_length,
            'collision_count': self.collision_count,
            'collision_rate': self.collision_count / max(self.episode_count, 1)
        }
    
    def save_model(self, filepath):
        """Save model"""
        self.agent.save(filepath)
        print(f"💾 Model saved: {filepath}")
    
    def load_model(self, filepath):
        """Load model"""
        self.agent.load(filepath)
        print(f"💾 Model loaded: {filepath}")

def main():
    """Main training loop"""
    from controller import Robot
    
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    env = PPOEnv(robot, max_steps=1500)
    
    print("=" * 60)
    print("🚀 PPO TRAINING - COLLISION AVOIDANCE")
    print("=" * 60)
    print("✅ Features:")
    print("   - NO REVERSE: Robot only moves forward")
    print("   - Enhanced GUI: Real-time lidar + status monitoring")
    print("   - Smart rewards: Heavy penalty for collisions")
    print("   - Temporal state: 3-frame lidar history")
    print("   - Reduced speed: 4.0 m/s max for better control")
    print("   - SAFETY OVERRIDE: Aggressive collision avoidance")
    print("   - RULE-BASED BOOTSTRAP: First 50 episodes use rules")
    print("   - CURRICULUM LEARNING: Difficulty increases with success")
    print("   - SPAWN SAFETY: Emergency avoidance if spawned near wall")
    print("=" * 60)
    
    state = env.reset()
    
    while robot.step(timestep) != -1:
        state, reward, done, info = env.step(env.last_action)
        
        if done:
            stats = env.get_stats()
            
            # Save best model
            if len(env.episode_rewards) > 10:
                if env.episode_reward > max(env.episode_rewards[:-1]):
                    env.save_model('best_ppo_model.pth')
                    print("⭐ NEW BEST MODEL!")
            
            # Save checkpoint
            if stats['episode'] % 50 == 0:
                env.save_model(f'checkpoint_ep{stats["episode"]}.pth')
            
            state = env.reset()

if __name__ == "__main__":
    main()