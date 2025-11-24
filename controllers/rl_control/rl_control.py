from controller import Supervisor
import math
import numpy as np
import socket
import json
import time
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ============================================================
# RL Parameters
# ============================================================
STATE_DIM = 7  # 7 lidar sectors
ACTION_DIM = 3  # turn_right, turn_left, straight
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

# ============================================================
# Robot Parameters
# ============================================================
FORWARD_SPEED = 3.0
TURN_SPEED = 2.0
MAX_SPEED = 6.4
AVOID_THRESHOLD = 0.45
GOAL_THRESHOLD = 0.5
MAX_STEPS = 1000

# ============================================================
# DQN Network
# ============================================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# ============================================================
# RL Agent
# ============================================================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.steps_done = 0
        
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.action_dim)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (GAMMA * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps_done += 1
        
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']

# ============================================================
# Webots RL Controller following BaseEnv convention
# ============================================================
class Pioneer3ATScenario:
    """Scenario class following BaseEnv convention"""
    
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.timestep = int(supervisor.getBasicTimeStep())
        self.scenario_name = "Pioneer3ATNavigation"
        
        # Setup robot components
        self.setup_robot()
        
        # Positions
        self.start_pose = self.get_node_position('start')
        self.goal_pose = self.get_node_position('finish')
        
        print(f"Scenario initialized: {self.scenario_name}")
        print(f"Start position: {self.start_pose}")
        print(f"Goal position: {self.goal_pose}")
    
    def setup_robot(self):
        # Motors
        self.wheels = [
            self.supervisor.getDevice("front left wheel"),
            self.supervisor.getDevice("back left wheel"),
            self.supervisor.getDevice("front right wheel"),
            self.supervisor.getDevice("back right wheel")
        ]
        
        for w in self.wheels:
            w.setPosition(float('inf'))
            w.setVelocity(0)
        
        # Lidar
        self.lidar = self.supervisor.getDevice("Sick LMS 291")
        self.lidar.enable(self.timestep)
        
        # Robot node for position control
        self.robot_node = self.supervisor.getFromDef("Pioneer_3AT")
    
    def get_obs_dim(self):
        """Get observation dimension"""
        return 7  # 7 lidar sectors
    
    def get_act_dim(self):
        """Get action dimension"""
        return 3  # discrete actions: turn_right, turn_left, straight
    
    def get_node_position(self, def_name):
        """Get node position (x, y coordinates in Webots - Z is vertical)"""
        node = self.supervisor.getFromDef(def_name)
        if node:
            pos = node.getPosition()
            return [pos[0], pos[1]]  # x, y coordinates (z is vertical)
        return [0, 0]
    
    def get_observation(self):
        """Get current observation"""
        ranges = self.lidar.getRangeImage()
        return self.get_lidar_sectors(ranges)
    
    def get_lidar_sectors(self, ranges):
        """Get 7 lidar sectors as state observation"""
        if len(ranges) == 0:
            return np.zeros(7)
        
        sectors = np.zeros(7)
        sectors[0] = np.min(ranges[0:30])    # far_left
        sectors[1] = np.min(ranges[30:60])   # left
        sectors[2] = np.min(ranges[60:90])   # front_left
        sectors[3] = np.min(ranges[90:110])  # front
        sectors[4] = np.min(ranges[110:140]) # front_right
        sectors[5] = np.min(ranges[140:170]) # right
        sectors[6] = np.min(ranges[170:])    # far_right
        
        return sectors
    
    def get_robot_position(self):
        """Get robot position (x, y coordinates - Z is vertical)"""
        pos = self.robot_node.getPosition()
        return pos[0], pos[1]  # x, y coordinates (z is vertical)
    
    def get_distance_to_goal(self):
        """Calculate distance to goal"""
        robot_pos = self.get_robot_position()
        return math.sqrt((robot_pos[0] - self.goal_pose[0])**2 + 
                        (robot_pos[1] - self.goal_pose[1])**2)
    
    def reset(self):
        """Reset the scenario"""
        # Reset robot to start position
        start_translation = [self.start_pose[0], self.start_pose[1], 0.1]  # x, y, z (z is vertical)
        rotation = [0, 1, 0, 0]  # Identity rotation
        
        self.robot_node.getField('translation').setSFVec3f(start_translation)
        self.robot_node.getField('rotation').setSFRotation(rotation)
        self.robot_node.resetPhysics()
        
        # Reset speeds
        self.set_speed(0, 0)
        
        # Step simulation to apply reset
        self.supervisor.step(self.timestep)
        
        return self.get_observation()
    
    def apply_action(self, action):
        """Apply discrete action to the robot"""
        # Action mapping: 0=turn_right, 1=turn_left, 2=straight
        if action == 0:  # turn_right
            left_speed = FORWARD_SPEED + TURN_SPEED
            right_speed = FORWARD_SPEED - TURN_SPEED
        elif action == 1:  # turn_left
            left_speed = FORWARD_SPEED - TURN_SPEED
            right_speed = FORWARD_SPEED + TURN_SPEED
        else:  # straight
            left_speed = FORWARD_SPEED
            right_speed = FORWARD_SPEED
        
        self.set_speed(left_speed, right_speed)
    
    def set_speed(self, left, right):
        """Set wheel speeds with clamping"""
        left = max(min(left, MAX_SPEED), -MAX_SPEED)
        right = max(min(right, MAX_SPEED), -MAX_SPEED)
        self.wheels[0].setVelocity(left)
        self.wheels[1].setVelocity(left)
        self.wheels[2].setVelocity(right)
        self.wheels[3].setVelocity(right)
    
    def step(self):
        """Step the simulation"""
        self.supervisor.step(self.timestep)
    
    def compute_reward(self, current_obs, action, next_obs, step_count):
        """Compute reward for the transition"""
        reward = 0
        
        # Base reward for surviving
        reward += 0.1
        
        # Distance to goal reward
        distance_to_goal = self.get_distance_to_goal()
        reward += (1.0 / (distance_to_goal + 1.0)) * 0.5
        
        # Safety reward for keeping safe distances
        front_dist = next_obs[3]  # front sector
        front_left_dist = next_obs[2]  # front_left sector
        front_right_dist = next_obs[4]  # front_right sector
        
        min_front_dist = min(front_dist, front_left_dist, front_right_dist)
        if min_front_dist < AVOID_THRESHOLD:
            reward -= 0.5
        elif min_front_dist > AVOID_THRESHOLD * 2:
            reward += 0.2
        
        # Check terminal conditions
        done = False
        termination_reason = ""
        
        # Check if reached goal
        if distance_to_goal < GOAL_THRESHOLD:
            done = True
            termination_reason = "Reached goal"
            reward += 100.0
            # Time bonus (faster is better)
            time_bonus = max(0, 50 - step_count * 0.1)
            reward += time_bonus
        
        # Check if crashed (front sectors too close)
        elif (next_obs[2] < AVOID_THRESHOLD * 0.3 or 
              next_obs[3] < AVOID_THRESHOLD * 0.3 or 
              next_obs[4] < AVOID_THRESHOLD * 0.3):
            done = True
            termination_reason = "Crashed into obstacle"
            reward -= 50.0
        
        # Check if max steps exceeded
        elif step_count >= MAX_STEPS:
            done = True
            termination_reason = "Max steps reached"
            reward -= 10.0
        
        return reward, done, termination_reason
    
    def check_self_collision(self):
        """Check for self-collision (not used in this scenario)"""
        return False, ""
    
    def get_episode_metric(self, observation):
        """Get episode metric for logging (distance to goal)"""
        return self.get_distance_to_goal()
    
    def is_success(self, observation, done):
        """Check if episode was successful"""
        if not done:
            return False
        return self.get_distance_to_goal() < GOAL_THRESHOLD


class Pioneer3ATEnv:
    """Environment class following BaseEnv convention"""
    
    def __init__(self):
        self.supervisor = Supervisor()
        self.scenario = Pioneer3ATScenario(self.supervisor)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.total_episodes = 0
        self.current_step = 0
        self.episode_reward = 0.0
        
        # RL Agent
        self.agent = DQNAgent(STATE_DIM, ACTION_DIM)
        
        # Socket for plotting
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.plotter_address = ('localhost', 8888)
        
        print("Pioneer3AT Environment initialized")
    
    def reset(self):
        """Reset the environment"""
        self.current_step = 0
        self.episode_reward = 0.0
        observation = self.scenario.reset()
        self.total_episodes += 1
        return observation
    
    def step(self, action):
        """Execute one time step"""
        # Get current observation
        current_obs = self.scenario.get_observation()
        
        # Apply action
        self.scenario.apply_action(action)
        
        # Step simulation
        self.scenario.step()
        
        # Get next observation
        next_obs = self.scenario.get_observation()
        
        # Compute reward and done flag
        reward, done, termination_reason = self.scenario.compute_reward(
            current_obs, action, next_obs, self.current_step
        )
        
        # Update tracking
        self.episode_reward += reward
        self.current_step += 1
        
        # Prepare info dictionary
        info = {
            'episode': {
                'r': self.episode_reward,
                'l': self.current_step,
            },
            'termination_reason': termination_reason,
            'metric': self.scenario.get_episode_metric(next_obs),
            'success': self.scenario.is_success(next_obs, done) if done else False
        }
        
        return next_obs, reward, done, info
    
    def get_episode_stats(self):
        """Get episode statistics"""
        return {
            'episode_number': self.total_episodes,
            'episode_steps': self.current_step,
            'episode_reward': self.episode_reward,
            'scenario_name': self.scenario.scenario_name
        }
    
    def send_training_data(self, episode, total_reward, steps, success):
        """Send training progress data to plotter"""
        data = {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'success': success,
            'epsilon': self.agent.epsilon,
            'success_rate': self.success_count / max(1, self.total_episodes)
        }
        
        try:
            message = json.dumps(data).encode('utf-8')
            self.sock.sendto(message, self.plotter_address)
        except:
            pass
    
    def train(self, num_episodes=1000):
        """Train the RL agent"""
        print("Starting RL training...")
        
        for episode in range(num_episodes):
            state = self.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and self.supervisor.step(self.scenario.timestep) != -1:
                # Select and execute action
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.step(action)
                
                # Store experience and train
                self.agent.store_transition(state, action, reward, next_state, done)
                self.agent.train()
                
                state = next_state
                steps += 1
                total_reward += reward
            
            # Update metrics
            success = info.get('success', False)
            if success:
                self.success_count += 1
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            self.agent.update_epsilon()
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else total_reward
                success_rate = self.success_count / max(1, self.total_episodes)
                print(f"Episode {episode}, Reward: {total_reward:.2f}, Steps: {steps}, "
                      f"Success: {success}, Epsilon: {self.agent.epsilon:.3f}, "
                      f"Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.3f}")
            
            # Send training data to plotter
            self.send_training_data(episode, total_reward, steps, success)
            
            # Save model periodically
            if episode % 100 == 0:
                self.agent.save_model(f"dqn_model_episode_{episode}.pth")
        
        # Save final model
        self.agent.save_model("dqn_model_final.pth")
        print("Training completed!")
    
    def run_trained(self, model_path):
        """Run with trained model"""
        self.agent.load_model(model_path)
        self.agent.epsilon = 0.0  # Always use greedy policy
        
        state = self.reset()
        steps = 0
        done = False
        
        while not done and self.supervisor.step(self.scenario.timestep) != -1:
            action = self.agent.select_action(state)
            state, reward, done, info = self.step(action)
            steps += 1
            
            # Print progress
            distance_to_goal = self.scenario.get_distance_to_goal()
            print(f"Step: {steps}, Distance to goal: {distance_to_goal:.2f}")
            
            if steps > MAX_STEPS:
                break
        
        print("Run completed!")


# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    env = Pioneer3ATEnv()
    
    # Choose mode: train or run with trained model
    mode = "train"  # Change to "run" to use trained model
    
    if mode == "train":
        env.train(num_episodes=1000)
    else:
        env.run_trained("dqn_model_final.pth")