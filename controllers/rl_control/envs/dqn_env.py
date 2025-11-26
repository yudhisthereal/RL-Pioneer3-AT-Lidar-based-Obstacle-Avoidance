import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import json
import socket
from collections import deque
from controller import Robot
import matplotlib.pyplot as plt
import threading

# ============================================================
# Parameters for RL and Robot
# ============================================================
FORWARD_SPEED = 3.0
TURN_SPEED = 2.0
MAX_SPEED = 6.4
AVOID_THRESHOLD = 0.45
GOAL_THRESHOLD = 0.5
MAX_STEPS = 1000
STATE_DIM = 7
ACTION_DIM = 3  # 0 = Turn Right, 1 = Turn Left, 2 = Move Forward

# ============================================================
# DQN Network for RL
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
# DQN Agent Class
# ============================================================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.gamma = 0.99
        self.target_update = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
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
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
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
# Webots Environment Setup
# ============================================================
class DQNEnv:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.wheels = [
            self.robot.getDevice("front left wheel"),
            self.robot.getDevice("back left wheel"),
            self.robot.getDevice("front right wheel"),
            self.robot.getDevice("back right wheel")
        ]
        for wheel in self.wheels:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)
        
        self.lidar = self.robot.getDevice("Sick LMS 291")
        self.lidar.enable(self.timestep)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.plotter_address = ('localhost', 8888)
        
        self.agent = DQNAgent(STATE_DIM, ACTION_DIM)
        self.current_step = 0

        # Real-time LIDAR Plotting (Polar and Cartesian views)
        plt.ion()
        self.fig = plt.figure(figsize=(15, 12))

        # Polar plot - Configure for front at top (180 degrees)
        self.ax1 = self.fig.add_subplot(221, projection='polar')
        self.scatter_polar = self.ax1.scatter([], [], c=[], cmap='RdYlGn_r', s=30, alpha=0.7)
        self.ax1.set_ylim(0, 2.5)
        self.ax1.set_title("LIDAR - Polar View (Front = 180°)", fontsize=14, fontweight='bold')

        # Cartesian plot - Configure for front at top
        self.ax2 = self.fig.add_subplot(222)
        self.scatter_cartesian = self.ax2.scatter([], [], c=[], cmap='RdYlGn_r', s=30, alpha=0.7)
        self.ax2.set_xlim(-2.5, 2.5)
        self.ax2.set_ylim(-2.5, 2.5)
        self.ax2.set_title("LIDAR - Cartesian View (Front = Up)", fontsize=14, fontweight='bold')

        plt.tight_layout()

    def reset(self):
        self.current_step = 0
        self.set_speed(0, 0)
        
        # Reset the robot position - Z is vertical
        start_translation = [0.0, 0.0, 0.1]  # x, y, z (z is vertical)
        rotation = [0, 1, 0, 0]  # Identity rotation
        
        self.robot.getField('translation').setSFVec3f(start_translation)
        self.robot.getField('rotation').setSFRotation(rotation)
        
        return self.get_observation()

    def get_observation(self):
        ranges = self.lidar.getRangeImage()
        return self.get_lidar_sectors(ranges)

    def get_lidar_sectors(self, ranges):
        sectors = np.zeros(7)
        sectors[0] = np.min(ranges[0:30])    # far_left
        sectors[1] = np.min(ranges[30:60])   # left
        sectors[2] = np.min(ranges[60:90])   # front_left
        sectors[3] = np.min(ranges[90:110])  # front
        sectors[4] = np.min(ranges[110:140]) # front_right
        sectors[5] = np.min(ranges[140:170]) # right
        sectors[6] = np.min(ranges[170:])    # far_right
        return sectors

    def apply_action(self, action):
        if action == 0:  # Turn Right
            left_speed = FORWARD_SPEED + TURN_SPEED
            right_speed = FORWARD_SPEED - TURN_SPEED
        elif action == 1:  # Turn Left
            left_speed = FORWARD_SPEED - TURN_SPEED
            right_speed = FORWARD_SPEED + TURN_SPEED
        else:  # Move Forward
            left_speed = FORWARD_SPEED
            right_speed = FORWARD_SPEED

        self.set_speed(left_speed, right_speed)

    def set_speed(self, left, right):
        left = max(min(left, MAX_SPEED), -MAX_SPEED)
        right = max(min(right, MAX_SPEED), -MAX_SPEED)
        self.wheels[0].setVelocity(left)
        self.wheels[1].setVelocity(left)
        self.wheels[2].setVelocity(right)
        self.wheels[3].setVelocity(right)

    def step(self, action):
        current_obs = self.get_observation()
        self.apply_action(action)
        self.robot.step(self.timestep)
        next_obs = self.get_observation()

        reward, done = self.compute_reward(current_obs, action, next_obs)
        self.current_step += 1
        return next_obs, reward, done

    def compute_reward(self, current_obs, action, next_obs):
        reward = -0.1  # Penalty for each step to encourage faster completion

        # Check goal (example: distance to goal)
        distance_to_goal = np.linalg.norm(next_obs[:2])  # Simplified goal check
        if distance_to_goal < GOAL_THRESHOLD:
            reward += 10  # Bonus for reaching goal
        
        # Check for collision
        if next_obs[3] < AVOID_THRESHOLD:  # If front lidar distance is too low
            reward -= 10  # Penalty for crashing

        done = self.current_step >= MAX_STEPS or distance_to_goal < GOAL_THRESHOLD
        return reward, done

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done = self.step(action)
                self.agent.store_transition(state, action, reward, next_state, done)
                self.agent.train()
                state = next_state
                total_reward += reward

            print(f"Episode {episode+1}: Total Reward = {total_reward}")

            # Save model periodically
            if episode % 10 == 0:
                self.agent.save_model(f"dqn_model_episode_{episode}.pth")

        self.agent.save_model("dqn_model_final.pth")
        print("Training complete!")


if __name__ == "__main__":
    # Create the environment
    env = DQNEnv()
    
    # Start the training process for a specified number of episodes
    env.train(num_episodes=1000)