# base_env.py
# Base environment class for RL training

import numpy as np
import gym
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any


class BaseEnv(gym.Env, ABC):
    """
    Base environment class for RL training with Webots.
    Compatible with both PPO and DQN algorithms.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, scenario, max_episode_steps=1000):
        """
        Initialize the RL environment.
        
        Args:
            scenario: BaseScenario instance
            max_episode_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.scenario = scenario
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.episode_count = 0
        self.total_steps = 0
        
        # Initialize observation and action spaces
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_metrics = {}
        
    def _get_observation_space(self) -> gym.Space:
        """
        Define the observation space based on scenario dimensions.
        """
        obs_dim = self.scenario.get_obs_dim()
        # Assuming continuous observations, modify if discrete needed
        return gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
    
    def _get_action_space(self) -> gym.Space:
        """
        Define the action space based on scenario dimensions.
        Override this method for discrete action spaces.
        """
        act_dim = self.scenario.get_act_dim()
        # Default to continuous action space, override for discrete
        return gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(act_dim,), 
            dtype=np.float32
        )
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode.
        
        Returns:
            observation: Initial observation
        """
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_metrics = {}
        
        # Reset the scenario
        observation = self.scenario.reset()
        
        # Update episode count
        self.episode_count += 1
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one time step in the environment.
        
        Args:
            action: Action to apply
            
        Returns:
            observation: Next observation
            reward: Reward received
            done: Whether episode is done
            info: Additional information
        """
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
        
        # Check for self-collision (additional termination condition)
        collision_detected, collision_info = self.scenario.check_self_collision()
        if collision_detected:
            done = True
            termination_reason = f"Self-collision: {collision_info}"
            # Apply penalty for collision if needed
            reward -= 10.0
        
        # Check episode length limit
        if self.current_step >= self.max_episode_steps - 1:
            done = True
            if not termination_reason:
                termination_reason = "Max episode steps reached"
        
        # Update tracking variables
        self.episode_reward += reward
        self.current_step += 1
        self.total_steps += 1
        
        # Get episode metric for logging
        metric_value = self.scenario.get_episode_metric(next_obs)
        
        # Prepare info dictionary
        info = {
            'episode': {
                'r': self.episode_reward,
                'l': self.current_step,
                't': round(self.total_steps * self.scenario.timestep / 1000.0, 2)  # Time in seconds
            },
            'termination_reason': termination_reason,
            'metric': metric_value,
            'success': self.scenario.is_success(next_obs, done) if done else False
        }
        
        return next_obs, reward, done, info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        In Webots, rendering is handled by the simulator itself.
        
        Args:
            mode: Rendering mode
            
        Returns:
            None for Webots (rendering is handled by simulator)
        """
        if mode == 'human':
            # Webots handles rendering, so we don't need to do anything
            return None
        else:
            super().render(mode=mode)
    
    def close(self) -> None:
        """
        Clean up environment resources.
        """
        # Webots cleanup if needed
        pass
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the current episode.
        
        Returns:
            Dictionary with episode statistics
        """
        return {
            'episode_number': self.episode_count,
            'episode_steps': self.current_step,
            'episode_reward': self.episode_reward,
            'total_steps': self.total_steps,
            'scenario_name': self.scenario.scenario_name
        }
    
    def seed(self, seed: Optional[int] = None) -> list:
        """
        Set the seed for this env's random number generator.
        
        Args:
            seed: Random seed
            
        Returns:
            List of seeds used
        """
        if seed is not None:
            np.random.seed(seed)
        return [seed]


class DiscreteActionEnv(BaseEnv):
    """
    Base environment with discrete action space for DQN.
    """
    
    def __init__(self, scenario, max_episode_steps=1000, num_actions: int = None):
        """
        Initialize discrete action environment.
        
        Args:
            scenario: BaseScenario instance
            max_episode_steps: Maximum steps per episode
            num_actions: Number of discrete actions (optional, can be inferred)
        """
        super().__init__(scenario, max_episode_steps)
        self.num_actions = num_actions or self.scenario.get_act_dim()
    
    def _get_action_space(self) -> gym.Space:
        """
        Define discrete action space for DQN.
        """
        return gym.spaces.Discrete(self.num_actions)
    
    def apply_discrete_action(self, action: int) -> np.ndarray:
        """
        Convert discrete action to continuous action for the scenario.
        Override this method based on your specific action mapping.
        
        Args:
            action: Discrete action index
            
        Returns:
            continuous_action: Continuous action array
        """
        # Default mapping: evenly spaced actions in [-1, 1] range
        continuous_action = np.array([(action / (self.num_actions - 1)) * 2 - 1])
        return continuous_action
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one time step with discrete action.
        
        Args:
            action: Discrete action index
            
        Returns:
            observation: Next observation
            reward: Reward received
            done: Whether episode is done
            info: Additional information
        """
        # Convert discrete action to continuous
        continuous_action = self.apply_discrete_action(action)
        
        # Use parent step method with continuous action
        return super().step(continuous_action)


class ContinuousActionEnv(BaseEnv):
    """
    Base environment with continuous action space for PPO.
    This is the default BaseEnv behavior, but provided for clarity.
    """
    
    def _get_action_space(self) -> gym.Space:
        """
        Define continuous action space for PPO.
        """
        act_dim = self.scenario.get_act_dim()
        return gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(act_dim,), 
            dtype=np.float32
        )