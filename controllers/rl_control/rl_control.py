import numpy as np
import torch
import sys
import os
import time
from collections import deque

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from ppo_env import PPOEnv
    print("✅ Successfully imported PPOEnv")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class RLControl:
    def __init__(self, robot, max_episodes=3000, max_steps=1500):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        
        # Training parameters
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.current_episode = 0
        
        print("=" * 60)
        print("🤖 INITIALIZING RL CONTROL")
        print("=" * 60)
        
        try:
            # Create PPO environment
            self.env = PPOEnv(robot, max_steps)
            print("✅ PPO Environment created successfully")
            print(f"📊 State size: {self.env.get_state_size()}")
            print(f"🎮 Action size: {self.env.get_action_size()}")
        except Exception as e:
            print(f"❌ Environment creation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -np.inf
        self.best_avg_reward = -np.inf
        
        # Training mode (always train for now)
        self.training_mode = True
        
        print(f"🏋️ Mode: {'TRAINING' if self.training_mode else 'EVALUATION'}")
        print("=" * 60)
    
    def train(self):
        """Main training loop"""
        print("\n🚀 STARTING TRAINING")
        print(f"🎯 Target: {self.max_episodes} episodes")
        print(f"⏱️ Max steps per episode: {self.max_steps}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Initialize environment
            state = self.env.reset()
            print("✅ Environment reset - Ready to train!\n")
            
            episode = 0
            
            while episode < self.max_episodes and self.robot.step(self.timestep) != -1:
                episode_reward = 0
                episode_length = 0
                done = False
                
                # Episode start
                if episode % 10 == 0 or episode < 5:
                    print(f"\n{'='*60}")
                    print(f"▶️ EPISODE {episode + 1}/{self.max_episodes}")
                    print(f"{'='*60}")
                
                # Run episode
                step_count = 0
                while not done and step_count < self.max_steps:
                    # Get action from agent
                    action = self.env.last_action
                    
                    # Take step
                    state, reward, done, info = self.env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    step_count += 1
                    
                    # Debug info for first episodes
                    if episode < 3 and step_count % 200 == 0:
                        lidar_data = self.env.get_lidar_data()
                        min_dist = np.min(lidar_data)
                        velocity = self.env.get_robot_velocity()
                        print(f"   Step {step_count:4d} | "
                              f"R: {reward:6.2f} | "
                              f"V: {velocity:4.2f} | "
                              f"MinD: {min_dist:.2f} | "
                              f"Act: [{action[0]:.2f}, {action[1]:.2f}]")
                    
                    if done:
                        break
                
                # Store episode results
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode += 1
                self.current_episode = episode
                
                # Calculate statistics
                avg_reward_10 = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else episode_reward
                avg_reward_50 = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else episode_reward
                avg_length = np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else episode_length
                
                # Get collision info
                collision_status = "💥 COLLISION" if np.min(self.env.get_lidar_data()) < 0.2 else "✅ Safe"
                
                # Print episode summary
                print(f"\n📊 Episode {episode:4d} Summary:")
                print(f"   Reward:        {episode_reward:8.2f}")
                print(f"   Length:        {episode_length:5d} steps")
                print(f"   Avg Reward10:  {avg_reward_10:8.2f}")
                print(f"   Avg Reward50:  {avg_reward_50:8.2f}")
                print(f"   Best Reward:   {self.best_reward:8.2f}")
                print(f"   Status:        {collision_status}")
                
                # Save best model based on single episode
                if episode_reward > self.best_reward and episode > 10:
                    self.best_reward = episode_reward
                    self.env.save_model('best_single_ppo.pth')
                    print(f"   ⭐ NEW BEST SINGLE EPISODE!")
                
                # Save best model based on average
                if avg_reward_10 > self.best_avg_reward and episode > 20:
                    self.best_avg_reward = avg_reward_10
                    self.env.save_model('best_avg_ppo.pth')
                    print(f"   ⭐ NEW BEST AVERAGE!")
                
                # Save checkpoint
                if episode % 100 == 0:
                    self.env.save_model(f'checkpoint_ep{episode}.pth')
                    elapsed = time.time() - start_time
                    print(f"\n💾 CHECKPOINT {episode}")
                    print(f"   Time elapsed: {elapsed/60:.1f} min")
                    print(f"   Avg time/ep: {elapsed/episode:.2f} sec")
                    print(f"   Est. remaining: {(elapsed/episode)*(self.max_episodes-episode)/60:.1f} min")
                
                # Detailed stats every 10 episodes
                if episode % 10 == 0 and episode > 0:
                    stats = self.env.get_stats()
                    print(f"\n📈 TRAINING STATS (Episode {episode}):")
                    print(f"   Total steps:      {stats['total_steps']}")
                    print(f"   Collision count:  {stats['collision_count']}")
                    print(f"   Collision rate:   {stats['collision_rate']*100:.1f}%")
                    print(f"   Avg episode len:  {stats['average_length']:.1f}")
                    print(f"   Avg reward:       {stats['average_reward']:.2f}")
                
                # Reset for next episode
                state = self.env.reset()
            
            # Training completed
            total_time = time.time() - start_time
            print("\n" + "=" * 60)
            print("🎉 TRAINING COMPLETED!")
            print("=" * 60)
            print(f"Total episodes:  {episode}")
            print(f"Total time:      {total_time/60:.1f} minutes")
            print(f"Best reward:     {self.best_reward:.2f}")
            print(f"Best avg reward: {self.best_avg_reward:.2f}")
            print(f"Final avg (10):  {avg_reward_10:.2f}")
            
            # Save final model
            self.env.save_model('final_ppo_model.pth')
            print("💾 Final model saved!")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
            self.env.save_model('interrupted_model.pth')
            print("💾 Progress saved!")
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
    
    def evaluate(self, model_path=None):
        """Evaluate trained model"""
        print("\n🔍 STARTING EVALUATION")
        print("=" * 60)
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.env.load_model(model_path)
            print(f"✅ Loaded model: {model_path}")
        else:
            # Try to find best model
            model_files = ['best_avg_ppo.pth', 'best_single_ppo.pth', 'final_ppo_model.pth']
            loaded = False
            for model_file in model_files:
                filepath = os.path.join(os.path.dirname(__file__), model_file)
                if os.path.exists(filepath):
                    self.env.load_model(filepath)
                    print(f"✅ Loaded model: {model_file}")
                    loaded = True
                    break
            
            if not loaded:
                print("❌ No trained model found!")
                print("   Please train first or specify model path")
                return
        
        # Run evaluation episodes
        eval_episodes = 10
        eval_rewards = []
        eval_lengths = []
        eval_collisions = 0
        
        print(f"🎯 Running {eval_episodes} evaluation episodes...")
        print("=" * 60)
        
        try:
            for episode in range(eval_episodes):
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                
                print(f"\n▶️ Evaluation Episode {episode + 1}/{eval_episodes}")
                
                while not done and episode_length < self.max_steps:
                    # Get action (no exploration)
                    action, _, _ = self.env.agent.get_action(state, training=False)
                    
                    state, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    if done:
                        break
                
                # Check collision
                if np.min(self.env.get_lidar_data()) < 0.2:
                    eval_collisions += 1
                    collision_str = "💥 COLLISION"
                else:
                    collision_str = "✅ Safe"
                
                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)
                
                print(f"   Reward: {episode_reward:7.2f} | Length: {episode_length:4d} | {collision_str}")
            
            # Print evaluation summary
            print("\n" + "=" * 60)
            print("📋 EVALUATION RESULTS")
            print("=" * 60)
            print(f"Episodes:        {eval_episodes}")
            print(f"Avg Reward:      {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
            print(f"Avg Length:      {np.mean(eval_lengths):.2f} ± {np.std(eval_lengths):.2f}")
            print(f"Min/Max Reward:  {np.min(eval_rewards):.2f} / {np.max(eval_rewards):.2f}")
            print(f"Collision Rate:  {eval_collisions}/{eval_episodes} ({eval_collisions/eval_episodes*100:.1f}%)")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Main run method"""
        if self.training_mode:
            self.train()
        else:
            self.evaluate()

def main():
    """Main entry point"""
    from controller import Robot
    
    # Create robot instance
    robot = Robot()
    
    # Create RL controller
    rl_control = RLControl(
        robot, 
        max_episodes=3000,  # Increased for better training
        max_steps=1500       # Increased step limit
    )
    
    # Run training
    rl_control.run()

if __name__ == "__main__":
    main()