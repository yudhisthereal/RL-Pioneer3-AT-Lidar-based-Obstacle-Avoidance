import numpy as np
import math
from controller import Robot, Motor, DistanceSensor, Lidar
from collections import deque

class BaseEnv:
    def __init__(self, robot, max_steps=1000):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.max_steps = max_steps
        self.current_step = 0
        
        # Initialize robot components
        self.setup_robot()
        
        # Environment parameters
        self.lidar_rays = 16
        self.max_velocity = 4.0  # Dikurangi dari 6.4 ke 4.0 untuk kontrol lebih baik
        
        # Position tracking untuk GUI
        self.position_history = []
        self.last_position = None
        
    def setup_robot(self):
        """Initialize robot motors and sensors"""
        # Get motors
        self.motors = []
        motor_names = ['front left wheel', 'front right wheel', 
                      'back left wheel', 'back right wheel']
        
        for name in motor_names:
            motor = self.robot.getDevice(name)
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)
            self.motors.append(motor)
        
        # Get lidar
        self.lidar = self.robot.getDevice('Sick LMS 291')
        if self.lidar:
            self.lidar.enable(self.timestep)
            self.lidar_width = self.lidar.getHorizontalResolution()
        
        # Get GPS for position tracking
        self.gps = self.robot.getDevice('gps')
        if self.gps:
            self.gps.enable(self.timestep)
        
        # Get IMU for orientation
        self.imu = self.robot.getDevice('inertial unit')
        if self.imu:
            self.imu.enable(self.timestep)
            
        # Setup GUI Display
        self.setup_gui()
    
    def setup_gui(self):
        """Setup GUI untuk monitoring real-time"""
        try:
            self.display = self.robot.getDevice('display')
            if self.display:
                self.display_width = self.display.getWidth()
                self.display_height = self.display.getHeight()
                print(f"✅ GUI Display: {self.display_width}x{self.display_height}")
            else:
                self.display = None
        except:
            self.display = None
            print("❌ No display available")
    
    def update_display(self, lidar_data, velocity, angular_vel, action):
        """Update GUI dengan informasi lengkap"""
        if not self.display:
            return
        
        # Clear display dengan background hitam
        self.display.setColor(0x000000)
        self.display.fillRectangle(0, 0, self.display_width, self.display_height)
        
        # === LIDAR VISUALIZATION ===
        center_x = self.display_width // 2
        center_y = self.display_height // 2
        max_range = 10.0
        scale = min(center_x, center_y) - 30
        
        # Draw robot body (lingkaran hijau di tengah)
        self.display.setColor(0x00FF00)
        self.display.fillOval(center_x - 8, center_y - 8, 16, 16)
        
        # Draw lidar rays dengan warna berdasarkan jarak
        num_rays = len(lidar_data)
        for i, distance in enumerate(lidar_data):
            angle = (2 * math.pi * i / num_rays) - math.pi / 2  # Start from front
            
            # Normalisasi jarak
            normalized_dist = min(distance / max_range, 1.0)
            end_x = center_x + int(normalized_dist * scale * math.cos(angle))
            end_y = center_y + int(normalized_dist * scale * math.sin(angle))
            
            # Warna berdasarkan tingkat bahaya
            if distance < 0.4:
                color = 0xFF0000  # MERAH - Bahaya!
                thickness = 3
            elif distance < 0.8:
                color = 0xFF8800  # ORANGE - Hati-hati
                thickness = 2
            elif distance < 1.5:
                color = 0xFFFF00  # KUNING - Perhatian
                thickness = 2
            else:
                color = 0x00FF00  # HIJAU - Aman
                thickness = 1
            
            # Draw ray dengan thickness
            self.display.setColor(color)
            for t in range(thickness):
                self.display.drawLine(center_x, center_y, end_x + t, end_y)
            
            # Draw distance dots di ujung ray
            self.display.fillOval(end_x - 2, end_y - 2, 4, 4)
        
        # Draw direction indicator (panah arah robot)
        arrow_length = 20
        arrow_x = center_x + int(arrow_length * math.cos(-math.pi / 2))
        arrow_y = center_y + int(arrow_length * math.sin(-math.pi / 2))
        self.display.setColor(0x0000FF)  # Biru untuk arah
        self.display.drawLine(center_x, center_y, arrow_x, arrow_y)
        
        # === STATUS PANEL (Kiri Atas) ===
        self.display.setColor(0xFFFFFF)
        y_offset = 10
        
        # Step counter
        self.display.drawText(f"Step: {self.current_step}/{self.max_steps}", 10, y_offset)
        y_offset += 15
        
        # Velocity info dengan bar
        vel_normalized = abs(velocity) / self.max_velocity
        self.display.drawText(f"Velocity: {velocity:.2f} m/s", 10, y_offset)
        y_offset += 15
        
        # Velocity bar
        bar_width = 100
        bar_height = 8
        self.display.setColor(0x333333)
        self.display.drawRectangle(10, y_offset, bar_width, bar_height)
        self.display.setColor(0x00FF00)
        self.display.fillRectangle(10, y_offset, int(bar_width * vel_normalized), bar_height)
        y_offset += 15
        
        # Angular velocity
        ang_vel_color = 0x00FFFF if abs(angular_vel) > 0.1 else 0xFFFFFF
        self.display.setColor(ang_vel_color)
        direction = "↻" if angular_vel > 0 else "↺" if angular_vel < 0 else "→"
        self.display.drawText(f"Turn {direction}: {angular_vel:.2f} rad/s", 10, y_offset)
        y_offset += 20
        
        # === SENSOR READINGS (Kiri Tengah) ===
        self.display.setColor(0xFFFFFF)
        self.display.drawText("=== SENSORS ===", 10, y_offset)
        y_offset += 15
        
        # Front sensors (depan)
        front_indices = list(range(6, 11)) if num_rays >= 16 else [num_rays // 2]
        front_dist = np.min([lidar_data[i] for i in front_indices if i < num_rays])
        front_color = 0xFF0000 if front_dist < 0.4 else 0xFFFF00 if front_dist < 0.8 else 0x00FF00
        self.display.setColor(front_color)
        self.display.drawText(f"Front:  {front_dist:.2f}m", 10, y_offset)
        y_offset += 15
        
        # Left sensors (kiri)
        left_indices = list(range(0, 4)) if num_rays >= 16 else [0]
        left_dist = np.min([lidar_data[i] for i in left_indices if i < num_rays])
        left_color = 0xFF0000 if left_dist < 0.4 else 0xFFFF00 if left_dist < 0.8 else 0x00FF00
        self.display.setColor(left_color)
        self.display.drawText(f"Left:   {left_dist:.2f}m", 10, y_offset)
        y_offset += 15
        
        # Right sensors (kanan)
        right_indices = list(range(12, 16)) if num_rays >= 16 else [num_rays - 1]
        right_dist = np.min([lidar_data[i] for i in right_indices if i < num_rays])
        right_color = 0xFF0000 if right_dist < 0.4 else 0xFFFF00 if right_dist < 0.8 else 0x00FF00
        self.display.setColor(right_color)
        self.display.drawText(f"Right:  {right_dist:.2f}m", 10, y_offset)
        y_offset += 20
        
        # === ACTION INFO ===
        self.display.setColor(0xFFFFFF)
        self.display.drawText("=== MOTORS ===", 10, y_offset)
        y_offset += 15
        
        self.display.drawText(f"Left:  {action[0]:.2f}", 10, y_offset)
        y_offset += 15
        self.display.drawText(f"Right: {action[1]:.2f}", 10, y_offset)
        y_offset += 15
        
        # Motor balance indicator
        balance = abs(action[0] - action[1])
        if balance < 0.1:
            self.display.setColor(0x00FF00)
            self.display.drawText("Status: STRAIGHT", 10, y_offset)
        elif action[0] > action[1]:
            self.display.setColor(0x00FFFF)
            self.display.drawText("Status: LEFT TURN", 10, y_offset)
        else:
            self.display.setColor(0x00FFFF)
            self.display.drawText("Status: RIGHT TURN", 10, y_offset)
        y_offset += 20
        
        # === POSITION TRACKING (Kanan Atas) ===
        if self.gps:
            pos = self.gps.getValues()
            self.display.setColor(0xFFFFFF)
            self.display.drawText(f"X: {pos[0]:.2f}", self.display_width - 120, 10)
            self.display.drawText(f"Y: {pos[1]:.2f}", self.display_width - 120, 25)
            self.display.drawText(f"Z: {pos[2]:.2f}", self.display_width - 120, 40)
            
            # Track movement
            if self.last_position is not None:
                movement = math.sqrt((pos[0] - self.last_position[0])**2 + 
                                   (pos[1] - self.last_position[1])**2)
                self.display.drawText(f"Move: {movement:.3f}", self.display_width - 120, 55)
            self.last_position = pos
        
        # === WARNING MESSAGES (Bawah) ===
        warning_y = self.display_height - 40
        
        # Collision warning
        if front_dist < 0.3:
            self.display.setColor(0xFF0000)
            self.display.fillRectangle(10, warning_y, self.display_width - 20, 30)
            self.display.setColor(0xFFFFFF)
            self.display.drawText("!!! COLLISION WARNING !!!", 
                                 self.display_width // 2 - 80, warning_y + 10)
        elif front_dist < 0.5:
            self.display.setColor(0xFFAA00)
            self.display.fillRectangle(10, warning_y, self.display_width - 20, 30)
            self.display.setColor(0x000000)
            self.display.drawText("! CAUTION: TOO CLOSE !", 
                                 self.display_width // 2 - 70, warning_y + 10)
    
    def get_lidar_data(self):
        """Get processed lidar data"""
        if not self.lidar:
            return np.ones(self.lidar_rays) * 10.0
        
        lidar_data = self.lidar.getRangeImage()
        
        if len(lidar_data) == 0:
            return np.ones(self.lidar_rays) * 10.0
        
        total_rays = len(lidar_data)
        step = total_rays // self.lidar_rays
        
        processed_data = []
        for i in range(0, total_rays, step):
            if i < total_rays:
                value = lidar_data[i]
                if math.isinf(value) or value > 10.0:
                    value = 10.0
                processed_data.append(value)
        
        while len(processed_data) < self.lidar_rays:
            processed_data.append(10.0)
        
        return np.array(processed_data[:self.lidar_rays])
    
    def get_robot_velocity(self):
        """Get robot linear velocity"""
        left_speed = (self.motors[0].getVelocity() + self.motors[2].getVelocity()) / 2
        right_speed = (self.motors[1].getVelocity() + self.motors[3].getVelocity()) / 2
        return (left_speed + right_speed) / 2
    
    def get_robot_angular_velocity(self):
        """Get robot angular velocity"""
        left_speed = (self.motors[0].getVelocity() + self.motors[2].getVelocity()) / 2
        right_speed = (self.motors[1].getVelocity() + self.motors[3].getVelocity()) / 2
        return (right_speed - left_speed) / 2
    
    def get_robot_orientation(self):
        """Get robot orientation"""
        if self.imu:
            return np.array(self.imu.getRollPitchYaw())
        return np.array([0.0, 0.0, 0.0])
    
    def apply_action(self, action):
        """Apply motor actions with SMART ESCAPE mechanism"""
        if len(action) != 2:
            return
        
        # Get current lidar state
        lidar_data = self.get_lidar_data()
        
        # Define all zones
        front_rays = lidar_data[6:10] if len(lidar_data) >= 16 else [lidar_data[len(lidar_data)//2]]
        left_rays = lidar_data[0:4] if len(lidar_data) >= 16 else [lidar_data[0]]
        right_rays = lidar_data[12:16] if len(lidar_data) >= 16 else [lidar_data[-1]]
        back_rays = lidar_data[13:16] + lidar_data[0:3] if len(lidar_data) >= 16 else [lidar_data[-1]]
        
        min_front = np.min(front_rays)
        min_left = np.min(left_rays)
        min_right = np.min(right_rays)
        min_back = np.min(back_rays)
        avg_left = np.mean(left_rays)
        avg_right = np.mean(right_rays)
        
        # DETECT STUCK IN CORNER (all sides blocked)
        if min_front < 0.3 and min_left < 0.3 and min_right < 0.3:
            print(f"🆘 STUCK IN CORNER! F:{min_front:.2f} L:{min_left:.2f} R:{min_right:.2f}")
            
            # Find the LEAST blocked direction
            max_dist = max(min_front, min_left, min_right, min_back)
            
            if max_dist == min_back or max_dist < 0.25:
                # All sides very blocked - do AGGRESSIVE SPIN
                print("   ⚡ EMERGENCY SPIN - Finding escape!")
                if not hasattr(self, 'spin_direction'):
                    self.spin_direction = 1 if avg_left > avg_right else -1
                
                if self.spin_direction > 0:
                    action = np.array([0.1, 0.5])  # Spin left
                else:
                    action = np.array([0.5, 0.1])  # Spin right
                
                # Count stuck steps
                if not hasattr(self, 'stuck_counter'):
                    self.stuck_counter = 0
                self.stuck_counter += 1
                
                # After 20 steps stuck, try opposite direction
                if self.stuck_counter > 20:
                    self.spin_direction *= -1
                    self.stuck_counter = 0
                    print("   🔄 Trying opposite spin direction")
            
            else:
                # Turn towards most open space
                if max_dist == min_left:
                    print("   ↺ Turning to LEFT (most open)")
                    action = np.array([0.1, 0.6])
                elif max_dist == min_right:
                    print("   ↻ Turning to RIGHT (most open)")
                    action = np.array([0.6, 0.1])
                else:  # max_dist == min_front
                    print("   → Moving FORWARD (front clearest)")
                    action = np.array([0.4, 0.4])
                
                # Reset stuck counter if we found a direction
                if hasattr(self, 'stuck_counter'):
                    self.stuck_counter = 0
        
        # NORMAL SAFETY OVERRIDE (not stuck in corner)
        elif min_front < 0.35:
            # Clear stuck counter
            if hasattr(self, 'stuck_counter'):
                del self.stuck_counter
            if hasattr(self, 'spin_direction'):
                del self.spin_direction
            
            # Choose better side
            if avg_left > avg_right + 0.2:  # Left significantly more open
                action = np.array([0.15, 0.85])
                print(f"🚨 SAFETY: Hard LEFT (L:{avg_left:.2f} > R:{avg_right:.2f})")
            elif avg_right > avg_left + 0.2:  # Right significantly more open
                action = np.array([0.85, 0.15])
                print(f"🚨 SAFETY: Hard RIGHT (R:{avg_right:.2f} > L:{avg_left:.2f})")
            else:  # Both sides similar - pick based on min distance
                if min_left > min_right:
                    action = np.array([0.15, 0.85])
                    print(f"🚨 SAFETY: LEFT (MinL:{min_left:.2f} > MinR:{min_right:.2f})")
                else:
                    action = np.array([0.85, 0.15])
                    print(f"🚨 SAFETY: RIGHT (MinR:{min_right:.2f} > MinL:{min_left:.2f})")
        
        elif min_front < 0.6:  # WARNING ZONE
            # Clear stuck indicators
            if hasattr(self, 'stuck_counter'):
                del self.stuck_counter
            if hasattr(self, 'spin_direction'):
                del self.spin_direction
            
            # Gentle avoidance
            slowdown = (min_front - 0.35) / 0.25
            if avg_left > avg_right:
                action = action * slowdown * np.array([0.6, 1.0])
            else:
                action = action * slowdown * np.array([1.0, 0.6])
        
        else:
            # SAFE - Clear all stuck indicators
            if hasattr(self, 'stuck_counter'):
                del self.stuck_counter
            if hasattr(self, 'spin_direction'):
                del self.spin_direction
        
        # CLIP action ke range [0, 1] - HANYA MAJU!
        left_speed = np.clip(action[0], 0.0, 1.0) * self.max_velocity
        right_speed = np.clip(action[1], 0.0, 1.0) * self.max_velocity
        
        # Apply to motors
        self.motors[0].setVelocity(left_speed)
        self.motors[1].setVelocity(right_speed)
        self.motors[2].setVelocity(left_speed)
        self.motors[3].setVelocity(right_speed)
    
    def compute_reward(self, action):
        """Compute reward - FOKUS HINDARI TABRAKAN"""
        lidar_data = self.get_lidar_data()
        min_distance = np.min(lidar_data)
        velocity = self.get_robot_velocity()
        
        # 1. COLLISION PENALTY - SANGAT BESAR
        if min_distance < 0.25:
            return -20.0  # Increased penalty
        elif min_distance < 0.4:
            return -8.0   # Increased penalty
        elif min_distance < 0.6:
            return -3.0   # Increased penalty
        
        # 2. FORWARD MOTION REWARD - encourage movement
        forward_reward = 0.0
        if action[0] > 0.2 and action[1] > 0.2:  # Both wheels moving forward
            avg_action = (action[0] + action[1]) / 2
            forward_reward = avg_action * 0.5
        
        # 3. VELOCITY REWARD - only if safe
        velocity_reward = 0.0
        if min_distance > 1.0:
            velocity_reward = velocity * 0.2
        elif min_distance > 0.8:
            velocity_reward = velocity * 0.1
        
        # 4. SAFE DISTANCE REWARD - bonus for maintaining good distance
        safe_reward = 0.0
        if 0.8 < min_distance < 3.0:
            safe_reward = 0.5
        elif 0.6 < min_distance < 0.8:
            safe_reward = 0.2
        
        # 5. BALANCE REWARD - keep centered
        left_dist = np.mean(lidar_data[0:4])
        right_dist = np.mean(lidar_data[12:16])
        balance_diff = abs(left_dist - right_dist)
        balance_reward = 0.2 * (1.0 - min(balance_diff / 2.0, 1.0))
        
        # 6. SMOOTH ACTION - penalize erratic movements
        action_diff = abs(action[0] - action[1])
        smooth_reward = 0.1 * (1.0 - action_diff)
        
        # 7. STEP SURVIVAL REWARD - just for staying alive
        survival_reward = 0.1
        
        total_reward = (
            forward_reward + 
            velocity_reward + 
            safe_reward + 
            balance_reward + 
            smooth_reward +
            survival_reward
        )
        
        return np.clip(total_reward, -20.0, 2.0)
    
    def is_done(self):
        """Check termination with STUCK DETECTION"""
        self.current_step += 1
        
        if self.current_step >= self.max_steps:
            return True
        
        lidar_data = self.get_lidar_data()
        min_distance = np.min(lidar_data)
        
        # COLLISION
        if min_distance < 0.2:
            print(f"💥 COLLISION! Distance: {min_distance:.3f}")
            return True
        
        # STUCK DETECTION - improved
        velocity = self.get_robot_velocity()
        
        if not hasattr(self, 'velocity_history'):
            self.velocity_history = deque(maxlen=50)
        
        self.velocity_history.append(abs(velocity))
        
        # Check if stuck (very low velocity for extended period)
        if len(self.velocity_history) >= 50:
            avg_velocity = np.mean(self.velocity_history)
            
            # Also check if all directions are blocked
            front_dist = np.min(lidar_data[6:10]) if len(lidar_data) >= 16 else min_distance
            
            if avg_velocity < 0.15 and self.current_step > 100:
                if not hasattr(self, 'low_velocity_count'):
                    self.low_velocity_count = 0
                self.low_velocity_count += 1
                
                # If stuck for too long AND trapped, end episode
                if self.low_velocity_count > 150 and front_dist < 0.4:
                    print(f"🔄 STUCK DETECTED! Avg velocity: {avg_velocity:.3f}, Steps: {self.current_step}")
                    return True
            else:
                # Reset counter if moving
                if hasattr(self, 'low_velocity_count'):
                    self.low_velocity_count = 0
        
        return False
    
    def reset(self):
        """Reset environment"""
        self.current_step = 0
        self.last_position = None
        
        # Clear stuck detection
        if hasattr(self, 'velocity_history'):
            self.velocity_history.clear()
        if hasattr(self, 'low_velocity_count'):
            del self.low_velocity_count
        if hasattr(self, 'stuck_counter'):
            del self.stuck_counter
        if hasattr(self, 'spin_direction'):
            del self.spin_direction
        
        for motor in self.motors:
            motor.setVelocity(0.0)
        
        self.robot.step(self.timestep)
        return self.get_state()
    
    def get_state(self):
        """Get current state"""
        lidar_data = self.get_lidar_data()
        velocity = self.get_robot_velocity()
        angular_velocity = self.get_robot_angular_velocity()
        
        state = np.concatenate([
            lidar_data,
            [velocity / self.max_velocity,  # Normalized
             angular_velocity / 2.0]         # Normalized
        ])
        
        return state
    
    def get_state_size(self):
        return self.lidar_rays + 2
    
    def get_action_size(self):
        return 2
    
    def step(self, action):
        """Execute environment step"""
        # Update GUI
        lidar_data = self.get_lidar_data()
        velocity = self.get_robot_velocity()
        angular_vel = self.get_robot_angular_velocity()
        self.update_display(lidar_data, velocity, angular_vel, action)
        
        # Apply action
        self.apply_action(action)
        
        # Step simulation
        self.robot.step(self.timestep)
        
        # Get results
        next_state = self.get_state()
        reward = self.compute_reward(action)
        done = self.is_done()
        
        return next_state, reward, done, {}