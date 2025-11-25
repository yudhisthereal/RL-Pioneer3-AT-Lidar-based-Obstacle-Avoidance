import numpy as np
import math
from controller import Robot, Motor, DistanceSensor, Lidar

class BaseEnv:
    def __init__(self, robot, max_steps=1000):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.max_steps = max_steps
        self.current_step = 0
        
        # Initialize robot components
        self.setup_robot()
        
        # Environment parameters
        self.lidar_rays = 16  # Number of lidar rays to use
        self.max_velocity = 6.4  # Maximum motor velocity
        
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
        
        # Get distance sensors (sonar)
        self.distance_sensors = []
        for i in range(16):
            sensor_name = f'so{i}'
            sensor = self.robot.getDevice(sensor_name)
            if sensor:
                sensor.enable(self.timestep)
                self.distance_sensors.append(sensor)
        
        # Get GPS for position tracking (if available)
        self.gps = self.robot.getDevice('gps')
        if self.gps:
            self.gps.enable(self.timestep)
        
        # Get IMU for orientation (if available)
        self.imu = self.robot.getDevice('inertial unit')
        if self.imu:
            self.imu.enable(self.timestep)
    
    def get_lidar_data(self):
        """Get processed lidar data"""
        if not self.lidar:
            return np.ones(self.lidar_rays) * 10.0  # Default far distance
        
        # Get raw lidar data
        lidar_data = self.lidar.getRangeImage()
        
        if len(lidar_data) == 0:
            return np.ones(self.lidar_rays) * 10.0
        
        # Process lidar data - select subset of rays
        total_rays = len(lidar_data)
        step = total_rays // self.lidar_rays
        
        processed_data = []
        for i in range(0, total_rays, step):
            if i < total_rays:
                # Filter invalid values
                value = lidar_data[i]
                if math.isinf(value) or value > 10.0:
                    value = 10.0
                processed_data.append(value)
        
        # Ensure we have exactly lidar_rays values
        while len(processed_data) < self.lidar_rays:
            processed_data.append(10.0)
        
        return np.array(processed_data[:self.lidar_rays])
    
    def get_sonar_data(self):
        """Get distance sensor data"""
        sonar_data = []
        for sensor in self.distance_sensors:
            value = sensor.getValue()
            # Filter invalid values
            if math.isinf(value) or value > 2.0:
                value = 2.0
            sonar_data.append(value)
        
        return np.array(sonar_data)
    
    def get_robot_position(self):
        """Get robot position from GPS"""
        if self.gps:
            return np.array(self.gps.getValues())
        else:
            return np.array([0.0, 0.0, 0.0])
    
    def get_robot_velocity(self):
        """Get robot linear velocity"""
        if self.gps:
            # GPS can provide velocity in some implementations
            # This is a simplified version
            return 0.0
        else:
            # Estimate velocity from motor speeds
            left_speed = (self.motors[0].getVelocity() + self.motors[2].getVelocity()) / 2
            right_speed = (self.motors[1].getVelocity() + self.motors[3].getVelocity()) / 2
            return (left_speed + right_speed) / 2
    
    def get_robot_angular_velocity(self):
        """Get robot angular velocity"""
        if self.imu:
            # Get rotation rates from IMU
            imu_values = self.imu.getRollPitchYaw()
            return imu_values[2]  # Yaw rate
        else:
            # Estimate from wheel speed difference
            left_speed = (self.motors[0].getVelocity() + self.motors[2].getVelocity()) / 2
            right_speed = (self.motors[1].getVelocity() + self.motors[3].getVelocity()) / 2
            return (right_speed - left_speed) / 2
    
    def get_robot_orientation(self):
        """Get robot orientation"""
        if self.imu:
            return np.array(self.imu.getRollPitchYaw())
        else:
            return np.array([0.0, 0.0, 0.0])
    
    def apply_action(self, action):
        """Apply motor actions to the robot"""
        # action should be [left_speed, right_speed] in range [-1, 1]
        if len(action) == 2:
            left_speed = np.clip(action[0], -1, 1) * self.max_velocity
            right_speed = np.clip(action[1], -1, 1) * self.max_velocity
            
            # Apply to all motors
            self.motors[0].setVelocity(left_speed)   # front left
            self.motors[1].setVelocity(right_speed)  # front right
            self.motors[2].setVelocity(left_speed)   # back left
            self.motors[3].setVelocity(right_speed)  # back right
    
    def compute_reward(self, action):
        """Compute reward based on current state and action"""
        # Basic reward components
        lidar_data = self.get_lidar_data()
        min_distance = np.min(lidar_data)
        
        # Reward for moving forward
        velocity_reward = self.get_robot_velocity() * 0.1
        
        # Penalty for being too close to obstacles
        collision_penalty = 0.0
        if min_distance < 0.3:
            collision_penalty = -10.0
        elif min_distance < 0.5:
            collision_penalty = -2.0
        
        # Small penalty for large actions (encourage smooth driving)
        action_penalty = -0.01 * np.sum(np.square(action))
        
        # Total reward
        total_reward = velocity_reward + collision_penalty + action_penalty
        
        return total_reward
    
    def is_done(self):
        """Check if episode should end"""
        self.current_step += 1
        
        # Check max steps
        if self.current_step >= self.max_steps:
            return True
        
        # Check collision
        lidar_data = self.get_lidar_data()
        min_distance = np.min(lidar_data)
        if min_distance < 0.2:  # Collision threshold
            return True
        
        # Check if stuck (optional)
        # You can add more termination conditions here
        
        return False
    
    def reset(self):
        """Reset environment for new episode"""
        self.current_step = 0
        
        # Stop motors
        for motor in self.motors:
            motor.setVelocity(0.0)
        
        # Reset robot position (if simulation allows)
        # Note: This might require supervisor privileges
        # In Webots, you might need to use supervisor mode to reset position
        
        # Step simulation to apply changes
        self.robot.step(self.timestep)
        
        # Return initial state
        return self.get_state()
    
    def get_state(self):
        """Get current state representation"""
        # Basic state: lidar data + velocity info
        lidar_data = self.get_lidar_data()
        velocity = self.get_robot_velocity()
        angular_velocity = self.get_robot_angular_velocity()
        
        state = np.concatenate([
            lidar_data,
            [velocity, angular_velocity]
        ])
        
        return state
    
    def get_state_size(self):
        """Get size of state vector"""
        return self.lidar_rays + 2  # lidar + velocity + angular_velocity
    
    def get_action_size(self):
        """Get size of action vector"""
        return 2  # [left_motor_speed, right_motor_speed]
    
    def step(self, action):
        """Execute one environment step"""
        # Apply action
        self.apply_action(action)
        
        # Step simulation
        self.robot.step(self.timestep)
        
        # Get next state
        next_state = self.get_state()
        
        # Compute reward
        reward = self.compute_reward(action)
        
        # Check if done
        done = self.is_done()
        
        return next_state, reward, done, {}