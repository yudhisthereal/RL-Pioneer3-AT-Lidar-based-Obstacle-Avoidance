import matplotlib.pyplot as plt
import numpy as np
import socket
import json
import threading
from collections import deque
import signal
import sys

# ============================================================
# Real-time LIDAR Plotter (Standalone)
# ============================================================
class LidarPlotter:
    def __init__(self):
        self.ranges = []
        self.left_speed = 0
        self.right_speed = 0
        self.state = "INIT"
        self.avoid_threshold = 0.45
        self.safe_distance = 0.8
        self.data_lock = threading.Lock()
        self.running = True
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Setup socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('localhost', 8888))
        self.sock.settimeout(0.1)
        
        # Setup plots
        plt.ion()
        self.fig = plt.figure(figsize=(15, 12))
        
        # Polar plot
        self.ax1 = self.fig.add_subplot(221, projection='polar')
        self.scatter_polar = self.ax1.scatter([], [], c=[], cmap='RdYlGn_r', s=30, alpha=0.7)
        self.ax1.set_ylim(0, 2.5)
        self.ax1.set_title("LIDAR - Polar View", fontsize=14, fontweight='bold')
        self.ax1.grid(True)
        
        # Cartesian plot
        self.ax2 = self.fig.add_subplot(222)
        self.scatter_cartesian = self.ax2.scatter([], [], c=[], cmap='RdYlGn_r', s=30, alpha=0.7)
        self.ax2.set_xlim(-2.5, 2.5)
        self.ax2.set_ylim(-2.5, 2.5)
        self.ax2.set_title("LIDAR - Cartesian View", fontsize=14, fontweight='bold')
        self.ax2.grid(True)
        self.ax2.set_aspect('equal')
        
        # Sector distances plot
        self.ax3 = self.fig.add_subplot(223)
        self.sector_bars = self.ax3.bar([0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0, 0], 
                                       color=['red', 'orange', 'yellow', 'lightgreen', 'green', 'blue', 'purple'])
        self.ax3.set_ylim(0, 2.5)
        self.ax3.set_title("Obstacle Distances by Sector", fontsize=14, fontweight='bold')
        self.ax3.set_xticks([0, 1, 2, 3, 4, 5, 6])
        self.ax3.set_xticklabels(['Far L', 'Left', 'F-L', 'Front', 'F-R', 'Right', 'Far R'], rotation=45)
        self.ax3.axhline(y=self.avoid_threshold, color='red', linestyle='--', label='Avoid Threshold')
        self.ax3.axhline(y=self.safe_distance, color='green', linestyle='--', label='Safe Distance')
        self.ax3.legend()
        
        # Robot status plot
        self.ax4 = self.fig.add_subplot(224)
        self.ax4.axis('off')
        self.status_text = self.ax4.text(0.05, 0.95, 'Waiting for data...', transform=self.ax4.transAxes, 
                                       fontsize=10, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Add close event handler
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        plt.tight_layout()
        
        # Colorbar
        self.fig.colorbar(self.scatter_polar, ax=self.ax1, label='Distance (m)')
        self.fig.colorbar(self.scatter_cartesian, ax=self.ax2, label='Distance (m)')
        
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print('\n\nCtrl+C detected! Shutting down...')
        self.running = False
        self.cleanup()
        sys.exit(0)
        
    def on_close(self, event):
        """Handle window close event"""
        print('Plot window closed! Shutting down...')
        self.running = False
        self.cleanup()
        sys.exit(0)
        
    def cleanup(self):
        """Clean up resources"""
        try:
            self.sock.close()
            plt.close('all')
        except:
            pass
        
    def receive_data(self):
        """Receive data from robot controller"""
        try:
            data, addr = self.sock.recvfrom(65536)
            message = json.loads(data.decode('utf-8'))
            
            with self.data_lock:
                self.ranges = message['ranges']
                self.left_speed = message['left_speed']
                self.right_speed = message['right_speed']
                self.state = message['state']
                self.avoid_threshold = message.get('avoid_threshold', 0.45)
                self.safe_distance = message.get('safe_distance', 0.8)
                
        except socket.timeout:
            pass
        except Exception as e:
            if self.running:  # Only print errors if we're still running
                print(f"Receive error: {e}")
    
    def get_sector_distances(self, ranges):
        """Calculate minimum distances in each sector"""
        if not ranges:
            return [0, 0, 0, 0, 0, 0, 0]
            
        num_sectors = 7
        sector_size = len(ranges) // num_sectors
        sectors = []
        
        for i in range(num_sectors):
            start = i * sector_size
            end = (i + 1) * sector_size
            sector_data = ranges[start:end]
            valid_data = [x for x in sector_data if x < float('inf')]
            sectors.append(min(valid_data) if valid_data else 2.5)
            
        return sectors
    
    def update_plots(self):
        """Update all plots with new data"""
        with self.data_lock:
            ranges = self.ranges.copy()
            left_speed = self.left_speed
            right_speed = self.right_speed
            state = self.state
            threshold = self.avoid_threshold
            safe_dist = self.safe_distance
        
        if not ranges:
            return
        
        N = len(ranges)
        angles = np.linspace(0, 2*np.pi, N)
        
        # Filter out infinite ranges
        valid_mask = np.isfinite(ranges)
        plot_angles = angles[valid_mask]
        plot_ranges = np.array(ranges)[valid_mask]
        
        # Update polar plot
        self.scatter_polar.set_offsets(np.c_[plot_angles, plot_ranges])
        self.scatter_polar.set_array(plot_ranges)
        self.scatter_polar.set_clim(0, 2.5)
        
        # Update threshold circle in polar plot
        if hasattr(self, 'threshold_circle'):
            self.threshold_circle.remove()
        theta = np.linspace(0, 2*np.pi, 100)
        self.threshold_circle, = self.ax1.plot(theta, [threshold]*100, 'r--', linewidth=2, alpha=0.7, label='Avoid Threshold')
        
        # Update safe distance circle in polar plot
        if hasattr(self, 'safe_circle'):
            self.safe_circle.remove()
        self.safe_circle, = self.ax1.plot(theta, [safe_dist]*100, 'g--', linewidth=2, alpha=0.5, label='Safe Distance')
        
        # Update Cartesian plot
        x = plot_ranges * np.cos(plot_angles)
        y = plot_ranges * np.sin(plot_angles)
        self.scatter_cartesian.set_offsets(np.c_[x, y])
        self.scatter_cartesian.set_array(plot_ranges)
        self.scatter_cartesian.set_clim(0, 2.5)
        
        # Update threshold circle in Cartesian plot
        if hasattr(self, 'cartesian_threshold'):
            self.cartesian_threshold.remove()
        self.cartesian_threshold = plt.Circle((0, 0), threshold, color='red', fill=False, 
                                            linestyle='--', linewidth=2, alpha=0.7, label='Avoid Threshold')
        self.ax2.add_patch(self.cartesian_threshold)
        
        # Update safe distance circle in Cartesian plot
        if hasattr(self, 'cartesian_safe'):
            self.cartesian_safe.remove()
        self.cartesian_safe = plt.Circle((0, 0), safe_dist, color='green', fill=False, 
                                       linestyle='--', linewidth=2, alpha=0.5, label='Safe Distance')
        self.ax2.add_patch(self.cartesian_safe)
        
        # Update sector distances (7 sectors now)
        sector_distances = self.get_sector_distances(ranges)
        for bar, height in zip(self.sector_bars, sector_distances):
            bar.set_height(height)
            # Color based on distance to thresholds
            if height < threshold:
                bar.set_color('red')
            elif height < safe_dist:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        # Update status with more detailed information
        status_text = f"""Robot Status:
State: {state}
Left Speed: {left_speed:.2f} rad/s
Right Speed: {right_speed:.2f} rad/s
Avoid Threshold: {threshold:.2f} m
Safe Distance: {safe_dist:.2f} m
Data Points: {len(plot_ranges)}

Sector Distances:
Far Left: {sector_distances[0]:.2f} m
Left: {sector_distances[1]:.2f} m
Front-Left: {sector_distances[2]:.2f} m
Front: {sector_distances[3]:.2f} m
Front-Right: {sector_distances[4]:.2f} m
Right: {sector_distances[5]:.2f} m
Far Right: {sector_distances[6]:.2f} m"""
        
        self.status_text.set_text(status_text)
        
        # Add legends
        self.ax1.legend(loc='upper right', fontsize=8)
        self.ax2.legend(loc='upper right', fontsize=8)
        
        # Refresh plots
        plt.draw()
        try:
            plt.pause(0.001)
        except:
            self.running = False
    
    def run(self):
        """Main loop"""
        print("LIDAR Plotter started. Waiting for data from robot...")
        print("Make sure the robot controller is running in Webots")
        print("\n=== STOPPING INSTRUCTIONS ===")
        print("1. Press Ctrl+C in this terminal")
        print("2. Close the plot window")
        print("3. Press 'q' when plot window is focused")
        print("4. If all else fails, open new terminal and run: killall python3")
        print("==============================\n")
        
        try:
            while self.running:
                self.receive_data()
                self.update_plots()
                
                # Check for 'q' key press
                if plt.get_fignums() and plt.waitforbuttonpress(0.01):
                    if plt.gcf().canvas.get_key() == 'q':
                        print("'q' key pressed! Shutting down...")
                        break
                        
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received!")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self.cleanup()
            print("Plotter stopped successfully!")

if __name__ == "__main__":
    plotter = LidarPlotter()
    plotter.run()