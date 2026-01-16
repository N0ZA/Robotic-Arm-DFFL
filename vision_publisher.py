#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import time
import pyrealsense2 as rs

# Import your existing vision module
import sys
import os

# Ensure the current directory is in the path to find vision_module.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from vision_module import RealSenseVision

class VisionPublisher(Node):
    def __init__(self):
        super().__init__('vision_publisher')
        self.publisher_ = self.create_publisher(Point, '/robot/target_coordinates', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        
        # Initialize Vision System
        self.get_logger().info('Starting Vision System...')
        self.vision = RealSenseVision() # Models assumed to be in the same dir
        self.vision.start()
        self.vision.set_mode("tiny") # Default mode
        
        # Wait for camera to be ready and get intrinsics
        time.sleep(2.0) 
        self.intrinsics = self.vision.get_intrinsics()
        self.get_logger().info('Vision System Ready & Intrinsics Captured')

    def timer_callback(self):
        detection = self.vision.get_latest_coordinates()
        
        if detection:
            # Get pixel coordinates and depth
            u = int(detection['x'])
            v = int(detection['y'])
            z_idx = float(detection['z']) # Z from vision system (meters)
            
            # Deproject Pixel to 3D Point in Camera Frame
            # point = [x, y, z]
            point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], z_idx)
            
            msg = Point()
            msg.x = point[0]
            msg.y = point[1]
            msg.z = point[2]
            
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing: x={msg.x:.3f}, y={msg.y:.3f}, z={msg.z:.3f} (Type: {detection["type"]})')
        else:
            # self.get_logger().info('Scanning...')
            pass

    def destroy_node(self):
        self.vision.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisionPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
