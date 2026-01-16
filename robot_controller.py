#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.subscription = self.create_subscription(
            Point,
            '/robot/target_coordinates',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.get_logger().info('Robot Controller Node Started. Waiting for coordinates...')

    def listener_callback(self, msg):
        self.get_logger().info(f'Received Target: X={msg.x:.3f}m, Y={msg.y:.3f}m, Z={msg.z:.3f}m')
        
        # -----------------------------------------------------------------
        # TODO: INSERT YOUR ROBOT ARM MOVEMENT CODE HERE
        # -----------------------------------------------------------------
        # Example pseudo-code:
        # 1. Transform Camera Frame -> Robot Base Frame
        #    robot_x = msg.x + offset_x
        #    robot_y = msg.y + offset_y
        #    robot_z = msg.z + offset_z
        #
        # 2. Call Robot Driver (e.g., using a library or serial command)
        #    driver.move_to(robot_x, robot_y, robot_z)
        # -----------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
