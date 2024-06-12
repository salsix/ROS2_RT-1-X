import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from geometry_msgs.msg import Pose

import time

class PoseControl(Node):
    def __init__(self):
        super().__init__('pose_control')
        self.pose_publisher = self.create_publisher(Pose, 'target_pose', 10)
        self.grip_publisher = self.create_publisher(Float32, 'target_grip', 10)

        self.timer = self.create_timer(5, self.timer_callback)

    def timer_callback(self):
        self.timer.cancel()
        self.publish()
        time.sleep(10)
        # self.publish2()

    def publish(self):
        msg = Pose()
        msg.position.x = -0.4
        msg.position.y = 0.7
        msg.position.z = 0.5
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        msg.orientation.w = 1.0
        self.pose_publisher.publish(msg)

        msg = Float32()
        msg.data = 0.0
        self.grip_publisher.publish(msg)

        print('Published target pose and grip')

    def publish2(self):
        msg = Pose()
        msg.position.x = -0.2
        msg.position.y = 0.4
        msg.position.z = 0.5
        msg.orientation.x = 90.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        msg.orientation.w = 1.0
        self.pose_publisher.publish(msg)

        msg = Float32()
        msg.data = 0.0
        self.grip_publisher.publish(msg)

        print('Published target pose and grip')

def main(args=None):
    rclpy.init(args=args)

    pose_control = PoseControl()
    # pose_control.publish()

    rclpy.spin(pose_control)

    pose_control.destroy_node()
    rclpy.shutdown()