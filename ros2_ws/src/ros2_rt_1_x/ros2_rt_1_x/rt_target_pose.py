import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Pose


class RtTargetPose(Node):

    def __init__(self):
        super().__init__('rt_target_pose_publisher')
        self.publisher_ = self.create_publisher(Pose, 'rt_target_pose', 10)

    def test_pose_publish(self):
        msg = Pose()
        msg.position.x = 1.0
        msg.position.y = 2.0
        msg.position.z = 3.0
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        msg.orientation.w = 1.0
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg)

def main(args=None):
    rclpy.init(args=args)

    rt_target_pose = RtTargetPose()

    while(True):
        rt_target_pose.test_pose_publish()