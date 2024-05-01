import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
import cv_bridge
import time
import cv2

import ros2_rt_1_x.models.rt1_inference as rt1_inference


class RtTargetPose(Node):

    def __init__(self):
        super().__init__('rt_target_pose_publisher')
        self.img_converter = cv_bridge.CvBridge()

        # publisher for the target pose
        self.publisher_ = self.create_publisher(Pose, 'rt_target_pose', 10)

        # listener for input images
        self.subscription = self.create_subscription(Image, 'rt_input_image', self.image_listener_callback, 10)

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

    def image_listener_callback(self, msg):
        # store image to disk (for debugging purposes)
        filename = f'./data/received/test_{int(time.time())}.png'
        cv_image = self.img_converter.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv2.imwrite(filename, cv_image)

        self.get_logger().info(f'Received image ({filename})')

    def run_inference(self):
        pass

def main(args=None):
    rclpy.init(args=args)

    rt_target_pose = RtTargetPose()

    rclpy.spin(rt_target_pose)

    #while(True):
        #rt_target_pose.test_pose_publish()