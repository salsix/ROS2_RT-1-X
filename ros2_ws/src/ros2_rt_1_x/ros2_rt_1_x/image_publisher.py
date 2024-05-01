import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
import ros2_rt_1_x.camera as cam
import time
import cv_bridge
import cv2

class ImagePublisher(Node):

    def __init__(self):
        super().__init__('image_publisher')

        self.camera = cam.Camera()

        # cv bridge to convert between OpenCV and ROS Image messages
        self.img_converter = cv_bridge.CvBridge()

        # publisher for the image
        self.publisher_ = self.create_publisher(Image, 'rt_input_image', 10)

        timer_period = 5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        self.publish_image()

    def publish_image(self):
        ocv_img_4ch = self.camera.get_picture()

        # convert to 3-channel image
        ocv_img = cv2.cvtColor(ocv_img_4ch, cv2.COLOR_BGRA2BGR)

        img_msg = self.img_converter.cv2_to_imgmsg(ocv_img, encoding='bgr8')
        self.publisher_.publish(img_msg)
        self.get_logger().info('Published image.')


def main(args=None):
    rclpy.init(args=args)

    image_publisher = ImagePublisher()

    rclpy.spin(image_publisher)