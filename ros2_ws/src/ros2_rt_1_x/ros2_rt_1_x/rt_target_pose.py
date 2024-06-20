import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
import cv_bridge
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL 
import tensorflow as tf

import ros2_rt_1_x.models.rt1_inference as jax_models
import ros2_rt_1_x.camera as camera
import ros2_rt_1_x.tf_models.tf_rt1_inference as tf_models
import ros2_rt_1_x.output_logging as output_log


class RtTargetPose(Node):

    def __init__(self):
        # disable GPU for tensorflow, since it causes problems with resource allocation together 
        # with jax. We only use tensorflow for minor tasks, so it doesn't matter.
        tf.config.experimental.set_visible_devices([], "GPU")

        super().__init__('rt_target_pose_publisher')
        self.pose_publisher = self.create_publisher(Pose, 'target_pose', 10)
        self.grip_publisher = self.create_publisher(Float32, 'target_grip', 10)

        self.natural_language_instruction = "Place the yellow banana in the pan."
        self.inference_interval = 0.1
        self.inference_steps = 38

        # self.rt1_inferer = tf_models.RT1TensorflowInferer(self.natural_language_instruction)
        self.rt1_inferer = jax_models.RT1Inferer(self.natural_language_instruction)

        self.cur_x = 0.0
        self.cur_y = 0.5
        self.cur_z = 0.3
        self.cur_roll = 0.0
        self.cur_pitch = 90.0
        self.cur_yaw = 0.0
        self.cur_grip = 0.02

        self.pose_history = []
        self.pose_history.append([self.cur_x, self.cur_y, self.cur_z, self.cur_roll, self.cur_pitch, self.cur_yaw, self.cur_grip, [0,1,0]])

        self.img_converter = cv_bridge.CvBridge()

        self.run_inference()

    def store_image(self, image):
        filename = f'./data/received/test_{int(time.time())}.png'
        cv2.imwrite(filename, image)
        self.get_logger().info(f'Stored image to {filename}')

    def run_inference(self):
        self.init_target_pose()
        time.sleep(5)
        print('TOOK ON INIT POSE. RUNNING INFERENCE...')
        actions = []
        steps = 0
        while steps < self.inference_steps:

            # image = PIL.Image.open(f'/home/jonathan/Thesis/open_x_embodiment/imgs/bridge/{steps+1}.png')

            # # self.store_image(image)
            # image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            # action = self.rt1_inferer.run_inference_step(image)
            # self.get_logger().info(f'Action: {action}')
            # self.publish_target_pose(action)

            # act = self.rt1_inferer.run_inference(steps)
            act = self.rt1_inferer.run_bridge_inference(steps)

            # image = self.camera.get_picture()
            # act = self.rt1_jax_inferer.run_inference(image,steps)

            self.publish_target_pose_deltas(act)
            actions.append(act)

            print(hash(str(act)))

            time.sleep(self.inference_interval)
            steps += 1
            print(f'Step {steps} done.')
        print('DONE RUNNING INFERENCE.')

        filename = str(int(time.time()))

        output_log.create_full_log(self.pose_history, actions, self.natural_language_instruction, self.inference_interval, filename)


    def publish_target_pose(self, action):

        gripper_closedness_action = action["gripper_closedness_action"]
        rotation_delta = action["rotation_delta"]
        terminate_episode = action["terminate_episode"]
        world_vector = action["world_vector"]

        pos_x = float(world_vector[0])
        pos_y = float(world_vector[1])
        pos_z = float(world_vector[2])
        roll = float(rotation_delta[0])
        pitch = float(rotation_delta[1])
        yaw = float(rotation_delta[2])
        grip = float(gripper_closedness_action[0])

        # print(f'Publishing target pose: {pos_x}, {pos_y}, {pos_z}, {roll}, {pitch}, {yaw}, {grip}')
        self.get_logger().info(f'Publishing target pose and grip...')
        # self.get_logger().info(f'pos_x: {pos_x}, pos_y: {pos_y}, pos_z: {pos_z}, roll: {roll}, pitch: {pitch}, yaw: {yaw}, grip: {grip}')

        pose_msg = Pose()
        pose_msg.position.x = pos_x
        pose_msg.position.y = pos_y
        pose_msg.position.z = pos_z
        pose_msg.orientation.x = yaw
        pose_msg.orientation.y = pitch
        pose_msg.orientation.z = roll
        pose_msg.orientation.w = 1.0

        grip_msg = Float32()
        grip_msg.data = grip

        self.pose_publisher.publish(pose_msg)
        self.grip_publisher.publish(grip_msg)

    def publish_target_pose_deltas(self, action):

        gripper_closedness_action = action["gripper_closedness_action"]
        rotation_delta = action["rotation_delta"]
        terminate_episode = action["terminate_episode"]
        world_vector = action["world_vector"]

        self.cur_x += float(world_vector[0])
        self.cur_y += float(world_vector[1])
        self.cur_z += float(world_vector[2])
        self.cur_roll += float(rotation_delta[0])
        self.cur_pitch += float(rotation_delta[1])
        self.cur_yaw += float(rotation_delta[2])
        self.cur_grip = float(gripper_closedness_action[0])

        self.cur_x = min(max(self.cur_x, -0.5), 0.5)
        self.cur_y = min(max(self.cur_y, 0.2), 0.7)
        self.cur_z = min(max(self.cur_z, 0.0), 0.4)
        self.cur_roll = min(max(self.cur_roll, 0.0), 90.0)
        self.cur_pitch = min(max(self.cur_pitch, 0.0), 90.0)
        self.cur_yaw = min(max(self.cur_yaw, -10.0), 170.0)
        self.cur_grip = min(max(self.cur_grip, 0.02), 0.08)

        self.get_logger().info(f'Publishing target pose and grip...')

        pose_msg = Pose()
        pose_msg.position.x = self.cur_x
        pose_msg.position.y = self.cur_y
        pose_msg.position.z = self.cur_z
        pose_msg.orientation.x = self.cur_yaw
        pose_msg.orientation.y = self.cur_pitch
        pose_msg.orientation.z = self.cur_roll
        pose_msg.orientation.w = 1.0

        grip_msg = Float32()
        grip_msg.data = self.cur_grip
        print('GRIPPER CLOSEDNESS: ', self.cur_grip)

        self.pose_publisher.publish(pose_msg)
        self.grip_publisher.publish(grip_msg)

        self.pose_history.append([self.cur_x, self.cur_y, self.cur_z, self.cur_roll, self.cur_pitch, self.cur_yaw, self.cur_grip, terminate_episode])

    def init_target_pose(self):
        pose_msg = Pose()
        pose_msg.position.x = self.cur_x
        pose_msg.position.y = self.cur_y
        pose_msg.position.z = self.cur_z
        pose_msg.orientation.x = self.cur_yaw
        pose_msg.orientation.y = self.cur_pitch
        pose_msg.orientation.z = self.cur_roll
        pose_msg.orientation.w = 1.0

        grip_msg = Float32()
        grip_msg.data = self.cur_grip

        self.pose_publisher.publish(pose_msg)
        self.grip_publisher.publish(grip_msg)

def main(args=None):
    rclpy.init(args=args)

    rt_target_pose = RtTargetPose()

    rclpy.spin(rt_target_pose)