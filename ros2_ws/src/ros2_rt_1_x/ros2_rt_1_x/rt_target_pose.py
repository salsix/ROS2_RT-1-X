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

import ros2_rt_1_x.models.rt1_inference as rt1_inference
import ros2_rt_1_x.camera as camera
import ros2_rt_1_x.tf_models.tf_rt1_inference as tf_models


class RtTargetPose(Node):

    def __init__(self):
        super().__init__('rt_target_pose_publisher')
        self.img_converter = cv_bridge.CvBridge()

        self.pose_publisher = self.create_publisher(Pose, 'target_pose', 10)
        self.grip_publisher = self.create_publisher(Float32, 'target_grip', 10)

        # listener for input images
        self.subscription = self.create_subscription(Image, 'rt_input_image', self.image_listener_callback, 10)

        # self.rt1_inferer = rt1_inference.RT1Inferer()
        # self.camera = camera.Camera()

        self.natural_language_instruction = "Place the can to the left of the pot."
        self.rt1_tf_inferer = tf_models.RT1TensorflowInferer(self.natural_language_instruction)

        self.inference_interval = 0.2
        self.run_inference()

    def test_pose_publish(self):
        msg = Pose()
        msg.position.x = 1.0
        msg.position.y = 2.0
        msg.position.z = 3.0
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        msg.orientation.w = 1.0
        self.pose_publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg)

    def image_listener_callback(self, msg):
        # self.store_image_msg(msg)
        self.get_logger().info(f'Received image.')

    # for debugging: store image from ros image message to disk
    def store_image_msg(self, msg):
        cv_image = self.img_converter.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        filename = f'./data/received/test_{int(time.time())}.png'
        cv2.imwrite(filename, cv_image)
        self.get_logger().info(f'Stored image to {filename}')

    def store_image(self, image):
        filename = f'./data/received/test_{int(time.time())}.png'
        cv2.imwrite(filename, image)
        self.get_logger().info(f'Stored image to {filename}')

    def run_inference(self):
        actions = []
        steps = 0
        while steps < 38:
            # image = self.camera.get_picture()
            # # self.store_image(image)
            # image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            # action = self.rt1_inferer.run_inference_step(image)
            # self.get_logger().info(f'Action: {action}')
            # self.publish_target_pose(action)
            act = self.rt1_tf_inferer.run_inference(steps+1)
            self.publish_target_pose(act)
            actions.append(act)
            time.sleep(self.inference_interval)
            steps += 1
            print(f'Step {steps} done.')
        print('DONE RUNNING INFERENCE.')
        self.draw_plots(actions)

    def draw_plots(self, actions):
        fig, axs = plt.subplots(3, 3)
        axs[0,0].plot([a["world_vector"][0] for a in actions])
        axs[0,0].set_title('X')
        axs[0,0].set_ylim([-0.6, 0.6])
        axs[0,1].plot([a["world_vector"][1] for a in actions])
        axs[0,1].set_title('Y')
        axs[0,1].set_ylim([0.3, 0.8])
        axs[0,2].plot([a["world_vector"][2] for a in actions])
        axs[0,2].set_title('Z')
        axs[0,2].set_ylim([0.1, 0.7])

        axs[1,0].plot([a["rotation_delta"][0] for a in actions])
        axs[1,0].set_title('Roll')
        axs[1,0].set_ylim([0, 90])
        axs[1,1].plot([a["rotation_delta"][1] for a in actions])
        axs[1,1].set_title('Pitch')
        axs[1,1].set_ylim([0, 90])
        axs[1,2].plot([a["rotation_delta"][2] for a in actions])
        axs[1,2].set_title('Yaw')
        axs[1,2].set_ylim([-20, 200])
        
        axs[2,0].plot([a["gripper_closedness_action"][0] for a in actions])
        axs[2,0].set_title('Gripper')
        axs[2,0].set_ylim([0.02, 0.08])
        axs[2,1].plot([a["terminate_episode"][0] for a in actions])
        axs[2,1].set_title('Terminate')

        # dont show subplot [2,2]
        axs[2,2].axis('off')

        # print date and time of inference
        # fig.suptitle(f'\"{self.natural_language_instruction}", Frequency = {round(1/self.inference_interval,1)}s')
        # second line
        fig.text(0.5, 0.97, f'"{self.natural_language_instruction}", Frequency = {round(1/self.inference_interval,1)}Hz', ha='center')
        fig.text(0.5, 0.01, f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}', ha='center')

        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        fig.savefig(f'./data/plots/inference_{int(time.time())}.png')

        with open(f'./data/plots/inference_{int(time.time())}.csv', 'w') as f:
            f.write('X,Y,Z,Roll,Pitch,Yaw,Gripper,Terminate\n')
            for a in actions:
                f.write(f'{a["world_vector"][0]},{a["world_vector"][1]},{a["world_vector"][2]},{a["rotation_delta"][0]},{a["rotation_delta"][1]},{a["rotation_delta"][2]},{a["gripper_closedness_action"][0]},{a["terminate_episode"][0]}\n')
            


    def publish_target_pose(self, action):
        # to see if something changed, print a hash of the action
        # print("ACTION: " + str(hash(str(action))))

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

def main(args=None):
    rclpy.init(args=args)

    rt_target_pose = RtTargetPose()

    rclpy.spin(rt_target_pose)

    #while(True):
        #rt_target_pose.test_pose_publish()