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


class RtTargetPose(Node):

    def __init__(self):

        tf.config.experimental.set_visible_devices([], "GPU")

        super().__init__('rt_target_pose_publisher')
        self.img_converter = cv_bridge.CvBridge()

        self.pose_publisher = self.create_publisher(Pose, 'target_pose', 10)
        self.grip_publisher = self.create_publisher(Float32, 'target_grip', 10)

        # listener for input images
        self.subscription = self.create_subscription(Image, 'rt_input_image', self.image_listener_callback, 10)

        # self.rt1_inferer = rt1_inference.RT1Inferer()
        self.camera = camera.Camera()

        self.natural_language_instruction = "Place the can to the left of the pot."
        # self.rt1_tf_inferer = tf_models.RT1TensorflowInferer(self.natural_language_instruction)
        self.rt1_jax_inferer = jax_models.RT1Inferer(self.natural_language_instruction)

        self.inference_interval = 2

        self.cur_x = 0.0
        self.cur_y = 0.5
        self.cur_z = 0.3
        self.cur_roll = 0.0
        self.cur_pitch = 90.0
        self.cur_yaw = 0.0
        self.cur_grip = 0.02

        self.pose_history = []

        # self.camera = camera.Camera()

        self.run_inference()

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
        self.init_target_pose()
        time.sleep(5)
        print('TOOK ON INIT POSE. RUNNING INFERENCE...')
        actions = []
        steps = 0
        while steps < 38:
            image = self.camera.get_picture()

            image = PIL.Image.open(f'/home/jonathan/Thesis/open_x_embodiment/imgs/bridge/{steps+1}.png')

            # # self.store_image(image)
            # image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            # action = self.rt1_inferer.run_inference_step(image)
            # self.get_logger().info(f'Action: {action}')
            # self.publish_target_pose(action)

            # act = self.rt1_tf_inferer.run_inference(steps)
            act = self.rt1_jax_inferer.run_inference(image,steps)

            self.publish_target_pose_deltas(act)
            actions.append(act)

            print(hash(str(act)))

            # time.sleep(self.inference_interval)
            steps += 1
            print(f'Step {steps} done.')
        print('DONE RUNNING INFERENCE.')
        # self.draw_plots(actions)
        # self.draw_pose_history_plots()
        self.draw_bridge_example_plots(actions)

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
        axs[2,1].set_ylim([0, 1])

        # subplots [2,2] is the image ./data/tmp_inference.png
        axs[2,2].imshow(plt.imread('./data/tmp_inference.png'))
        axs[2,2].axis('off')

        # dont show subplot [2,2]
        # axs[2,2].axis('off')

        # print date and time of inference
        # fig.suptitle(f'\"{self.natural_language_instruction}", Frequency = {round(1/self.inference_interval,1)}s')
        # second line
        fig.text(0.5, 0.97, f'"{self.natural_language_instruction}", Frequency = {round(1/self.inference_interval,1)}Hz', ha='center')
        fig.text(0.5, 0.01, f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}', ha='center')

        filename = f'inference_{int(time.time())}'

        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        fig.savefig(f'./data/plots/{filename}.png', dpi=300)

        with open(f'./data/plots/{filename}.csv', 'w') as f:
            f.write('X,Y,Z,Roll,Pitch,Yaw,Gripper,Terminate,BaseDispl,BaseVertRot\n')
            for a in actions:
                f.write(f'{a["world_vector"][0]},{a["world_vector"][1]},{a["world_vector"][2]},{a["rotation_delta"][0]},{a["rotation_delta"][1]},{a["rotation_delta"][2]},{a["gripper_closedness_action"][0]},{a["terminate_episode"]},{a["base_displacement_vector"]},{a["base_displacement_vertical_rotation"]}\n')
            
    def draw_pose_history_plots(self):
        fig, axs = plt.subplots(3, 3)

        # set plot line color to green


        axs[0,0].plot([a[0] for a in self.pose_history], color='green')
        axs[0,0].set_title('X')
        axs[0,0].set_ylim([-0.6, 0.6])
        axs[0,1].plot([a[1] for a in self.pose_history], color='green')
        axs[0,1].set_title('Y')
        axs[0,1].set_ylim([0.3, 0.8])
        axs[0,2].plot([a[2] for a in self.pose_history], color='green')
        axs[0,2].set_title('Z')
        axs[0,2].set_ylim([0.1, 0.7])

        axs[1,0].plot([a[3] for a in self.pose_history], color='green')
        axs[1,0].set_title('Roll')
        axs[1,0].set_ylim([0, 90])
        axs[1,1].plot([a[4] for a in self.pose_history], color='green')
        axs[1,1].set_title('Pitch')
        axs[1,1].set_ylim([0, 90])
        axs[1,2].plot([a[5] for a in self.pose_history], color='green')
        axs[1,2].set_title('Yaw')
        axs[1,2].set_ylim([-20, 200])

        axs[2,0].plot([a[6] for a in self.pose_history], color='green')
        axs[2,0].set_title('Gripper')
        axs[2,0].set_ylim([0.02, 0.08])
        axs[2,1].plot([a[7][0] for a in self.pose_history], color='green')
        axs[2,1].set_title('Terminate')
        axs[2,1].set_ylim([0, 1])

        axs[2,2].imshow(plt.imread('./data/tmp_inference.png'))
        axs[2,2].axis('off')

        # print date and time of inference
        # fig.suptitle(f'\"{self.natural_language_instruction}", Frequency = {round(1/self.inference_interval,1)}s')
        # second line
        fig.text(0.5, 0.97, f'"{self.natural_language_instruction}", Frequency = {round(1/self.inference_interval,1)}Hz', ha='center')
        fig.text(0.5, 0.01, f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}', ha='center')

        filename = f'inference_pos_{int(time.time())}'

        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        fig.savefig(f'./data/plots/{filename}.png', dpi=300)

        with open(f'./data/plots/{filename}.csv', 'w') as f:
            f.write('X,Y,Z,Roll,Pitch,Yaw,Gripper\n')
            for a in self.pose_history:
                f.write(f'{a[0]},{a[1]},{a[2]},{a[3]},{a[4]},{a[5]},{a[6]}\n')

        # draw heatmap
        
        # get all x and y values
        x = [a[0] for a in self.pose_history]
        y = [a[1] for a in self.pose_history]

        # create heatmap
        # Create a 2D histogram (heatmap)
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=(50, 50))

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Density')
        plt.title('Heatmap of 2D Coordinates')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        # Save the heatmap as a PNG file
        plt.savefig('heatmap.png')

        # Show the plot (optional)

    def draw_bridge_example_plots(self, act):
        # show the raw model output
        data = act

        fig, axs = plt.subplots(1, 10)

        fig.set_size_inches([45, 5])
        fig.subplots_adjust(left=0.03, right=0.97)

        print(data)

        axs[0].plot([a['terminate_episode'][0] for a in data], color='red')
        axs[0].set_title('terminate_episode_0')
        axs[1].plot([a['terminate_episode'][1] for a in data], color='red')
        axs[1].set_title('terminate_episode_1')
        axs[2].plot([a['terminate_episode'][2] for a in data], color='red')
        axs[2].set_title('terminate_episode_2')
        axs[3].plot([a['world_vector'][0] for a in data], color='red')
        axs[3].set_title('world_vector_0')
        axs[4].plot([a['world_vector'][1] for a in data], color='red')
        axs[4].set_title('world_vector_1')
        axs[5].plot([a['world_vector'][2] for a in data], color='red')
        axs[5].set_title('world_vector_2')
        axs[6].plot([a['rotation_delta'][0] for a in data], color='red')
        axs[6].set_title('rotation_delta_0')
        axs[7].plot([a['rotation_delta'][1] for a in data], color='red')
        axs[7].set_title('rotation_delta_1')
        axs[8].plot([a['rotation_delta'][2] for a in data], color='red')
        axs[8].set_title('rotation_delta_2')
        axs[9].plot([a['gripper_closedness_action'][0] for a in data], color='red')
        axs[9].set_title('gripper_closedness_action_0')

        filename = f'inference_raw_{int(time.time())}'

        # plt.subplots_adjust(wspace=0.3, hspace=0.5)
        fig.savefig(f'./data/plots/{filename}.png')



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

    def publish_target_pose_deltas(self, action):
        # to see if something changed, print a hash of the action
        # print("ACTION: " + str(hash(str(action))))

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
        self.cur_grip += float(gripper_closedness_action[0])

        self.cur_x = min(max(self.cur_x, -0.5), 0.5)
        self.cur_y = min(max(self.cur_y, 0.3), 0.8)
        self.cur_z = min(max(self.cur_z, 0.0), 0.4)
        self.cur_roll = min(max(self.cur_roll, 0.0), 90.0)
        self.cur_pitch = min(max(self.cur_pitch, 0.0), 90.0)
        self.cur_yaw = min(max(self.cur_yaw, -10.0), 170.0)
        self.cur_grip = min(max(self.cur_grip, 0.02), 0.08)

        # print(f'Publishing target pose: {pos_x}, {pos_y}, {pos_z}, {roll}, {pitch}, {yaw}, {grip}')
        self.get_logger().info(f'Publishing target pose and grip...')
        # self.get_logger().info(f'pos_x: {pos_x}, pos_y: {pos_y}, pos_z: {pos_z}, roll: {roll}, pitch: {pitch}, yaw: {yaw}, grip: {grip}')

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