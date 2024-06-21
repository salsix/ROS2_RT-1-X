import numpy as np
import time
import cv2
import tensorflow as tf
from PIL import Image
import copy

from geometry_msgs.msg import Pose
from std_msgs.msg import Float32

import ros2_rt_1_x.camera as camera

class EpisodeLogger:
    def __init__(self, initial_pose: Pose, initial_grip: Float32):
        tf.config.experimental.set_visible_devices([], "GPU")

        self.episode = []
        self.camera = camera.Camera()
        self.current_pose = copy.deepcopy(initial_pose)
        self.current_grip = copy.deepcopy(initial_grip)
        self.last_log_time = time.time()
        self.episode_length = 0
        self.episode_started = False

        self.natural_language_instruction = 'Pick up the yellow banana.'

    def log(self, new_pose, new_grip, terminate=False):
        self.episode.append({
            'image': self._take_picture(),
            'action': self._get_action(new_pose, new_grip, terminate),
            'language_instruction': self.natural_language_instruction,
            'state': self._get_state(terminate)
        })
        self.episode_length += 1
        print(f'Logged action {self.episode_length}.')
        self.last_log_time = time.time()
        self.current_pose = copy.deepcopy(new_pose)
        self.current_grip = copy.deepcopy(new_grip)

    def stop_and_save(self, filename):
        self.episode_started = False
        self.log(self.current_pose, self.current_grip, terminate=True)
        # np.save(f'./episodes/{filename}.npy', self.episode)
        np.save(f'./episodes/test_epi.npy', self.episode)

    def reset(self):
        self.episode = []

    def _take_picture(self):
        image = Image.fromarray(cv2.cvtColor(self.camera.get_picture(), cv2.COLOR_BGRA2RGB)).convert('RGB')
        image = tf.image.resize(image, (300, 300))
        # image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]
        image = np.asarray(image, dtype=np.uint8)
        return image
    
    def _get_action(self, new_pose, new_grip, terminate=False):
        # calculate the action between the current pose and the new pose
        x = new_pose.position.x - self.current_pose.position.x
        y = new_pose.position.y - self.current_pose.position.y
        z = new_pose.position.z - self.current_pose.position.z
        yaw = new_pose.orientation.x - self.current_pose.orientation.x
        pitch = new_pose.orientation.y - self.current_pose.orientation.y
        roll = new_pose.orientation.z - self.current_pose.orientation.z
        grip = new_grip.data - self.current_grip.data
        terminate_episode = int(terminate)

        print(f'Current pose: x={self.current_pose.position.x}, y={self.current_pose.position.y}, z={self.current_pose.position.z}, yaw={self.current_pose.orientation.x}, pitch={self.current_pose.orientation.y}, roll={self.current_pose.orientation.z}, grip={self.current_grip.data}')
        print(f'New pose: x={new_pose.position.x}, y={new_pose.position.y}, z={new_pose.position.z}, yaw={new_pose.orientation.x}, pitch={new_pose.orientation.y}, roll={new_pose.orientation.z}, grip={new_grip.data}')

        print(f'Action: x={x}, y={y}, z={z}, yaw={yaw}, pitch={pitch}, roll={roll}, grip={grip}, terminate={terminate_episode}')

        return [x, y, z, yaw, pitch, roll, grip, terminate_episode]
    
    def _get_state(self, terminate=False):
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        z = self.current_pose.position.z
        yaw = self.current_pose.orientation.x
        pitch = self.current_pose.orientation.y
        roll = self.current_pose.orientation.z
        grip = self.current_grip.data
        terminate_episode = int(terminate)
        return [x, y, z, yaw, pitch, roll, grip, terminate_episode]
        
    
