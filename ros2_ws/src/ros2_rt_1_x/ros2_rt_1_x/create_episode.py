import numpy as np
import time
import cv2
import tensorflow as tf
from PIL import Image

from geometry_msgs.msg import Pose
from std_msgs.msg import Float32

import ros2_rt_1_x.camera as camera

class EpisodeLogger:
    def __init__(self, initial_pose: Pose, initial_grip: Float32):
        tf.config.experimental.set_visible_devices([], "GPU")

        self.episode = []
        self.filename = f'epi_{str(int(time.time()))}'
        self.camera = camera.Camera()
        self.current_pose = initial_pose
        self.current_grip = initial_grip
        self.last_log_time = time.time()
        self.episode_length = 0

        self.natural_language_instruction = 'Move the robot to the target pose.'

    def log(self, new_pose, new_grip, terminate=False):
        self.episode.append({
            'image': self._take_picture(),
            'action': self._get_action(new_pose, new_grip, terminate),
            'language_instruction': self.natural_language_instruction,
        })
        self.episode_length += 1
        print(f'Logged action {self.episode_length}.')
        self.last_log_time = time.time()

    def save(self):
        self.log(self.current_pose, self.current_grip, terminate=True)
        np.save(f'./episodes/{self.filename}.npy', self.episode)

    def reset(self):
        self.episode = []

    def _take_picture(self):
        image = Image.fromarray(cv2.cvtColor(self.camera.get_picture(), cv2.COLOR_BGRA2RGB)).convert('RGB')
        image = tf.image.resize(image, (300, 300))
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]
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
        return [x, y, z, yaw, pitch, roll, grip, terminate_episode]
        
    
