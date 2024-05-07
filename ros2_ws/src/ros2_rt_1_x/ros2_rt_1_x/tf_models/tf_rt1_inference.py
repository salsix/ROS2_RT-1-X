import tensorflow as tf
import tensorflow_datasets as tfds
import rlds
from PIL import Image
import numpy as np
from tf_agents.policies import py_tf_eager_policy
import tf_agents
from tf_agents.trajectories import time_step as ts
from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow_hub as hub

import ros2_rt_1_x.camera as camera
import time

class RT1TensorflowInferer:
  
    def __init__(self):
        saved_model_path = './checkpoints/rt_1_x_tf_trained_for_002272480_step'

        self.tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
            model_path=saved_model_path,
            load_specs_from_pbtxt=True,
            use_tf_function=True
        )

        self.observation = tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(self.tfa_policy.time_step_spec.observation))

        self.embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')

        self.policy_state = self.tfa_policy.get_initial_state(batch_size=1)

        self.natural_language_instruction = "Put the banana in the pan."
        self.natural_language_embedding = self.embed([self.natural_language_instruction])[0]

        self.cam = camera.Camera()

    def run_inference(self):
        image = resize(Image.fromarray(self.cam.get_picture()).convert('RGB'))

        self.observation['image'] = image
        self.observation['natural_language_embedding'] = self.natural_language_embedding
        self.observation['natural_language_instruction'] = self.natural_language_instruction

        tfa_time_step = ts.transition(self.observation, reward=np.zeros((), dtype=np.float32))

        policy_step = self.tfa_policy.action(tfa_time_step, self.policy_state)
        action = policy_step.action
        self.policy_state = policy_step.state

        return rescale_for_umi(action)

def resize(image):
  image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
  image = tf.cast(image, tf.uint8)
  return image

def rescale_for_umi(action):
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

    abs_pos = 0.05
    abs_rot = 0.25

    umi_pos_x = rescale_dimension(pos_x, -abs_pos, abs_pos, -0.06, 0.06)
    umi_pos_y = rescale_dimension(pos_y, -abs_pos, abs_pos, 0.2, 0.7)
    umi_pos_z = rescale_dimension(pos_z, -abs_pos, abs_pos, 0.1, 0.7)
    umi_roll = rescale_dimension(roll, -abs_rot, abs_rot, 0.0, 90.0)
    umi_pitch = rescale_dimension(pitch, -abs_rot, abs_rot, 0.0, 90.0)
    umi_yaw = rescale_dimension(yaw, -abs_rot, abs_rot, -20.0, 200.0)
    umi_grip = rescale_dimension(grip, -0.05, 0.05, 0.02, 0.08)

    action["world_vector"] = [umi_pos_x, umi_pos_y, umi_pos_z]
    action["rotation_delta"] = [umi_roll, umi_pitch, umi_yaw]
    action["gripper_closedness_action"] = [umi_grip]

    return action


def rescale_dimension(
    value: float,
    low: float,
    high: float,
    post_scaling_min: float = -1.0,
    post_scaling_max: float = 1.0,
) -> float:
  """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
  return (value - low) / (high - low) * (
      post_scaling_max - post_scaling_min
  ) + post_scaling_min
