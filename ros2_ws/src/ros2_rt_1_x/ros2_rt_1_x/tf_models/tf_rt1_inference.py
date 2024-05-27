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
import cv2

class RT1TensorflowInferer:
  
    def __init__(self, natural_language_instruction: str):
        saved_model_path = './checkpoints/rt_1_x_tf_trained_for_002272480_step'

        self.tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
            model_path=saved_model_path,
            load_specs_from_pbtxt=True,
            use_tf_function=True
        )

        self.observation = tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(self.tfa_policy.time_step_spec.observation))

        self.embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')

        self.policy_state = self.tfa_policy.get_initial_state(batch_size=1)

        self.natural_language_instruction = natural_language_instruction
        self.natural_language_embedding = self.embed([self.natural_language_instruction])[0]

        self.cam = camera.Camera()

    def run_inference(self,i):
        image = Image.fromarray(cv2.cvtColor(self.cam.get_picture(), cv2.COLOR_BGRA2RGB)).convert('RGB')
        
        # save image for debugging
        if i == 0:
            image.save(f'./data/tmp_inference.png')

        image = resize(image)
        # write image data to text file
        # np.set_printoptions(threshold=np.inf)
        # with open('./data/output_img', 'w') as f:
        #     f.write(str(type(image.numpy())))
        #     f.write(str(image.numpy()))


        # array of 15 of the same image
        # image = tf.convert_to_tensor(np.array([np.array(image) for _ in range(15)]))

        # image = resize(Image.open(f'/home/jonathan/Thesis/open_x_embodiment/imgs/bridge/{i}.png').convert('RGB'))
        # image = tf.convert_to_tensor(np.array(image))


        self.observation['image'] = image
        self.observation['natural_language_embedding'] = self.natural_language_embedding
        # self.observation['natural_language_instruction'] = self.natural_language_instruction

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

    print('BEFORE SCALE')
    print(f'pos_x: {pos_x}, pos_y: {pos_y}, pos_z: {pos_z}')
    print(f'roll: {roll}, pitch: {pitch}, yaw: {yaw}, grip: {grip}')
    print(f'gr: {gripper_closedness_action}')
    print(f'terminate: {terminate_episode}')

    # write to csv
    with open('./data/output_csv', 'a') as f:
        f.write(f'{pos_x},{pos_y},{pos_z},{roll},{pitch},{yaw},{grip}\n')

    abs_pos = 2.0
    abs_rot = np.pi

    # umi_pos_x = rescale_dimension(pos_x, -abs_pos, abs_pos, -0.6, 0.6)
    # umi_pos_y = rescale_dimension(pos_y, -abs_pos, abs_pos, 0.3, 0.8)
    # umi_pos_z = rescale_dimension(pos_z, -abs_pos, abs_pos, 0.1, 0.7)
    # umi_roll = rescale_dimension(roll, -abs_rot, abs_rot, 0.0, 90.0)
    # umi_pitch = rescale_dimension(pitch, -abs_rot, abs_rot, 0.0, 90.0)
    # umi_yaw = rescale_dimension(yaw, -abs_rot, abs_rot, -20.0, 200.0)
    # umi_grip = rescale_dimension(grip, -1, 1, 0.02, 0.08)

    umi_pos_x = rescale_dimension(pos_x, -abs_pos, abs_pos, -0.6, 0.6)
    umi_pos_y = rescale_dimension(pos_y, -abs_pos, abs_pos, -0.25, 0.25) # 0.5
    umi_pos_z = rescale_dimension(pos_z, -abs_pos, abs_pos, -0.3, 0.3) # 0.6
    umi_roll = rescale_dimension(roll, -abs_rot, abs_rot, -45.0, 45.0) # 90
    umi_pitch = rescale_dimension(pitch, -abs_rot, abs_rot, -45.0, 45.0) # 90
    umi_yaw = rescale_dimension(yaw, -abs_rot, abs_rot, -80.0, 80.0) # 220
    umi_grip = rescale_dimension(grip, -1, 1, -0.03, 0.03) # 0.06

    # print(f'2. pos_x: {umi_pos_x}, pos_y: {umi_pos_y}, pos_z: {umi_pos_z}')

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
    if value < low:
        print(f"VALUE BELOW LOW: {value} < {low}")
    if value > high:
        print(f"VALUE ABOVE HIGH: {value} > {high}")

    # val = value
    val = (value - low) / (high - low) * (
        post_scaling_max - post_scaling_min
    ) + post_scaling_min
    if val < post_scaling_min:
        print("VALUE BELOW MIN")
        return post_scaling_min
    if val > post_scaling_max:
        print("VALUE ABOVE MAX")
        return post_scaling_max
    return val
