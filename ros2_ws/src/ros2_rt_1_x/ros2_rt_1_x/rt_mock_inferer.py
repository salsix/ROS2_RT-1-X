import rclpy
from rclpy.node import Node
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
import tensorflow as tf
import copy

import ros2_rt_1_x.models.rt1_inference as jax_models
import ros2_rt_1_x.output_logging as output_log


class RtMockInferer(Node):
    def __init__(self):
        tf.config.experimental.set_visible_devices([], "GPU")

        super().__init__('rt_mock_inferer')

        self.natural_language_instruction = "Pick up the yellow banana."

        self.epi_steps_iterator = self.load_dataset()
        
        self.rt1_inferer = jax_models.RT1Inferer(self.natural_language_instruction)

        self.run_inference()

    def load_dataset(self):
        dataset_builder = tfds.builder_from_directory('/home/jonathan/tensorflow_datasets/episodes/1.0.0')
        # dataset_builder = tfds.builder_from_directory(builder_dir='gs://gresearch/robotics/bridge/0.1.0/')
        dataset = dataset_builder.as_dataset(split='train[:10]')
        iter_dataset = iter(dataset)
        first_episode = next(iter_dataset)
        episode_steps = first_episode['steps']
        step_iterator = iter(episode_steps)
        return step_iterator

    def run_inference(self):
        actions = []
        ground_truth_actions = []

        for index, step in enumerate(self.epi_steps_iterator):

            image = Image.fromarray(np.array(step['observation']['image']))

            # print(step['action'])

            # # save image for debugging
            # image.save(f'./data/HALLOHALLO.png')

            act = self.rt1_inferer.run_umi_mock_inference(image, index)
            act = self.scale_back_to_umi(act)
            actions.append(act)
            print(act)

            # transform ground truth action to match the output of the model, so it can be plotted

            step['action']['gripper_closedness_action'] = [step['action']['gripper_closedness_action']]

            # if step['action']['open_gripper'] == True:
            #     step['action']['gripper_closedness_action'] = [-1.0]
            # else:
            #     step['action']['gripper_closedness_action'] = [1.0]

            if step['action']['terminate_episode'] == 0.0:
                step['action']['terminate_episode'] = [0,1,0]
            else:
                step['action']['terminate_episode'] = [1,0,0]

            ground_truth_actions.append(step['action'])

            print(f'Inference step {index+1}')

        output_log.draw_compare_model_output(actions, ground_truth_actions, 'umi_mock_inference')

    def scale_back_to_umi(self, action):
        # in the finetuning code, we scale our actions to the UMI range,
        # which is 2;2 for coordinates and pi/2 for rotations. We need
        # to do the opposite of that here, so we get back to UMI range.
        # The world vector as existed in the dataset on disk ranges from -0.01 to 0.01

        scaled_action = copy.deepcopy(action)
        
        # We scale by 200.0 so that the action better spans the limit of the
        # world_vector action, from -2.0 to 2.0.
        # scaled_action['world_vector'] = [i / 200.0 for i in action['world_vector']]

        # Similarly, the rotation_delta in the dataset on disk ranges from -2.0 to
        # 2.0
        # We scale by 0.79 so that the rotation_delta almost spans the limit of
        # rotation_delta, from -pi/2 to pi/2.
        # scaled_action['rotation_delta'] = [i / 0.79 for i in action['rotation_delta']]

        # scale grip from space 0.02 to 0.08, to -1 to 1, with 0.02 being 1, and 0.08 being -1
        # scaled_action['gripper_closedness_action'] = (
        #     self.rescale_value(
        #         value=action['gripper_closedness_action'],
        #         value_low=1.0,
        #         value_high=-1.0,
        #         output_low=0.02,
        #         output_high=0.08,
        #     )
        # )

        scaled_action['world_vector'][0] = self.rescale_value(
            value=action['world_vector'][0],
            output_low=-0.05,
            output_high=0.05,
            value_low=-1.75,
            value_high=1.75,
        )

        scaled_action['world_vector'][1] = self.rescale_value(
            value=action['world_vector'][1],
            output_low=-0.05,
            output_high=0.05,
            value_low=-1.75,
            value_high=1.75,
        )

        scaled_action['world_vector'][2] = self.rescale_value(
            value=action['world_vector'][2],
            output_low=-0.05,
            output_high=0.05,
            value_low=-1.75,
            value_high=1.75,
        )

        scaled_action['rotation_delta'][0] = self.rescale_value(
            value=action['rotation_delta'][0],
            output_low=-0.25,
            output_high=0.25,
            value_low=-1.4,
            value_high=1.4,
        )

        scaled_action['rotation_delta'][1] = self.rescale_value(
            value=action['rotation_delta'][1],
            output_low=-0.25,
            output_high=0.25,
            value_low=-1.4,
            value_high=1.4,
        )

        scaled_action['rotation_delta'][2] = self.rescale_value(
            value=action['rotation_delta'][2],
            output_low=-0.25,
            output_high=0.25,
            value_low=-1.4,
            value_high=1.4,
        )
    


        return scaled_action
        
    def rescale_value(self, value, value_low, value_high, output_low, output_high):
        """Rescale value from [value_low, value_high] to [output_low, output_high]."""
        return (value - value_low) / (value_high - value_low) * (
            output_high - output_low
        ) + output_low



def main(args=None):
    rclpy.init(args=args)

    rt_mock_inferer = RtMockInferer()

    rclpy.spin(rt_mock_inferer)
