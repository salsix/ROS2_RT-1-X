"""Runs inference with a RT-1 model."""

import copy

from flax.training import checkpoints
from PIL import Image
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from collections import deque

import ros2_rt_1_x.models.rt1 as rt1
# import ros2_rt_1_x.camera as camera
import time
import cv2


class RT1Policy:
  """Runs inference with a RT-1 policy."""

  def __init__(
      self,
      checkpoint_path="/home/jonathan/Thesis/ROS2_RT-1-X/ros2_ws/src/ros2_rt_1_x/ros2_rt_1_x/checkpoints/rt_1_x_jax/checkpoint",
      model=rt1.RT1(),
      variables=None,
      seqlen=15,
      rng=None,
  ):
    """Initializes the policy.

    Args:
      checkpoint_path: A checkpoint point from which to load variables. Either
        this or variables must be provided.
      model: A nn.Module to use for the policy. Must match with the variables
        provided by checkpoint_path or variables.
      variables: If provided, will use variables instead of loading from
        checkpoint_path.
      seqlen: The history length to use for observations.
      rng: a jax.random.PRNGKey to use for the random number generator.
    """
    if not variables and not checkpoint_path:
      raise ValueError(
          'At least one of `variables` or `checkpoint_path` must be defined.'
      )
    self.model = model
    self._checkpoint_path = checkpoint_path
    self.seqlen = seqlen

    self._run_action_inference_jit = jax.jit(self._run_action_inference)

    if rng is None:
      self.rng = jax.random.PRNGKey(0)
    else:
      self.rng = rng

    if variables:
      self.variables = variables
    else:
      print("checkpoint_path: ", checkpoint_path)
      state_dict = checkpoints.restore_checkpoint(checkpoint_path, None)
      variables = {
          'params': state_dict['params'],
          'batch_stats': state_dict['batch_stats'],
      }
      self.variables = variables

  def _run_action_inference(self, observation, rng):
    """A jittable function for running inference."""

    # We add zero action tokens so that the shape is (seqlen, 11).
    # Note that in the vanilla RT-1 setup, where 
    # `include_prev_timesteps_actions=False`, the network will not use the
    # input tokens and instead uses zero action tokens, thereby not using the
    # action history. We still pass it in for simplicity.
    act_tokens = jnp.zeros((1, 6, 11))

    # Add a batch dim to the observation.
    batch_obs = jax.tree_map(lambda x: jnp.expand_dims(x, 0), observation)

    _, random_rng = jax.random.split(rng)

    output_logits = self.model.apply(
        self.variables,
        batch_obs,
        act=None,
        act_tokens=act_tokens,
        train=False,
        rngs={'random': random_rng},
    )

    time_step_tokens = (
        self.model.num_image_tokens + self.model.num_action_tokens
    )
    output_logits = jnp.reshape(
        output_logits, (1, self.seqlen, time_step_tokens, -1)
    )
    action_logits = output_logits[:, -1, ...]
    action_logits = action_logits[:, self.model.num_image_tokens - 1 : -1]

    action_logp = jax.nn.softmax(action_logits)
    action_token = jnp.argmax(action_logp, axis=-1)

    # Detokenize the full action sequence.
    detokenized = rt1.detokenize_action(
        action_token, self.model.vocab_size, self.model.world_vector_range
    )

    detokenized = jax.tree_map(lambda x: x[0], detokenized)

    return detokenized

  def action(self, observation):
    """Outputs the action given observation from the env."""
    # Assume obs has no batch dimensions.
    observation = copy.deepcopy(observation)

    # Jax does not support string types, so remove it from the dict if it
    # exists.
    if 'natural_language_instruction' in observation:
      del observation['natural_language_instruction']

    image = observation['image']
    # Resize using TF image resize to avoid any issues with using different
    # resize implementation, since we also use tf.image.resize in the data
    # pipeline. Also scale image to [0, 1].
    image = tf.image.resize(image, (300, 300)).numpy()
    image /= 255.0
    observation['image'] = image

    self.rng, rng = jax.random.split(self.rng)
    action = self._run_action_inference_jit(
        observation, rng
    )
    action = jax.device_get(action)

    # Use the base pose mode if the episode if the network outputs an invalid
    # `terminate_episode` action.
    if np.sum(action['terminate_episode']) == 0:
      action['terminate_episode'] = np.zeros_like(action['terminate_episode'])
      action['terminate_episode'][-1] = 1
    return action
  
# CUS: preprocess image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB').resize((300, 300))
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]
    return image

def preprocess_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert('RGB')
    image = tf.image.resize(image, (300, 300))
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]
    return image

class RT1Inferer:
  """Runs inference with the RT-1 model."""

  def __init__(self, natural_language_instruction):
    """Initializes the inferer."""
    # self.cam = camera.Camera()

    tf.config.experimental.set_visible_devices([], "GPU")

    self.sequence_length = 15
    self.num_action_tokens = 11
    self.layer_size = 256
    self.vocab_size = 512
    self.num_image_tokens = 81
    self.rt1x_model = rt1.RT1(
        num_image_tokens=self.num_image_tokens,
        num_action_tokens=self.num_action_tokens,
        layer_size=self.layer_size,
        vocab_size=self.vocab_size,
        # Use token learner to reduce tokens per image to 81.
        use_token_learner=True,
        # RT-1-X uses (-2.0, 2.0) instead of (-1.0, 1.0).
        world_vector_range=(-2.0, 2.0),
    )

    self.embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')

    self.policy = RT1Policy(
        model=self.rt1x_model,
        seqlen=self.sequence_length,
    )

    self.language_instruction = natural_language_instruction
    self.language_embedding = self.embed([self.language_instruction])[0]
    self.img_queue = deque(maxlen=15)

    print("RT1Inferer initialized.")

  def run_inference(self,img,i):
    """Outputs the action given observation from the env."""

    # image = Image.fromarray(cv2.cvtColor(self.cam.get_picture(), cv2.COLOR_BGRA2RGB)).convert('RGB')
        
    # # save image for debugging
    # if i == 0:
    #     image.save(f'./data/tmp_inference.png')

    # image = tf.image.resize_with_pad(image, target_width=300, target_height=300)
    # image = tf.cast(image, tf.uint8)

    image = preprocess_image(img)

    # if this is the first step (the queue is empty), fill all 15 places of the queue with the new image
    if len(self.img_queue) == 0:
      self.img_queue.extend([image for j in range(0,15)])
    else:
      self.img_queue.append(image)

    # # save all images from the queue to a folder
    # for i, img in enumerate(self.img_queue):
    #   img_path = f"./data/queue_img_{i}.jpg"
    #   Image.fromarray((img * 255).astype(np.uint8)).save(img_path)

    img_array = np.array(self.img_queue)

    embeddings = [jnp.ones((512,)) for _ in range(15)]
    embeddings[-1] = self.language_embedding

    # embeddings = [self.language_embedding for i in range(0,15)]


    observation = {
      'image': img_array,
      'natural_language_embedding': np.array(embeddings),
    }

    return self.policy.action(observation)


def main():
  image_path = "./data/test.jpg"
  image = Image.open(image_path)

  inferer = RT1Inferer()
  action = inferer.run_inference_step(image)
  print("action: ", action)

  # sequence_length = 15
  # num_action_tokens = 11
  # layer_size = 256
  # vocab_size = 512
  # num_image_tokens = 81
  # rt1x_model = rt1.RT1(
  #     num_image_tokens=num_image_tokens,
  #     num_action_tokens=num_action_tokens,
  #     layer_size=layer_size,
  #     vocab_size=vocab_size,
  #     # Use token learner to reduce tokens per image to 81.
  #     use_token_learner=True,
  #     # RT-1-X uses (-2.0, 2.0) instead of (-1.0, 1.0).
  #     world_vector_range=(-2.0, 2.0),
  # )
  # policy = RT1Policy(
  #     model=rt1x_model,
  #     seqlen=sequence_length,
  # )


  # # create array of 15 images
  # image_path = "./data/test.jpg"
  # images = np.array([load_and_preprocess_image(image_path) for i in range(0,15)])

  # #images = jnp.array(images)  # Convert to JAX array

  # # EMBEDDING:
  # embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')

  # natural_language_instruction = "Pick up yellow plush toy and place it on the white rectangle."
  # natural_language_embedding = embed([normalize_task_name(natural_language_instruction)])[0].numpy()

  # obs = {
  #   'image': images,
  #   'natural_language_embedding': np.array([natural_language_embedding for i in range(0,15)]),
  #   # 'natural_language_embedding': jnp.ones((15, 512)),
  # }

  # output_actions = policy.action(obs)
  # gripper_closedness_action = output_actions["gripper_closedness_action"]
  # rotation_delta = output_actions["rotation_delta"]
  # terminate_episode = output_actions["terminate_episode"]
  # world_vector = output_actions["world_vector"]

  # print("gripper_closedness_action: ", gripper_closedness_action)
  # print("rotation_delta: ", rotation_delta)
  # print("terminate_episode: ", terminate_episode)
  # print("world_vector: ", world_vector)
