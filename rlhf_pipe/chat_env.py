import gymnasium as gym
from gymnasium import spaces

import numpy as np
import random

import torch
from torch.nn import functional as F


class ChatEnvironment(gym.Env):
	"""A chat environment for a reinforcement learning model."""

	def __init__(self, rl_model, reward_model, tokenizer, max_seq_len, samples, reward_function):
		super(ChatEnvironment, self).__init__()
		self.rl_model = rl_model
		self.reward_model = reward_model
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.samples = samples
		self.current_sample_index = 0
		self.current_conversation_length = 0
		self.reward_function = reward_function
		self.action_space = spaces.Discrete(2)  # Binary action space
		self.observation_space = spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(1024,), dtype=np.int32)
		self.current_state = None

	def reset(self, **kwargs):
		"""Resets the environment for the next sample."""
		# Use self.current_sample_index or another method to select the next sample
		# set random seed
		seed = kwargs.get('seed', 42)
		random.seed(seed)
		rand_index = random.randint(0, len(self.samples) - 1)
		self.current_state = self.samples[rand_index]
		self.current_sample_index = (self.current_sample_index + 1) % len(self.samples)
		# Reset any other environment state as needed
		observation = self._next_observation()

		# Generate the info dictionary (can contain auxiliary information)
		info = self._info()

		return observation, info

	def _next_observation(self):
		"""Construct the next observation from the current conversation state."""
		self.current_conversation_length += 1
		output = self.tokenizer.encode(self.current_state, bos=True, eos=False)
		pad_length = self.max_seq_len - len(output["input_ids"])
		pad_token_id = 2
		if pad_length > 0:
			output["input_ids"] = F.pad(output["input_ids"], (0, pad_length), value=pad_token_id)

		# Convert the PyTorch tensor to a NumPy array
		observation = output["input_ids"].detach().cpu().numpy()

		return observation.squeeze()

	def _info(self):
		"""Provide any additional information about the current state."""
		# Placeholder - can return any additional information that may be useful for training
		return {}

	def step(self, action):
		"""Executes a step in the environment based on the action."""
		# Generate a response using your model. This is simplified; actual implementation will vary.
		# Example: action could determine response generation parameters
		with torch.no_grad():
			response = self.rl_model.generate(self.current_state, 512)

		# Calculate reward based on the generated response
		reward = self.reward_function(self.reward_model, self.tokenizer, self.current_state, response)

		# Update conversation state based on the reward
		if reward[0] > reward[1]:
			response = response1
		else:
			response = response2

		# Update the conversation state with the new response
		self.current_state += " " + response

		# Determine if the conversation/episode should end
		done = self.current_conversation_length >= 6

		# The observation for the next step is the updated conversation state
		observation = self._next_observation()

		return observation, reward, done, {}

	def render(self, mode='human', close=False):
		# For debugging or visualization, print the current conversation state
		if mode == 'human':
			print(self.current_state)
