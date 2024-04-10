import gymnasium as gym
from gymnasium import spaces

import numpy as np
import random

import torch
from torch.nn import functional as F


class ChatEnvironment(gym.Env):
    """A simplified chat environment for training an RL model based on initial responses."""

    def __init__(self, rl_model, rl_tokenizer, reward_model, reward_tok, reward_function, max_seq_len, samples):
        super(ChatEnvironment, self).__init__()
        self.rl_model = rl_model
        self.rl_tokenizer = rl_tokenizer
        self.reward_function = reward_function
        self.reward_model = reward_model
        self.reward_tok = reward_tok
        self.max_seq_len = max_seq_len
        self.samples = samples
        self.observation_space = spaces.Box(low=0, high=np.iinfo(np.int32).max, shape=(1024,), dtype=np.int32)

    def reset(self):
        """Resets the environment to start with a new sample."""
        # Select a new sample to start the episode
        self.current_state = random.choice(self.samples)
        # Prepare the initial observation
        observation = self._encode_observation(self.current_state)
        return observation

    def _encode_observation(self, state):
        """Encodes the current state into a format suitable for the model."""
        output = self.rl_tokenizer.encode(state, bos=True, eos=False)
        output = torch.tensor(output, dtype=torch.long).unsqueeze(0).detach().cpu().numpy()
        return output

    def step(self, _):
        """Generates a response, evaluates it, and ends the episode."""
        # Generate a response for the current state
        with torch.no_grad():
            response = self.rl_model.generate(self.current_state, max_length=512)[0]  # Adjust based on your model's method

        decoded_response = self.rl_tokenizer.decode(response)
        # Evaluate the response using the reward model
        reward = self.reward_function(self.reward_model, self.reward_tok, decoded_response)  # Adjust based on your reward model's method

        # Since we're focusing on the first interaction, the episode ends after one step
        done = True
        observation = self._encode_observation(self.current_state + " " + response)  # Next state, not used here but for formality

        return observation, reward, done, {}

    def render(self, mode='human', close=False):
        """Optionally, for debugging or visualization, print the current state."""
        if mode == 'human':
            print(self.current_state)
