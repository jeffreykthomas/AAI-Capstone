import os
from typing import Callable, List
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, evaluate_policy
from gymnasium import Env
from .chat_env import ChatEnvironment
from .reward_function import reward_function

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama import model as llama_model
from llama.tokenizer import Tokenizer

# Load the data
data_folder = '/data/datasets/mental_health_dialogues/'
train_data = pd.read_csv(f'{data_folder}train_dataset.csv')
val_data = pd.read_csv(f'{data_folder}val_dataset.csv')
train_samples = train_data['text'].tolist()
val_samples = val_data['text'].tolist()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the model and tokenizer
ckpt_path = '/data/models/llama_mh_dialogues/mh_fine_tune.pt'
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model']
model_config = checkpoint['model_config']
rl_model = llama_model.Transformer(model_config)

unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
	if k.startswith(unwanted_prefix):
		state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

rl_model.load_state_dict(state_dict)
rl_model = rl_model.to(device)
rl_model = torch.compile(rl_model)

tokenizer = Tokenizer('llama/models/tokenizer.model')
tokenizer.pad_token = "<pad>"
max_seq_len = 1024

# Load the reward model
model_name = 'meta-llama/Llama-2-7b-chat-hf'
cache_dir = '/data/hf_home/hub'
bnb_config = {
	'load_in_4bit': True,
	'bnb_4bit_compute_dtype': torch.bfloat16,
	'bnb_4bit_quant_type': 'nf4',
	'bnb_4bit_use_double_quant': True
}
reward_model = AutoModelForCausalLM.from_pretrained(
	model_name,
	device_map='auto',
	quantization_config=bnb_config,
	cache_dir=cache_dir)


def create_chat_env(samples: List[str]) -> Callable[[], Env]:
	def _init() -> Env:
		return ChatEnvironment(
			rl_model=rl_model,
			reward_model=reward_model,
			tokenizer=tokenizer,
			max_seq_len=max_seq_len,
			samples=samples,
			reward_function=reward_function
		)

	return _init


class EvalCallback(BaseCallback):
	def __init__(self, eval_env, eval_freq=1000, verbose=1):
		super(EvalCallback, self).__init__(verbose)
		self.eval_env = eval_env
		self.eval_freq = eval_freq

	def _on_step(self) -> bool:
		if self.n_calls % self.eval_freq == 0:
			mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=10)
			print(f"Step: {self.n_calls}, Mean reward: {mean_reward}, Std Reward: {std_reward}")
		return True


# Create the environment
train_env = make_vec_env(create_chat_env(train_samples), n_envs=1)
val_env = make_vec_env(create_chat_env(val_samples), n_envs=1)

# Initialize the callback and pass it to the learn method
eval_callback = EvalCallback(val_env, eval_freq=1000)

ppo_model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log='./ppo_chatbot_tensorboard/')
ppo_model.learn(total_timesteps=10000, callback=eval_callback)

# Save dir
save_dir = '/data/models/llama_ppo/'
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
# Save the entire PPO model
ppo_model.save(os.path.join(save_dir, 'ppo_chatbot_model'))

# Save the underlying PyTorch model's state dictionary
torch.save(ppo_model.policy.state_dict(), f'{save_dir}model_state_dict.pt')
