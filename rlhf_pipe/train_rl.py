import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from chat_env import ChatEnvironment
from reward_function import reward_function

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the data
data_folder = '/data/datasets/mental_health_dialogues/'
train_data = pd.read_csv(f'{data_folder}train_dataset.csv')
val_data = pd.read_csv(f'{data_folder}val_dataset.csv')
samples = train_data['text'].tolist()

# Load the model and tokenizer
model_name = 'TheBloke/Llama-2-7B-Chat-GPTQ'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = "<pad>"
tokenizer.padding_side = "right"

env = make_vec_env(lambda: ChatEnvironment(model, tokenizer, samples, reward_function), n_envs=1)

ppo_model = PPO("MlpPolicy", env, verbose=1)
ppo_model.learn(total_timesteps=10000)
