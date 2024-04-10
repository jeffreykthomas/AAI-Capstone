import torch
from torch.nn import functional as F


def reward_function(reward_model, reward_tokenizer, rl_model_response):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# run two responses through the reward model, which predicts which response a human would prefer
	# the reward model is a classifier that predicts which of two responses is better
	# the reward model is trained on human feedback data
	# the reward model is a BERT model fine-tuned on a human feedback dataset

	# encode the response
	encoded_response = reward_tokenizer(rl_model_response, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

	# run the response through the reward model
	with torch.no_grad():
		output = reward_model(**encoded_response)

	# get the loss from the reward model
	reward = output.item()

	# the lower the loss, the better the response
	return reward
