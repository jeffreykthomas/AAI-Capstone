import torch
from torch.nn import functional as F


def score_response(reward_model, tokenizer, prompt, rl_model_response1, rl_model_response2):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# run two responses through the reward model, which predicts which response a human would prefer
	# the reward model is a classifier that predicts which of two responses is better
	# the reward model is trained on human feedback data
	# the reward model is a BERT model fine-tuned on a human feedback dataset

	# encode the prompt and responses
	encoded_prompt = tokenizer.encode(prompt, return_tensors='pt').to(device)
	encoded_response1 = tokenizer.encode(rl_model_response1, return_tensors='pt').to(device)
	encoded_response2 = tokenizer.encode(rl_model_response2, return_tensors='pt').to(device)

	# pad the responses to the same length
	max_length = max(encoded_response1.shape[1], encoded_response2.shape[1])
	encoded_response1 = F.pad(encoded_response1, (0, max_length - encoded_response1.shape[1]), value=0)
	encoded_response2 = F.pad(encoded_response2, (0, max_length - encoded_response2.shape[1]), value=0)

	# run the prompt and responses through the reward model
	output1 = reward_model(input_ids=encoded_prompt, labels=torch.tensor([0]).to(device), encoder_hidden_states=None, decoder_input_ids=encoded_response1)
	output2 = reward_model(input_ids=encoded_prompt, labels=torch.tensor([1]).to(device), encoder_hidden_states=None, decoder_input_ids=encoded_response2)

	# get the loss from the reward model
	loss1 = output1.loss.item()
	loss2 = output2.loss.item()

	# the reward is the negative of the loss
	# the lower the loss, the better the response
	return loss1, loss2

def reward_function(model, tokenizer, prompt, response1, response2):
	score1 = score_response(model, tokenizer, prompt, response1)
	score2 = score_response(model, tokenizer, prompt, response2)

	# For example, reward the response with the lower score
	if score1 < score2:
		return 1, 0  # Response 1 is better
	else:
		return 0, 1  # Response 2 is better
