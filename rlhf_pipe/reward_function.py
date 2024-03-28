import torch
from torch.nn import functional as F


def score_response(reward_model, tokenizer, instruction_prompt, prompt, rl_model_response):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	inputs = tokenizer(instruction_prompt + prompt, return_tensors="pt")
	inputs = {k: v.to(device) for k, v in inputs.items()}
	with torch.no_grad():
		reward_outputs = reward_model.generate(
			**inputs,
			temperature=0.7,
			do_sample=True,
			top_p=0.95,
			top_k=40,
			max_new_tokens=100)
	# Length of the instruction prompt to be removed
	instruction_prompt_length = len(instruction_prompt)
	# Perform slicing to remove the instruction prompt part
	# This maintains the tensor structure and utilizes GPU if reward_outputs is on a GPU
	reward_minus_instruction = reward_outputs[:, instruction_prompt_length:]

	loss = F.cross_entropy(reward_minus_instruction, rl_model_response)

	return loss


def reward_function(model, tokenizer, prompt, response1, response2):
	score1 = score_response(model, tokenizer, prompt, response1)
	score2 = score_response(model, tokenizer, prompt, response2)

	# For example, reward the response with the lower score
	if score1 < score2:
		return 1, 0  # Response 1 is better
	else:
		return 0, 1  # Response 2 is better
