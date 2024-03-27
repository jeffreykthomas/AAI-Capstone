import torch


def score_response(model, tokenizer, prompt, response):
	inputs = tokenizer(prompt + response, return_tensors="pt")
	with torch.no_grad():
		outputs = model(**inputs, labels=inputs["input_ids"])
	return outputs.loss.item()


def reward_function(model, tokenizer, prompt, response1, response2):
	score1 = score_response(model, tokenizer, prompt, response1)
	score2 = score_response(model, tokenizer, prompt, response2)

	# For example, reward the response with the lower score
	if score1 < score2:
		return 1, 0  # Response 1 is better
	else:
		return 0, 1  # Response 2 is better
