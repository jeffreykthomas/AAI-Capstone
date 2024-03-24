import torch
import torch.nn as nn
from transformers import BertModel

import torch.optim as optim


class RewardModel(nn.Module):
	def __init__(self):
		super(RewardModel, self).__init__()
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.comparison_layer = nn.Linear(768, 1)  # Linear layer to score each response

	def forward(self, prompt, responses):
		prompt_embedding = self.bert(prompt)['pooler_output']
		response_embeddings = [self.bert(response)['pooler_output'] for response in responses]

		# Element-wise multiplication
		combined_embeddings = [prompt_embedding * resp_emb for resp_emb in response_embeddings]

		# Score each response
		scores = [self.comparison_layer(comb_emb) for comb_emb in combined_embeddings]

		return scores


model = RewardModel()
num_epochs = 10
dataset = [(prompt, responses, preferred_response), ...]  # List of tuples

optimizer = optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
	for batch in dataset:
		optimizer.zero_grad()

		prompt, responses, preferred_response = batch
		scores = model(prompt, responses)  # Get scores for each response

		scores_tensor = torch.stack(scores).squeeze()  # Convert scores to a tensor
		loss = criterion(scores_tensor, preferred_response)  # Calculate loss

		loss.backward()  # Backpropagation
		optimizer.step()  # Update model weights

	print(f"Epoch {epoch}, Loss: {loss.item()}")
