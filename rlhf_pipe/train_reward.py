from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from sklearn.model_selection import train_test_split
import transformers

num_epochs = 5
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transformers.logging.set_verbosity_error()

# Use a service account
cred = credentials.Certificate('rlhf_pipe/jt-designs-79-firebase-adminsdk-r3inm-6192f78083.json')
firebase_admin.initialize_app(cred)

db = firestore.client()


def preprocess_text(text):
	# Split the text into parts based on the <user> and <agent> tags
	parts = text.split('<agent>')
	user_part = parts[0].split('<user>')[-1].strip()  # Extracts the user's message
	agent_part = parts[1].split('<user>')[0].strip() if len(parts) > 1 else ""  # Extracts the agent's message

	# Reconstruct the text with just the user's prompt and the agent's response
	preprocessed_text = f"<user> {user_part} <agent> {agent_part}"
	return preprocessed_text


def fetch_and_format_data_from_firestore():
	# Assuming db is your Firestore client
	prompts_ref = db.collection('prompts')
	docs = prompts_ref.stream()

	formatted_data = []

	for doc in docs:
		doc_data = doc.to_dict()
		# Check if the document has votes and responses
		if 'votes' in doc_data and doc_data['votes'] and 'responses' in doc_data and len(doc_data['responses']) == 2:
			# Process votes to determine preference
			vote_preference = round(sum(doc_data['votes']) / len(doc_data['votes']))
			# truncate responses at appearance of second '<user>' message
			preprocessed_responses = [preprocess_text(response) for response in doc_data['responses']]

			formatted_data.append((preprocessed_responses, vote_preference))

	return formatted_data


class PairwiseComparisonDataset(Dataset):
	def __init__(self, data):
		"""
		Args:
			data (list of tuples): A list where each tuple contains (`responses`, `vote`),
			with `responses` being an array of two text responses and `vote` being 0 or 1.
		"""
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		responses, vote = self.data[idx]
		return responses, vote


def collate_fn(batch):
	# Combine prompt and outputs for each example in the batch
	texts, labels = zip(*batch)

	# Tokenize texts. This also adds special tokens (CLS, SEP) and converts to token IDs.
	# attention_mask will automatically be created by the tokenizer.
	# If you're using different segments, you might also want to manually create token_type_ids.
	encoding = tokenizer(
		list(texts),
		padding=True,
		truncation=True,
		return_tensors="pt",
		max_length=512,
		return_overflowing_tokens=False,
	)

	# Convert labels into a tensor
	labels_tensor = torch.tensor(labels, dtype=torch.long)

	# In this simplified example, token_type_ids are not manually created. Depending on your
	# model's needs, you might need to create and handle them specifically.
	return {
		'input_ids': encoding['input_ids'],
		'attention_mask': encoding['attention_mask'],
		'labels': labels_tensor,
		'token_type_ids': encoding['token_type_ids']
	}


class RewardModel(nn.Module):
	def __init__(self):
		super(RewardModel, self).__init__()
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		# Binary classification (choice between output 1 and output 2)
		self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

	def forward(self, input_ids, attention_mask, token_type_ids):
		outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
		pooled_output = outputs.pooler_output
		logits = self.classifier(pooled_output)
		return logits


data = fetch_and_format_data_from_firestore()
# Separate the features and labels
X = [item[0] for item in data]  # This would be your responses pairs
y = [item[1] for item in data]  # This would be your vote preferences

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Now, you can create your Dataset instances for training and validation
train_dataset = PairwiseComparisonDataset(list(zip(X_train, y_train)))
val_dataset = PairwiseComparisonDataset(list(zip(X_val, y_val)))

# And then create DataLoaders for each
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

model = RewardModel()
optimizer = AdamW(model.parameters(), lr=5e-5)

loss_fn = nn.CrossEntropyLoss()
best_val_loss = float('inf')

for epoch in range(num_epochs):
	model.train()
	total_train_loss = 0
	for batch in train_dataloader:
		optimizer.zero_grad()
		input_ids = batch['input_ids']
		attention_mask = batch['attention_mask']
		token_type_ids = batch['token_type_ids']
		labels = batch['labels']

		outputs = model(input_ids, attention_mask, token_type_ids)
		loss = loss_fn(outputs, labels)  # Define your loss function
		loss.backward()
		optimizer.step()

		total_train_loss += loss.item()

	avg_train_loss = total_train_loss / len(train_dataloader)

	# Validation
	model.eval()
	total_val_loss = 0
	correct_predictions = 0
	total_predictions = 0
	with torch.no_grad():
		for batch in val_dataloader:
			input_ids = batch['input_ids']
			attention_mask = batch['attention_mask']
			token_type_ids = batch['token_type_ids']
			labels = batch['labels']  # Your labels here

			outputs = model(input_ids, attention_mask, token_type_ids)
			loss = loss_fn(outputs, labels)

			# Calculate validation accuracy
			_, preds = torch.max(outputs, dim=1)

			# Update counts
			correct_predictions += torch.sum(preds == labels).item()
			total_predictions += labels.size(0)

			total_val_loss += loss.item()

	avg_val_loss = total_val_loss / len(val_dataloader)
	accuracy = correct_predictions / total_predictions

	if avg_val_loss < best_val_loss:
		best_val_loss = avg_val_loss
		torch.save(model.state_dict(), 'rlhf_pipe/reward_model/model.pth')

	print(f"Epoch {epoch + 1}/{num_epochs}, Train loss: {avg_train_loss}, Validation loss: {avg_val_loss}, Validation accuracy: {accuracy}")
