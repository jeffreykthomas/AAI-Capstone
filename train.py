import torch
from llama import chat_model
from llama.tokenizer import Tokenizer
import wandb
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import numpy as np
from transformers import get_linear_schedule_with_warmup

# Model Configuration
vocab_size = 32000  # Vocabulary size of the pre-trained SentencePiece model
max_length = 128  # Maximum sequence length
n_layers = 6  # Number of transformer layers
n_heads = 8  # Number of attention heads
d_model = 512  # Dimension of the transformer model
d_ff = 2048  # Dimension of the feed-forward layer
dropout = 0.1  # Dropout rate

# wandb logging
wandb_project = "Llama-Chatbot"
wandb_run_name = "run" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

data_folder = 'data/'

# Initialize the model
model_config = chat_model.TransformerConfig()
model_config.model_dimension = d_model
model_config.num_layers = n_layers
model_config.num_attention_heads = n_heads
model_config.vocabulary_size = vocab_size
model_config.swiglu_multiple = 256
model_config.normalization_epsilon = 1e-5
model_config.max_batch_size = 32
model_config.max_sequence_length = max_length


class CustomDataset(Dataset):
	def __init__(self, input_encodings):
		self.input_encodings = input_encodings

	def __getitem__(self, idx):
		input_item = {key: self.input_encodings[key][idx] for key in self.input_encodings}
		return input_item

	def __len__(self):
		return len(self.input_encodings['input_ids'])


class PadCollate():
	def __init__(self, pad_id):
		self.pad_id = pad_id

	def pad_collate(self, batch):
		input_ids, attn_masks = [], []
		max_len = max(len(seqs['input_ids']) for seqs in batch)  # Find the maximum sequence length in this batch

		for idx, seqs in enumerate(batch):
			pad_len = max_len - len(seqs['input_ids'])  # Calculate how much padding is needed

			input_ids.append(F.pad(torch.LongTensor(seqs['input_ids'].long()), (pad_len, 0), value=self.pad_id))
			attn_masks.append(
				F.pad(torch.LongTensor(seqs['attention_mask'].long()), (pad_len, 0), value=0))

		# Stack the tensors along a new dimension
		input_ids = torch.stack(input_ids)
		attn_masks = torch.stack(attn_masks)

		x_encodings = {'input_ids': input_ids,
		               'attention_mask': attn_masks}

		return x_encodings


def load_data(train_X, val_X, test_X, pad_id):
	ppd = PadCollate(pad_id)
	train_dataset = CustomDataset(train_X)
	val_dataset = CustomDataset(val_X)
	test_dataset = CustomDataset(test_X)

	# Create the dataloader
	train_loader = DataLoader(train_dataset, batch_size=model_config.max_batch_size, shuffle=True,
	                          collate_fn=ppd.pad_collate)
	val_loader = DataLoader(val_dataset, batch_size=model_config.max_batch_size, shuffle=False,
	                        collate_fn=ppd.pad_collate)
	test_loader = DataLoader(test_dataset, batch_size=model_config.max_batch_size, shuffle=False,
	                         collate_fn=ppd.pad_collate)

	return train_loader, val_loader, test_loader


def run_training():
	wandb.init(project=wandb_project, name=wandb_run_name)
	num_train_epochs = 3
	accumulation_steps = 4

	tokenizer = Tokenizer(model_path='tokenizer.model')
	pad_id = tokenizer.pad_id

	# Initialize the scaler
	scaler = GradScaler()

	train_encodings = torch.load(data_folder + 'train_encodings.pt')
	val_encodings = torch.load(data_folder + 'val_encodings.pt')
	test_encodings = torch.load(data_folder + 'test_encodings.pt')

	train_loader, val_loader, test_loader = load_data(train_encodings, val_encodings, test_encodings, pad_id)

	num_train_steps = len(train_loader) // accumulation_steps * num_train_epochs

	model = chat_model.TransformerModel(model_config)

	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = torch.nn.DataParallel(model)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	optimizer = torch.optim.AdamW(lr=5e-5, eps=1e-8)
	scheduler = get_linear_schedule_with_warmup(
		optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
	)

	best_val_loss = float('inf')
	global_step = 0

	for epoch in range(num_train_epochs):
		model.train()
		for batch_idx, batch in enumerate(train_loader):
			# Move tensors to the device
			input_ids = batch['input_ids'].to(device)
			attention_mask = batch.get('attention_mask', None)
			if attention_mask is not None:
				attention_mask = attention_mask.to(device)

			target_ids = batch['input_ids'].to(device)

			with autocast():
				# Forward pass
				outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
				loss = outputs.loss.mean()

			# Normalize the loss
			loss = loss / accumulation_steps

			# Backward pass and optimization
			scaler.scale(loss).backward()

			# Update only after accumulating gradients for n steps
			if (batch_idx + 1) % accumulation_steps == 0:
				# Update step count
				global_step += 1

				# Clip gradients
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

				# Perform optimization step
				scaler.step(optimizer)
				scaler.update()

				# Zero gradients
				optimizer.zero_grad()

				# Update learning rate schedule
				scheduler.step()

				print(
					f'\rEpoch {epoch + 1}, batch: {batch_idx + 1}/{len(train_loader)}, scaled_loss: {loss.item()}, '
					f'effective_loss: {loss.item() * accumulation_steps}',
					end='',
					flush=True)

			# Validation loop
			if batch_idx % 1000 == 0:
				max_val_batches = 64
				# choose random indices to evaluate on
				random_indices = np.random.randint(0, len(val_loader), max_val_batches)
				model.eval()
				val_losses = []
				with torch.no_grad():
					for val_idx, batch in enumerate(val_loader):
						if val_idx not in random_indices:
							continue
						# Move tensors to the device
						val_input_ids = batch['input_ids'].to(device)
						val_attention_mask = batch.get('attention_mask', None)
						if val_attention_mask is not None:
							val_attention_mask = val_attention_mask.to(device)
						val_target_ids = batch['input_ids'].to(device)

						with autocast():
							# Forward pass
							outputs = model(
								input_ids=val_input_ids,
								attention_mask=val_attention_mask,
								labels=val_target_ids
							)

							val_loss = outputs.loss.mean()
							val_losses.append(val_loss.item())

							# print out a sample of the validation set
							if val_idx == random_indices[0]:
								print(f'\nContext: {tokenizer.decode(val_input_ids[0])}')
								print(
									f'Generated output: {tokenizer.decode(outputs.logits[0].argmax(dim=-1))}')
								labels = val_target_ids.where(val_target_ids != -100, tokenizer.pad_id)
								print(f'Ground truth: {tokenizer.decode(labels[0])}')

				combined_val_loss = sum(val_losses) / len(val_losses)
				wandb.log(
					{
						'global_step': global_step,
						'loss': loss.item() * accumulation_steps,
						'val_loss': combined_val_loss,
						'lr': scheduler.get_last_lr()[0]})
				print(
					f'\nEpoch {epoch + 1}, '
					f'batch: {batch_idx + 1}/{len(train_loader)}, '
					f'loss: {loss.item() * accumulation_steps}, '
					f'val_loss: {combined_val_loss}')

				if combined_val_loss < best_val_loss:
					best_val_loss = combined_val_loss
					# Save the model

					checkpoint = {
						'model': model.state_dict(),
						'optimizer': optimizer.state_dict(),
						'model_config': model_config,
						'scaler': scaler.state_dict(),
						'scheduler': scheduler.state_dict(),
						'best_val_loss': best_val_loss,
						'global_step': global_step
					}
					print(f'Saving model with val_loss: {best_val_loss} and global_step: {global_step}')
					torch.save(checkpoint, 'best_model.pt')

				model.train()


if __name__ == '__main__':
	run_training()
