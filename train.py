import os
import torch
from llama import model as llama_model
import wandb
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from transformers import get_linear_schedule_with_warmup

# Model Configuration
vocab_size = 32000  # Vocabulary size of the pre-trained SentencePiece model
max_length = 256  # Maximum sequence length
n_layers = 16  # Number of transformer layers
n_heads = 16  # Number of attention heads
d_model = 2048  # Dimension of the transformer model
dropout = 0.1  # Dropout rate
max_batch_size = 128  # Batch size

# Training Configuration
warmup_steps = 2000
num_train_steps = 600000
min_lr = 6e-5
learning_rate = 6e-4
grad_clip = 1.0
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
accumulation_steps = 16  # Number of steps to accumulate gradients
eval_steps = 2000  # Number of steps between evaluation of the model
eval_iters = 10  # Number of iterations to evaluate the model

# wandb logging
wandb_project = "Llama-Health-Chatbot"
wandb_run_name = "run" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

data_folder = '/data/datasets/openwebtext'
output_dir = '/data/models/llama_health'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the model
model_config = llama_model.TransformerConfig()
model_config.model_dimension = d_model
model_config.num_layers = n_layers
model_config.num_attention_heads = n_heads
model_config.num_kv_heads = n_heads
model_config.vocabulary_size = vocab_size
model_config.swiglu_multiple = 32
model_config.normalization_epsilon = 1e-5
model_config.max_batch_size = max_batch_size
model_config.max_sequence_length = max_length
model_config.dropout_rate = dropout

dtype = torch.bfloat16


def get_batch(split, data_dir, block_size, batch_size, device, device_type='cuda'):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def run_training():
    wandb.init(project=wandb_project, name=wandb_run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = GradScaler()
    model = llama_model.TransformerModel(model_config)
    # Print number of parameters
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model)

    model = model.to(device)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps
    )
    unoptimized_model = model
    model = torch.compile(model)

    best_val_loss = float('inf')
    global_step = 0

    for step in range(num_train_steps):
        model.train()
        batch_size = max_batch_size // accumulation_steps
        X, y = get_batch('train', data_folder, max_length, batch_size, device)
        # Move tensors to the device

        with autocast():
            # Forward pass
            logits = model(X, y)
            loss = model.last_loss
            # Normalize the loss
            loss = loss / accumulation_steps

        # Backward pass and optimization
        scaler.scale(loss).backward()

        # Update only after accumulating gradients for n steps
        if (step + 1) % accumulation_steps == 0:
            # Update step count
            global_step += 1

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Perform optimization step
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients
            optimizer.zero_grad()

            # Update learning rate schedule
            scheduler.step()

            print(
                f'\rTraining step {step + 1}/{num_train_steps}, scaled_loss: {loss.item()}, '
                f'effective_loss: {loss.item() * accumulation_steps}',
                end='',
                flush=True)

        # Validation loop
        if step % eval_steps == 0:
            model.eval()
            val_losses = torch.zeros(eval_iters)
            with torch.no_grad():
                for val_step in range(eval_iters):
                    X_val, y_val = get_batch('val', data_folder, max_length, batch_size, device)

                    with autocast():
                        # Forward pass
                        val_logits = model(X_val, y_val)
                        val_loss = model.last_loss

                    val_losses[val_step] = val_loss.item()

            val_loss = val_losses.mean()

            wandb.log(
                {
                    'global_step': global_step,
                    'loss': loss.item() * accumulation_steps,
                    'val_loss': val_loss,
                    'lr': scheduler.get_last_lr()[0]})
            print(
                f'\nStep {step + 1}, '
                f'loss: {loss.item() * accumulation_steps}, '
                f'val_loss: {val_loss}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
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
                torch.save(checkpoint, os.path.join(output_dir, 'best_chat_llama_model.pt'))

            model.train()


if __name__ == '__main__':
    run_training()
