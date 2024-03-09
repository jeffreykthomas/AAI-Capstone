import os
from functools import partial
import time

import wandb
from datetime import datetime

from llama import model as llama_model
from data.prepare_data import Task

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import get_cosine_schedule_with_warmup

# Model Configuration
vocab_size = 32000  # Vocabulary size of the pre-trained SentencePiece model
max_length = 1024  # Maximum sequence length
num_layers = 16  # Number of transformer layers
num_heads = 16  # Number of attention heads
d_model = 2048  # Dimension of the transformer model
dropout = 0.1  # Dropout rate
micro_batch_size = 8  # Batch size per GPU
compile_model = True  # Whether to compile the model

# Training Configuration
warmup_steps = 1000
accumulation_steps = 32  # Number of steps to accumulate gradients
num_global_steps = 100000  # Number of global steps
min_lr = 3e-5
learning_rate = 3e-4
lr_decay_iters = num_global_steps
grad_clip = 1.0
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
eval_steps = 50  # Number of steps between evaluation of the model
eval_iters = 100  # Number of iterations to evaluate the model
log_interval = 4  # Number of steps between logging

# wandb logging
wandb_project = 'Llama-Health-Chatbot'
wandb_run_name = 'run' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

data_folder = '/data/datasets/openwebtext'
output_dir = '/data/models/llama_health'

# Initialize the model
model_config = llama_model.TransformerConfig()
model_config.model_dimension = d_model
model_config.num_layers = num_layers
model_config.num_attention_heads = num_heads
model_config.num_kv_heads = num_heads
model_config.vocabulary_size = vocab_size
model_config.multiple_of = 32
model_config.norm_eps = 1e-5
model_config.max_seq_len = max_length
model_config.dropout = dropout

dtype = 'bfloat16'  # Data type for the model

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ.get('RANK'))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
    ddp_world_size = int(os.environ.get('WORLD_SIZE'))
    ddp_device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(ddp_device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert accumulation_steps % ddp_world_size == 0, 'accumulation_steps must be divisible by the number of processes'
    accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = accumulation_steps * micro_batch_size * max_length * ddp_world_size
if master_process:
    print(f'Number of tokens per iteration: {tokens_per_iter}')
    print(f'breakdown: {accumulation_steps} steps * {micro_batch_size} batch size * {max_length} sequence length * {ddp_world_size} world size')
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

torch.manual_seed(42 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
ptdtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float64': torch.float64}[dtype]


iter_batches = partial(
    Task.iter_batches,
    batch_size=micro_batch_size,
    max_seq_len=max_length,
    vocab_size=vocab_size,
    device=device_type,
    num_workers=0,
)


def run_training():
    if master_process:
        wandb.init(project=wandb_project, name=wandb_run_name)

    scaler = GradScaler()
    device = torch.device(ddp_device if ddp else device_type)
    model = llama_model.Transformer(model_config)
    model = model.to(device)
    if compile_model:
        unoptimized_model = model
        model = torch.compile(model)

    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device.type)

    # Print number of parameters
    if master_process:
        print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if ddp:
        # Need to ignore the 'freqs_cis' parameter in the DDP wrapper
        prefix = '_orig_mod.' if compile_model else ''
        model._ddp_params_and_buffers_to_ignore = {f'{prefix}freqs_cis'}
        model = DDP(model, device_ids=[ddp_local_rank])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_global_steps
    )

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            batch_iter = iter_batches(split=split)
            losses = torch.zeros(eval_iters)  # keep on CPU
            for k in range(eval_iters):
                X, y = next(batch_iter)
                with autocast(dtype=ptdtype):
                    logits = model(X, y)
                    loss = raw_model.last_loss
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    best_val_loss = float('inf')
    local_iter_num = 0
    running_mfu = -1.0
    raw_model = model.module if ddp else model
    train_batch_iter = iter_batches(split='train')
    X, y = next(train_batch_iter)  # first batch
    t0 = time.time()

    for global_step in range(num_global_steps):
        # Validation loop
        if global_step % eval_steps == 0 and master_process:
            losses = estimate_loss()
            val_loss = losses['val']
            train_loss = losses['train']
            wandb.log(
                {
                    'global_step': global_step,
                    'loss': train_loss,
                    'val_loss': val_loss,
                    'lr': scheduler.get_last_lr()[0],
                    'mfu': running_mfu * 100,
                }
            )
            print(
                f'\nStep {global_step + 1}, '
                f'loss: {train_loss}, '
                f'val_loss: {val_loss}, ')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the model
                if global_step > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
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

        for micro_step in range(accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = micro_step == accumulation_steps - 1
            with autocast(dtype=ptdtype):
                # Forward pass
                logits = model(X, y)
                loss = raw_model.last_loss
                # Normalize the loss
                loss = loss / accumulation_steps
            X, y = next(train_batch_iter)
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Perform optimization step
        scaler.step(optimizer)
        scaler.update()

        # Zero gradients
        optimizer.zero_grad()
        scheduler.step()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if global_step % log_interval == 0 and master_process:
            lossf = loss.item() * accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(micro_batch_size * accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(
                f'\r{global_step + 1}/{num_global_steps} | '
                f'loss {lossf:.4f} | '
                f'lr {scheduler.get_last_lr()[0]:e} | '
                f'{dt*1000:.2f}ms | '
                f'mfu {running_mfu*100:.2f}%',
                end='',
                flush=True
            )

        local_iter_num += 1

    if ddp:
        destroy_process_group()


if __name__ == '__main__':
    run_training()
