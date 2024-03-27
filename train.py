# from project root run: python train.py
# Distributed training: torchrun --standalone --nproc_per_node=2 train.py
# Using Galore: torchrun --standalone --nproc_per_node=2 train.py --use_galore
# Fine-tuning model on empathetic dialogues:
# torchrun --standalone --nproc_per_node=2 train.py /
# --use_galore --dataset empathetic_dialogues --output_dir /data/models/llama_empathetic_dialogues /
# --wandb_run_nam empathetic-dialogues-run --dropout 0.4 /
# --warmup_steps 500 --num_global_steps 1000 --min_lr 3e-5 --learning_rate 5e-5 /
# --lr_decay_iters 1000 --grad_clip 1.0 --weight_decay 0.01 --beta1 0.9 --beta2 0.95
# --eval_steps 20 --eval_iters 100 --log_interval 10


# Max on 2 A6000 GPUs using DDP: torchrun --standalone --nproc_per_node=2 train.py --use_galore -num_layers 28

import os
from functools import partial
import time
import argparse

import wandb
from datetime import datetime

from llama import model as llama_model
from data.prepare_data import Task

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import get_cosine_schedule_with_warmup

args_parser = argparse.ArgumentParser()

# Data Configuration
args_parser.add_argument('--dataset', type=str, default='openwebtext')
args_parser.add_argument('--load_pretrained', type=bool, default=False)
args_parser.add_argument('--pretrained_path', type=str, default=None)
args_parser.add_argument('--output_dir', type=str, default='/data/models/llama_health_galore')
args_parser.add_argument('--wandb_project', type=str, default='Llama-Health-Chatbot')
args_parser.add_argument('--wandb_run_name', type=str, default='run')

# Model Configuration
args_parser.add_argument('--d_model', type=int, default=2048)  # Dimension of the transformer model
args_parser.add_argument('--num_layers', type=int, default=16)  # Number of transformer layers
args_parser.add_argument('--num_heads', type=int, default=32)  # Number of attention heads
args_parser.add_argument('--n_kv_heads', type=int, default=32)  # Number of key-value heads
args_parser.add_argument('--vocab_size', type=int, default=32000)
args_parser.add_argument('--multiple_of', type=int, default=32)  # Multiple of the model dimension
args_parser.add_argument('--norm_eps', type=float, default=1e-5)  # Epsilon value for layer normalization
args_parser.add_argument('--max_seq_len', type=int, default=1024)  # Maximum sequence length
args_parser.add_argument('--dropout', type=float, default=0.1)  # Dropout rate
args_parser.add_argument('--micro_batch_size', type=int, default=8)  # Batch size per GPU
args_parser.add_argument('--compile_model', type=bool, default=True)  # Whether to compile the model

# Training Configuration
args_parser.add_argument('--warmup_steps', type=int, default=1000)
args_parser.add_argument('--accumulation_steps', type=int, default=32)
args_parser.add_argument('--num_global_steps', type=int, default=75000)
args_parser.add_argument('--learning_rate', type=float, default=3e-4)
args_parser.add_argument('--lr_decay_iters', type=int, default=75000)
args_parser.add_argument('--grad_clip', type=float, default=1.0)
args_parser.add_argument('--weight_decay', type=float, default=0.01)
args_parser.add_argument('--beta1', type=float, default=0.9)
args_parser.add_argument('--beta2', type=float, default=0.95)
args_parser.add_argument('--eval_steps', type=int, default=50)
args_parser.add_argument('--eval_iters', type=int, default=100)
args_parser.add_argument('--log_interval', type=int, default=4)

# Galore Configuration
args_parser.add_argument('--use_galore', action='store_true', help='Enable Galore Optimization')
args_parser.add_argument('--rank', type=int, default=128)
args_parser.add_argument('--update_proj_gap', type=int, default=200)
args_parser.add_argument('--scale', type=float, default=0.25)
args_parser.add_argument('--proj_type', type=str, default='std')

args = args_parser.parse_args()

# Initialize the model
model_config = llama_model.TransformerConfig()
model_config.model_dimension = args.d_model
model_config.num_layers = args.num_layers
model_config.num_heads = args.num_heads
model_config.n_kv_heads = args.n_kv_heads
model_config.vocab_size = args.vocab_size
model_config.multiple_of = args.multiple_of
model_config.norm_eps = args.norm_eps
model_config.max_seq_len = args.max_seq_len
model_config.dropout = args.dropout

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
    assert args.accumulation_steps % ddp_world_size == 0, \
        'accumulation_steps must be divisible by the number of processes'
    args.accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = args.accumulation_steps * args.micro_batch_size * args.max_seq_len * ddp_world_size
if master_process:
    print(f'Number of tokens per iteration: {tokens_per_iter}')
    print(f'breakdown: {args.accumulation_steps} steps * {args.micro_batch_size} batch size * {args.max_seq_len} sequence length * {ddp_world_size} world size')
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

torch.manual_seed(42 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
ptdtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float64': torch.float64}[dtype]


iter_batches = partial(
    Task.iter_batches,
    batch_size=args.micro_batch_size,
    max_seq_len=args.max_seq_len,
    vocab_size=args.vocab_size,
    dataset=args.dataset,
    device=device_type,
    num_workers=0,
)


def run_training():
    if master_process:
        wandb.init(project=args.wandb_project,
                   name=(args.wandb_run_name + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),
                   config={k: v for k, v in vars(args).items()})

    scaler = GradScaler()
    device = torch.device(ddp_device if ddp else device_type)
    model = llama_model.Transformer(model_config)
    model = model.to(device)
    if args.load_pretrained:
        if args.pretrained_path is None:
            ckpt_path = '/data/models/llama_health/model_step_74001.pt'
        else:
            ckpt_path = args.pretrained_path
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

    if args.compile_model:
        unoptimized_model = model
        model = torch.compile(model)

    optimizer = model.configure_optimizers(
        args.weight_decay,
        args.learning_rate,
        (args.beta1, args.beta2),
        device.type,
        args.use_galore,
        args.rank,
        args.update_proj_gap,
        args.scale,
        args.proj_type)
    
    # Print number of parameters
    if master_process:
        print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    if ddp:
        # Need to ignore the 'freqs_cis' parameter in the DDP wrapper
        prefix = '_orig_mod.' if args.compile_model else ''
        model._ddp_params_and_buffers_to_ignore = {f'{prefix}freqs_cis'}
        model = DDP(model, device_ids=[ddp_local_rank])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_global_steps
    )

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            batch_iter = iter_batches(split=split)
            losses = torch.zeros(args.eval_iters)  # keep on CPU
            for k in range(args.eval_iters):
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

    for global_step in range(args.num_global_steps):
        # Validation loop
        if global_step % args.eval_steps == 0 and master_process:
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

            if (val_loss < best_val_loss) or args.load_pretrained:
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
                    # save the model, new file for every 1000 steps
                    rounded_global_step = global_step // 1000 * 1000 + 1
                    torch.save(checkpoint, os.path.join(args.output_dir, f'model_step_{rounded_global_step}.pt'))

        model.train()

        for micro_step in range(args.accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = micro_step == args.accumulation_steps - 1
            with autocast(dtype=ptdtype):
                # Forward pass
                logits = model(X, y)
                loss = raw_model.last_loss
                # Normalize the loss
                loss = loss / args.accumulation_steps
            X, y = next(train_batch_iter)
            scaler.scale(loss).backward()

        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Perform optimization step
        scaler.step(optimizer)
        scaler.update()

        # Zero gradients
        optimizer.zero_grad()
        scheduler.step()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if global_step % args.log_interval == 0 and master_process:
            lossf = loss.item() * args.accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(args.micro_batch_size * args.accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(
                f'\r{global_step + 1}/{args.num_global_steps} | '
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
