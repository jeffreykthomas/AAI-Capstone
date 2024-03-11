import pandas as pd
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import transformers
from datasets import load_dataset

data_folder = 'data/ubuntu/'


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
            input_ids.append(F.pad(torch.LongTensor(seqs['input_ids'].long()), (0, pad_len), value=self.pad_id))
            attn_masks.append(
                F.pad(torch.LongTensor(seqs['attention_mask'].long()), (0, pad_len), value=0))

        # Stack the tensors along a new dimension
        input_ids = torch.stack(input_ids)
        attn_masks = torch.stack(attn_masks)

        x_encodings = {'input_ids': input_ids,
                       'attention_mask': attn_masks}

        return x_encodings


model_name = 'jeffreykthomas/llama-mental-health'

# Set base model loading in 4-bits
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "bfloat16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = True

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant
)
# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    quantization_config=bnb_config,
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token_id = -1
tokenizer.paddding_side = "right"

wandb_project = 'aai-520-final-project'
wandb_run_name = 'llama-2-7b-ubuntu-generation'

wandb.init(project=wandb_project, name=wandb_run_name)

# Load the data
train_texts = pd.read_csv(data_folder + 'train_dataset.csv')
val_texts = pd.read_csv(data_folder + 'val_dataset.csv')
test_texts = pd.read_csv(data_folder + 'test_dataset.csv')

dataset = load_dataset('csv', data_files={'train': data_folder + 'train_dataset.csv',
                                          'validation': data_folder + 'val_dataset.csv',
                                          'test': data_folder + 'test_dataset.csv'})


ppd = PadCollate(tokenizer.pad_token_id)

# LoRA attention dimension
lora_r = 8
# Alpha for LoRA scaling
lora_alpha = 32
# Dropout probability for LoRA
lora_dropout = 0.05

peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=["query_key_value"],
    lora_dropout=lora_dropout,
    inference_mode=False,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

output_dir = "./results"
final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint_3epochs")

num_train_epochs = 3
max_steps = -1
bf16 = True
fp16 = False
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
gradient_accumulation_steps = 8
max_grad_norm = 0.1
# optim = "paged_adamw_32bit"
optim = "adamw_torch"
learning_rate = 2e-5
lr_scheduler_type = "constant"
warmup_ratio = 0.03
weight_decay = 0.01
group_by_length = True
gradient_checkpointing = True
save_steps = 50
logging_steps = 10

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to=["wandb"],
)

max_seq_length = 1024
packing = False

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

transformers.logging.set_verbosity_info()
resume_checkpoint = None

if __name__ == '__main__':
    trainer.train()
    trainer.save_model(final_checkpoint_dir)
