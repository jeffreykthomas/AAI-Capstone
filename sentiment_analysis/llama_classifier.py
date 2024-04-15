import os
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
import evaluate
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForCausalLM
import torch.optim as optim
import wandb
from datetime import datetime


# Config
num_epochs = 2
batch_size = 16
learning_rate = 1e-6
num_layers_freeze = 16

save_dir = '/data/models/llama_classifier_model/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def load_model(model_path):
    return AutoModelForCausalLM.from_pretrained(model_path)


def load_tokenizer(model_path):
    padding_side = 'right'
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=padding_side)
    tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token
    return tokenizer


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors='pt',
        return_overflowing_tokens=False)


class SentimentModel(nn.Module):
    def __init__(self, llama_model, num_labels, num_layers_freeze=0):
        super(SentimentModel, self).__init__()
        self.base_model = load_model(llama_model)
        # Freeze 'num_layers_freeze' layers in the base model
        layer_count = 0
        for name, child in self.base_model.named_modules():
            if layer_count < num_layers_freeze:
                for param in child.parameters():
                    param.requires_grad = False
            layer_count += 1
        self.num_labels = num_labels
        # Classifier to be applied on pooled output
        self.classifier = nn.Linear(self.base_model.config.vocab_size, num_labels)

    def forward(self, input_ids, attention_mask, labels):
        # Get output from the base model
        base_model_output = self.base_model(input_ids)
        pooled_output = torch.mean(base_model_output[0], dim=1)
        logits = self.classifier(pooled_output)

        return logits


df = pd.read_csv('data/datasets/goEmotions/cleaned_data.csv')

# Train, val, test split
df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42)

model_name = 'jeffreykthomas/llama-mental-health-base'
num_labels = df_train['emotion_label'].nunique()
model = SentimentModel(model_name, num_labels, num_layers_freeze=num_layers_freeze)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device) 

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
tokenizer = load_tokenizer(model_name)

unique_labels = df_train['emotion_label'].unique()
emotion_index_to_label = {label: index for index, label in enumerate(unique_labels)}

df_train['emotion_label'] = df_train['emotion_label'].apply(lambda x: emotion_index_to_label[x])
df_val['emotion_label'] = df_val['emotion_label'].apply(lambda x: emotion_index_to_label[x])
df_test['emotion_label'] = df_test['emotion_label'].apply(lambda x: emotion_index_to_label[x])

train = df_train.rename(columns={'emotion_label': 'labels'})
val = df_val.rename(columns={'emotion_label': 'labels'})
test = df_test.rename(columns={'emotion_label': 'labels'})

train = Dataset.from_pandas(train)
val = Dataset.from_pandas(val)
test = Dataset.from_pandas(test)

data_dict = DatasetDict({
    'train': train,
    'validation': val,
    'test': test
})

tokenized_dataset = data_dict.map(tokenize, batched=True, remove_columns=['text'])

tokenized_dataset.set_format("torch", columns=["input_ids", 'attention_mask', "labels"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator, drop_last=True)
val_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=batch_size, collate_fn=data_collator, drop_last=True)
test_data_loader = DataLoader(tokenized_dataset["test"], batch_size=batch_size, collate_fn=data_collator, drop_last=True)

metric_accuracy = evaluate.load('accuracy')
metric_f1 = evaluate.load('f1')
best_val_loss = float('inf')
lossfn = nn.CrossEntropyLoss()

num_training_steps = num_epochs * len(train_dataloader)
progress_bar_train = tqdm(range(num_training_steps))
progress_bar_eval = tqdm(range(num_epochs * len(val_dataloader)))

# Logging
wandb.init(project='llama-sentiment-analysis',
           name='run' + datetime.now().strftime("%Y%m%d-%H%M%S"),
           config={
               'model': model_name,
               'num_labels': num_labels,
               'num_epochs': num_epochs,
               'frozen_layers': num_layers_freeze,
               'batch_size': batch_size,
               'learning_rate': learning_rate}
           )


for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = lossfn(outputs, batch['labels']) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_bar_train.update(1)
        progress_bar_train.set_description(f"Epoch {epoch + 1}/{num_epochs}, Train loss: {loss.item()}")
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss = lossfn(outputs, batch['labels']) 
                
            _, predictions = torch.max(outputs, dim=1)
            metric_accuracy.add_batch(predictions=predictions, references=batch["labels"])
            progress_bar_eval.update(1)
            progress_bar_eval.set_description(f"Epoch {epoch + 1}/{num_epochs}, Val loss: {loss.item()}")
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f'{save_dir}model.pth')
        
    epoch_metrics = metric_accuracy.compute()

    wandb.log({
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'val_accuracy': epoch_metrics['accuracy'],
    })

# evaluate accuracy and f1 on test dataloader
model.load_state_dict(torch.load(f'{save_dir}model.pth'))
model.eval()
with torch.no_grad():
    for batch in test_data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        _, predictions = torch.max(outputs, dim=1)
        metric_accuracy.add_batch(predictions=predictions, references=batch["labels"])
        metric_f1.add_batch(predictions=predictions, references=batch["labels"])

accuracy = metric_accuracy.compute()
f1 = metric_f1.compute(average='weighted')

wandb.log({
    'test_accuracy': accuracy['accuracy'],
    'test_f1': f1['f1']
})
