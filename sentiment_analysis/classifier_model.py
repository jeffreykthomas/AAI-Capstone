import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict, load_metric
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import TokenClassifierOutput
import torch.optim as optim


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = model.to(device)
    return model

def load_tokenizer(model_path):
    padding_side = 'left' if model_path == 'Qwen/Qwen1.5-0.5B-Chat' else 'right'
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=padding_side)
    tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token
    return tokenizer

def tokenize(batch):
    return tokenizer(batch["cleaned_text"], truncation=True, max_length=512, padding = True, return_tensors = 'pt', return_overflowing_tokens = False)

class SentimentModel(nn.Module):
    def __init__(self, llama_model, num_labels):
        super(SentimentModel, self).__init__()
        self.base_model = load_model(llama_model)
        self.num_labels = num_labels
        # Classifier to be applied on pooled output
        #self.classifier = nn.Linear(self.base_model.config.hidden_size, num_labels)
        self.classifier = nn.Linear(self.base_model.config.vocab_size, num_labels)

    def forward(self, input_ids, attention_mask, labels):
        # Get output from the base model
        base_model_output = self.base_model(input_ids)
        pooled_output = torch.mean(base_model_output[0], dim=1)
        logits = self.classifier(pooled_output)
        #sequence_output = base_model_output[0]
        #logits = self.classifier(sequence_output[:,0,:].reshape(-1, self.base_model.config.hidden_size))

        return logits


model_name = 'jeffreykthomas/llama-mental-health-base'
num_labels = 13
model = SentimentModel(model_name, num_labels)
num_epochs = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device) 

optimizer = optim.Adam(model.parameters(), lr=1e-5)
tokenizer = load_tokenizer(model_name)

train = pd.read_csv('data/train.csv')
val = pd.read_csv('data/validate.csv')

train = train.dropna()
val = val.dropna()

unique_labels = train['emotion_label'].unique()
emotion_index_to_label = {label: index for index, label in enumerate(unique_labels)}

train['emotion_label'] = train['emotion_label'].apply(lambda x: emotion_index_to_label[x])
val['emotion_label'] = val['emotion_label'].apply(lambda x: emotion_index_to_label[x])

train = train.rename(columns = {'emotion_label': 'labels'})
val = val.rename(columns = {'emotion_label': 'labels'})

train = Dataset.from_pandas(train)
val = Dataset.from_pandas(val)

data_dict = DatasetDict({
    'train': train,
    'validation': val
})

tokenized_dataset = data_dict.map(tokenize, batched = True, remove_columns = ['cleaned_text'])
#tokenized_dataset = tokenized_dataset.remove_columns('cleaned_text')
tokenized_dataset.set_format("torch",columns=["input_ids",'attention_mask', "labels"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size = 8, collate_fn=data_collator, drop_last = True)
val_dataloader = DataLoader(tokenized_dataset["validation"], batch_size = 8, collate_fn=data_collator, drop_last = True)

metric = load_metric('accuracy')
best_val_loss = float('inf')
lossfn = nn.CrossEntropyLoss()

num_training_steps = num_epochs * len(train_dataloader)
progress_bar_train = tqdm(range(num_training_steps))
progress_bar_eval = tqdm(range(num_epochs * len(val_dataloader)))

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
            metric.add_batch(predictions=predictions, references=batch["labels"])
            progress_bar_eval.update(1)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'sentiment analysis/classifier_model/model.pth')
        
    print(metric.compute())
	#print(f"Epoch {epoch + 1}/{num_epochs}, Train loss: {avg_train_loss}, Validation loss: {avg_val_loss}, Validation accuracy: {metric.compute()}")
