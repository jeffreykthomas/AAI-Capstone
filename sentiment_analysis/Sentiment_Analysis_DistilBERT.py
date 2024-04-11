import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

df_preprocessed = pd.read_csv('data/processed.csv')
df_preprocessed.head(10)

model_ckpt = "/data/models/distilbert-base-uncased"
pretrained_model_path = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_path)


def tokenize(text, max_length=200):
    if text is None:
        text = ''
    return tokenizer(text, padding='max_length', truncation=True)


df_preprocessed['input_ids'] = None
df_preprocessed['attention_mask'] = None

for index, row in df_preprocessed.iterrows():
    tokenized_output = tokenize(row['cleaned_text'])
    df_preprocessed.at[index, 'input_ids'] = tokenized_output['input_ids']
    df_preprocessed.at[index, 'attention_mask'] = tokenized_output['attention_mask']

label_list = df_preprocessed['emotion_label'].unique()
num_labels = len(label_list)
print('Labels: ', label_list, '\n Qty:', num_labels)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df_preprocessed['emotion_label'])

emotions_encoded = pd.DataFrame(columns=['attention_mask', 'input_ids', 'label',
                                         'text'])
emotions_encoded['attention_mask'] = df_preprocessed['attention_mask']
emotions_encoded['input_ids'] = df_preprocessed['input_ids']
emotions_encoded['label'] = labels
emotions_encoded['text'] = df_preprocessed['cleaned_text']

train_df, val_df = train_test_split(
    emotions_encoded,
    test_size=0.2,
    stratify=emotions_encoded['label'],
    random_state=42
)

input_ids = torch.tensor(train_df['input_ids'].tolist())
attention_mask = torch.tensor(train_df['attention_mask'].tolist())
labels = torch.tensor(train_df['label'].tolist())


class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx].unsqueeze(0)
        }


print('Input IDs shape:', input_ids.shape)
print('Attention mask shape:', attention_mask.shape)
print('Labels shape:', labels.shape)
train_dataset = CustomDataset(input_ids, attention_mask, labels)

val_input_ids = torch.tensor(val_df['input_ids'].tolist())
val_attention_mask = torch.tensor(val_df['attention_mask'].tolist())
val_labels = torch.tensor(val_df['label'].tolist())

val_dataset = CustomDataset(val_input_ids, val_attention_mask, val_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_model_name_or_path = 'distilbert-base-uncased'
dropout_prob = 0.1
config = DistilBertConfig.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels,
                                          hidden_dropout_prob=dropout_prob, attention_probs_dropout_prob=dropout_prob)

model = DistilBertForSequenceClassification(config)
model = model.to(device)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


batch_size = 64
logging_steps = len(emotions_encoded) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
num_epochs = 12
num_warmup_steps = 2000
training_args = TrainingArguments(output_dir=model_name, num_train_epochs=num_epochs,
                                  learning_rate=5e-05,
                                  lr_scheduler_type='cosine',
                                  warmup_steps=num_warmup_steps,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01, evaluation_strategy="epoch",
                                  disable_tqdm=False, logging_steps=logging_steps,
                                  push_to_hub=False, log_level="error")

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=train_dataset, eval_dataset=val_dataset,
                  tokenizer=tokenizer)

trainer.train()
