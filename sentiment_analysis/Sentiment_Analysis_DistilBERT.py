import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

df = pd.read_csv('data/datasets/goEmotions/cleaned_data.csv')

model_ckpt = "/data/models/distilbert-base-uncased"
pretrained_model_path = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_path)

# Train, val, test split
df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42)

df_train['input_ids'] = None
df_train['attention_mask'] = None

df_val['input_ids'] = None
df_val['attention_mask'] = None

df_test['input_ids'] = None
df_test['attention_mask'] = None


def tokenize(df):
    for index, row in df.iterrows():
        tokenized_output = tokenizer(row['text'], padding='max_length', truncation=True)
        df.at[index, 'input_ids'] = tokenized_output['input_ids']
        df.at[index, 'attention_mask'] = tokenized_output['attention_mask']
    return df


df_train = tokenize(df_train)
df_val = tokenize(df_val)
df_test = tokenize(df_test)

label_list = df_train['emotion_label'].unique()
num_labels = len(label_list)
print('Labels: ', label_list, '\n Qty:', num_labels)

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(df_train['emotion_label'])
val_labels = label_encoder.transform(df_val['emotion_label'])
test_labels = label_encoder.transform(df_test['emotion_label'])

df_train['label'] = train_labels
df_val['label'] = val_labels
df_test['label'] = test_labels

# Rename cleaned_text to text
df_train = df_train.rename(columns={'cleaned_text': 'text'})
df_val = df_val.rename(columns={'cleaned_text': 'text'})
df_test = df_test.rename(columns={'cleaned_text': 'text'})

input_ids = torch.tensor(df_train['input_ids'].tolist())
attention_mask = torch.tensor(df_train['attention_mask'].tolist())
labels = torch.tensor(df_train['label'].tolist())


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

val_input_ids = torch.tensor(df_val['input_ids'].tolist())
val_attention_mask = torch.tensor(df_val['attention_mask'].tolist())
val_labels = torch.tensor(df_val['label'].tolist())

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


batch_size = 32
logging_steps = len(df_train) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
num_epochs = 8
num_warmup_steps = 500
training_args = TrainingArguments(output_dir=model_name, num_train_epochs=num_epochs,
                                  learning_rate=8e-06,
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

test_input_ids = torch.tensor(df_test['input_ids'].tolist())
test_attention_mask = torch.tensor(df_test['attention_mask'].tolist())
test_labels = torch.tensor(df_test['label'].tolist())

test_dataset = CustomDataset(test_input_ids, test_attention_mask, test_labels)

# print test results
test_results = trainer.predict(test_dataset)
print(test_results.metrics)

# Save the model
model.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)

