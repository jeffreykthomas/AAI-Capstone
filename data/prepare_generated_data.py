import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

tokenizer = AutoTokenizer.from_pretrained("jeffreykthomas/llama-mental-health-distill")
df = pd.read_csv('data/datasets/Predictions/predictions.csv')

message_batch = []

for i in range(len(df)):
	messages = [
		{
			"role": "system",
			"content": "You are a helpful, respectful, expert mental health assistant. Respond to the User with empathy and respect."},
		{"role": "user", "content": df['Input'].iloc[i]},
		{"role": "assistant", "content": df['Reference'].iloc[i]}
	]
	message_batch.append(messages)
formatted_texts = [
	tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in message_batch
]

# Split the data into training, validation, and test sets with proportions 80%, 10%, 10%
train_texts, val_texts = train_test_split(formatted_texts, test_size=0.2)
val_texts, test_texts = train_test_split(val_texts, test_size=0.5)

# save as csv files
df_train = pd.DataFrame(train_texts, columns=['text'])
df_val = pd.DataFrame(val_texts, columns=['text'])
df_test = pd.DataFrame(test_texts, columns=['text'])

df_train.to_csv('data/datasets/generated_data/train.csv', index=False)
df_val.to_csv('data/datasets/generated_data/val.csv', index=False)
df_test.to_csv('data/datasets/generated_data/test.csv', index=False)