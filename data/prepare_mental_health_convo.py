import os
import pandas as pd
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def clean_data(df):
	# Drop null rows
	null_rows = df.isnull().any(axis=1)  # Identify rows with null values
	df = df[~null_rows]  # Drop rows with null values

	# Remove the duplicate rows
	df.drop_duplicates()

	return df


def custom_replace(match):
	# If the match is <user> or <agent>, keep it
	if match.group(0) in ['<user>', '<agent>']:
		return match.group(0)
	# Otherwise, replace it with an empty string
	return ""


def clean_text(text):
	text = re.sub("[\[].*?[\]]", "", text)  # Remove text within square brackets
	text = re.sub(r"http\S+", "", text)  # Remove URLs

	# Match <user>, <agent>, or any non-alphanumeric character (excluding spaces)
	text = re.sub(r"<user>|<agent>", custom_replace, text)
	return text


def replace_entities(text):
	# Replace <HUMAN> with <user>
	text = text.replace('<HUMAN>', '<user>')
	# Replace <ASSISTANT> with <agent>
	text = text.replace('<ASSISTANT>', '<agent>')
	# Replace : with space
	text = text.replace(':', '')
	return text


def preprocess_data(df_init, context_format=None):
	if context_format is None:
		context_format = ['Context', 'Response']

	# Normalize Text
	df_init = clean_data(df_init)
	df = df_init.copy()

	# For data 2, replace entities
	if 'text' in df.columns.tolist() and 'questionText' not in df.columns.tolist():
		# Replace entities
		df['text'] = df['text'].apply(replace_entities)
		df[['Context', 'Response']] = df['text'].str.split('\n', n=1, expand=True)
		# Apply the clean_text function
		df.loc[:, 'Context'] = df['Context'].apply(clean_text)
		df.loc[:, 'Response'] = df['Response'].apply(clean_text)
		df['Context'] = df['Context'].str.strip()
		df['Response'] = df['Response'].str.strip()

		df.drop('text', axis=1, inplace=True)
	else:
		# Add token
		df.loc[:, context_format[0]] = df[context_format[0]].apply(lambda x: "<user> " + str(x))
		df.loc[:, context_format[1]] = df[context_format[1]].apply(lambda x: "<agent> " + str(x))
		# Apply the clean_text function
		df.loc[:, context_format[0]] = df[context_format[0]].apply(clean_text)
		df.loc[:, context_format[1]] = df[context_format[1]].apply(clean_text)

	# Remove escape sequences
	df.loc[:, context_format[0]] = df[context_format[0]].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
	df.loc[:, context_format[1]] = df[context_format[1]].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

	# Check that every utterance start with a user or agent token
	assert all(df[context_format[0]].apply(lambda x: x.startswith('<user>') or x.startswith('<agent>'))), \
		'All utterances should start with <user> or <agent> token'
	utterances = [{
		"text": f"{row[context_format[0]].strip()} {row[context_format[1]].strip()}"
	} for index, row in df.iterrows()]

	return utterances


if __name__ == '__main__':
	# Load data
	dfs = []
	for dataset_path in tqdm([
		'Amod/mental_health_counseling_conversations',
		'heliosbrahma/mental_health_chatbot_dataset',
		'mpingale/mental-health-chat-dataset',
		'alexandreteles/mental-health-conversational-data'
	]):
		data = load_dataset(dataset_path)
		df = data['train'].to_pandas()
		dfs.append(df)

	# Preprocess data
	utterances1 = preprocess_data(dfs[0])
	utterances2 = preprocess_data(dfs[1])
	utterances3 = preprocess_data(dfs[2], context_format=['questionText', 'answerText'])
	utterances4 = preprocess_data(dfs[3])

	all_data = utterances1 + utterances2 + utterances3 + utterances4

	# train val split
	train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)
	output_dir = '/data/datasets/mental_health_dialogues/'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	pd.DataFrame(train_data).to_csv(f'{output_dir}train_dataset.csv', index=False)
	pd.DataFrame(val_data).to_csv(f'{output_dir}val_dataset.csv', index=False)
	print('done')
