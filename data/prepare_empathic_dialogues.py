import os
import re
import pandas as pd
from datasets import load_dataset

user_token = '<user>'
agent_token = '<agent>'
train_data = []
val_data = []
test_data = []
all_data = [train_data, val_data, test_data]


def remove_user_turn_if_last(example, user_token, agent_token):
	if example.rfind(user_token) > example.rfind(agent_token):
		example = example[:example.rfind(user_token)]
	return example


def prepare_empathetic_dialogues():
	df_train = open('data/empatheticdialogues/train.csv').readlines()
	df_val = open('data/empatheticdialogues/valid.csv').readlines()
	df_test = open('data/empatheticdialogues/test.csv').readlines()
	dfs = [df_train, df_val, df_test]
	current_conversation = ''
	current_conv_idx = ''

	for split, df in enumerate(dfs):
		for i in range(1, len(df)):
			new_turn = df[i].strip().split(',')
			new_utterance = new_turn[5].replace('_comma_', ',').strip()
			if new_turn[0] == current_conv_idx:
				utterance_idx = int(new_turn[1])
				if (utterance_idx % 2) == 0:
					current_conversation += ' ' + agent_token + ' ' + new_utterance
				else:
					current_conversation += ' ' + user_token + ' ' + new_utterance
			else:
				if current_conversation != '':
					# if user has the last turn, remove the last turn
					current_conversation = remove_user_turn_if_last(current_conversation, user_token, agent_token)
					all_data[split].append({"text": current_conversation})
				current_conv_idx = new_turn[0]
				current_conversation = user_token + ' ' + new_utterance

			if i == len(df) - 1:
				# if user has the last turn, remove the last turn
				current_conversation = remove_user_turn_if_last(current_conversation, user_token, agent_token)
				all_data[split].append({"text": current_conversation})


def clean_punctuation(text):
	# Handles general punctuation (space after, but not before)
	text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
	# Corrects space around apostrophes for contractions (e.g., "it 's" -> "it's")
	text = re.sub(r"(\w)\s+'\s*(\w)", r"\1'\2", text)
	# Additional handling for standalone contractions and possessives, expand as needed
	text = re.sub(r"(\w)\s+'\s*(re|s|t|ll|d|m|ve|ll)\b", r"\1'\2", text, flags=re.IGNORECASE)
	# Handles the removal of incorrect spaces before structural elements or end of text
	text = re.sub(r'\s+(?=<|$)', '', text)
	return text


def prepare_daily_dialog():
	dataset = load_dataset('daily_dialog')
	for i, split in enumerate(dataset):
		for j in range(len(dataset[split])):
			utterances = dataset[split][j]['dialog']
			utterances = [clean_punctuation(utterance.strip()) for utterance in utterances]
			current_conversation = ''
			for k in range(len(utterances)):
				if k % 2 == 0:
					current_conversation += ' ' + user_token + ' ' + utterances[k]
				else:
					current_conversation += ' ' + agent_token + ' ' + utterances[k]
			# if user has the last turn, remove the last turn
			current_conversation = remove_user_turn_if_last(current_conversation, user_token, agent_token)
			current_conversation = current_conversation.strip()
			all_data[i].append({"text": current_conversation})


if __name__ == '__main__':
	prepare_empathetic_dialogues()
	prepare_daily_dialog()
	# Save the data
	output_dir = '/data/datasets/dialogues/'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	# Save the data
	train_data = train_data + test_data
	pd.DataFrame(train_data).to_csv(f'{output_dir}train_dataset.csv', index=False)
	pd.DataFrame(val_data).to_csv(f'{output_dir}val_dataset.csv', index=False)
	print('done')
