import os
import pandas as pd

df_train = open('data/empatheticdialogues/train.csv').readlines()
df_val = open('data/empatheticdialogues/valid.csv').readlines()
df_test = open('data/empatheticdialogues/test.csv').readlines()
dfs = {
	'test': df_test,
	'train': df_train,
	'val': df_val
}
data = []
current_conversation = ''
current_conv_idx = ''
user_token = '<user>'
agent_token = '<agent>'


def remove_user_turn_if_last(example, user_token, agent_token):
	if example.rfind(user_token) > example.rfind(agent_token):
		example = example[:example.rfind(user_token)]
	return example


for split in dfs.items():
	df = split[1]
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
				data.append({"text": current_conversation})
			current_conv_idx = new_turn[0]
			current_conversation = user_token + ' ' + new_utterance

		if i == len(df) - 1:
			# if user has the last turn, remove the last turn
			current_conversation = remove_user_turn_if_last(current_conversation, user_token, agent_token)
			data.append({"text": current_conversation})

	if split[0] == 'test':
		continue

	output_dir = '/data/datasets/empathetic-dialogues/'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	# Save the data
	pd.DataFrame(data).to_csv(f'{output_dir}{split[0]}_dataset.csv', index=False)
	data = []
