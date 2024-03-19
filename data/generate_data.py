import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tqdm
import pandas as pd

prompts = [
	"Output only an example of a chat message sent by someone suffering from depression and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from generalized anxiety disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from borderline personality disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from bipolar disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from post-traumatic stress disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from social anxiety disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from obsessive-compulsive disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from panic disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from avoidant personality disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from dysthymia and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from schizophrenia and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from attention-deficit/hyperactivity disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from autism spectrum disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from eating disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from substance use disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from somatic symptom disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from dissociative identity disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from narcissistic personality disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from histrionic personality disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from antisocial personality disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from cyclothymic disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from acute stress disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from adjustment disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from paranoid personality disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from schizoid personality disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from schizotypal personality disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from dependent personality disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from obsessive-compulsive personality disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from postpartum depression and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from premenstrual dysphoric disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from seasonal affective disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from body dysmorphic disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from hoarding disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from trichotillomania (hair-pulling disorder) and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from excoriation (skin-picking) disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from gambling disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from internet gaming disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from delusional disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from brief psychotic disorder and nothing else. Format your output in quotation marks.",
	"Output only an example of a chat message sent by someone suffering from reactive attachment disorder and nothing else. Format your output in quotation marks."
]
chat_models = [
	'meta-llama/Llama-2-7b-chat-hf',
	'meta-llama/Llama-2-13b-chat-hf',
	'mistralai/Mistral-7B-Instruct-v0.2',
	'Qwen/Qwen1.5-0.5B-Chat']


num_iterations = 300
num_responses_per_prompt = 3


def generate_user_messages(model_name, num_iterations, num_responses_per_prompt, prompts, batch_size=1):
	model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer.pad_token = tokenizer.eos_token
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	all_outputs = []

	for _ in tqdm.tqdm(range(0, num_iterations, batch_size)):
		# choose a random batch of prompts
		batch_prompts = random.sample(prompts, batch_size)
		if not batch_prompts:
			print("All prompts have been processed.")
			continue
		input = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

		# Randomly select temp, top_k, and top_p
		temperature = random.uniform(0.6, 0.8)
		top_k = random.randint(46, 60)
		top_p = random.uniform(0.8, 0.95)

		output = model.generate(
			**input,
			max_length=150,
			num_return_sequences=num_responses_per_prompt,
			temperature=temperature,
			do_sample=True,
			pad_token_id=tokenizer.eos_token_id,
			top_k=top_k,
			top_p=top_p
		)
		for i, output in enumerate(output):
			response = tokenizer.decode(output, skip_special_tokens=True)
			print(f"Output: {response}")
			all_outputs.append(response)

	all_data = pd.DataFrame(all_outputs, columns=["text"])
	precise_model_name = model_name.split('/')[1]
	all_data['model'] = precise_model_name
	all_data.to_csv(f'{precise_model_name}_responses.csv', index=False, encoding='utf-8')

	print("All done!")


def clean_user_messages(file):
	df = pd.read_csv(file)
	# Extract the chat messages from the text column
	df['chat1'] = df['text'].str.extract(r'\"(.*?)\"')
	df['chat2'] = df['text'].str.extract(r'\"(.*?)$')
	df['chat3'] = df['text'].str.extract(r'User:(.*?)$')
	df['chat4'] = df['text'].str.extract(r'Example:(.*?)$')

	# Combine the chat messages into a single column
	df['final_chat'] = df['chat1'].fillna(df['chat2']).fillna(df['chat3']).fillna(df['chat4'])

	# Drop the unnecessary columns and any rows with missing chat messages
	df = df.drop(columns=['text', 'chat1', 'chat2', 'chat3', 'chat4'])
	df = df.dropna(subset=['final_chat'])

	# Save the cleaned data to a new file
	df.to_csv(f'cleaned_{file}', index=False, encoding='utf-8')


def generate_expert_messages(model_name, input_file):
	model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# Load the cleaned user messages
	df = pd.read_csv(input_file)
	all_outputs = []
	expert_prompt = '''
	<s>[INST] <<SYS>> As an expert in mental health, respond with advice to the user message
	<</SYS>> 
	'''
	for i, prompt in enumerate(df['final_chat']):
		print(f"Prompt {i + 1}/{len(df)}: {prompt}")
		input_prompt = expert_prompt + prompt + '[/INST]'
		input = tokenizer(input_prompt, return_tensors="pt").to(device)

		# Randomly select temp, top_k, and top_p
		temperature = random.uniform(0.6, 0.8)
		top_k = random.randint(46, 60)
		top_p = random.uniform(0.8, 0.95)

		output = model.generate(
			**input,
			max_length=512,
			num_return_sequences=1,
			temperature=temperature,
			do_sample=True,
			pad_token_id=tokenizer.eos_token_id,
			top_k=top_k,
			top_p=top_p
		)
		response = tokenizer.decode(output[0], skip_special_tokens=True)
		print(f"Output: {response}")
		all_outputs.append(response)

	all_data = pd.DataFrame(all_outputs, columns=["text"])
	all_data['model'] = model_name
	all_data.to_csv(f'{model_name}_responses_w_expert.csv', index=False, encoding='utf-8')

	print("All done!")


if __name__ == '__main__':
	for model in chat_models:
		if model == 'meta-llama/Llama-2-7b-chat-hf':
			precise_model_name = model.split('/')[1]
			generate_expert_messages(model, f'cleaned_{precise_model_name}_responses.csv')
		else:
			generate_user_messages(model, num_iterations, num_responses_per_prompt, prompts, batch_size=4)
			precise_model_name = model.split('/')[1]
			clean_user_messages(f'{precise_model_name}_responses.csv')
			generate_expert_messages(model, f'cleaned_{precise_model_name}_responses.csv')
