import pandas as pd
import torch
from llama.load_model import load_model
from llama.tokenizer import Tokenizer
import firebase_admin
from firebase_admin import credentials, firestore
import json

df_test = pd.read_csv('data/empathetic_dialogues/val_dataset.csv')

# generate examples from test data and upload to firestore
checkpoint_path = 'llama/models/mh_fine_tune.pt'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

model = load_model(checkpoint_path, map_location=device)
print(f"Using device: {device}")
model.to(device)
tokenizer = Tokenizer('llama/models/tokenizer.model')
model.eval()

# Generate examples
generated_examples = []
for idx, row in df_test.iterrows():
	print(f"Generating examples for prompt {idx} of {len(df_test)}")
	context = row['text']
	user_message, _ = context.split(' <agent>', 1)
	user_message = user_message + ' <agent>'
	input_ids = tokenizer.encode(user_message, eos=True, bos=False)
	input_tensors = torch.tensor(input_ids).unsqueeze(0).to(device)
	with torch.no_grad():
		generated_text_1 = model.generate(input_tensors, max_new_tokens=256)
		generated_text_2 = model.generate(input_tensors, max_new_tokens=256)
	decoded_text_1 = tokenizer.decode(generated_text_1.cpu().numpy().tolist()[0])
	decoded_text_2 = tokenizer.decode(generated_text_2.cpu().numpy().tolist()[0])
	generated_examples.append({
		'prompt': user_message,
		'responses': [decoded_text_1, decoded_text_2]
	})
	print(f"Generated examples for prompt: {user_message}")
	print(f"Response 1: {decoded_text_1}")
	print(f"Response 2: {decoded_text_2}")

# Save generated examples to file
with open('data/generated_examples.json', 'w') as f:
	json.dump(generated_examples, f)

# Upload to firestore
cred = credentials.Certificate('rlhf_pipe/jt-designs-79-firebase-adminsdk-r3inm-6192f78083.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# set previous prompts to not current
prompts_ref = db.collection('prompts')
prompts = prompts_ref.where('current', '==', True).stream()
for prompt in prompts:
	prompt.reference.update({
		'current': False
	})

doc_ref = db.collection('prompts').document()
doc_ref.set({
	'current': True,
	'prompts': generated_examples
})