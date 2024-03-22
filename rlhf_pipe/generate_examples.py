import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import firebase_admin
from firebase_admin import credentials, firestore

df_test = pd.read_csv('data/empathetic_dialogues/test_dataset.csv')

# generate examples from test data and upload to firestore
model_name = 'jeffreykthomas/llama-mental-health-chat'
# Load the base model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate examples
generated_examples = []
for idx, row in df_test.iterrows():
	context = row['text']
	input_ids = tokenizer(context, return_tensors='pt').input_ids
	generated_text = model.generate(input_ids, max_length=256, num_return_sequences=3, pad_token_id=tokenizer.eos_token_id)
	generated_text = tokenizer.batch_decode(generated_text, skip_special_tokens=True)[0]
	generated_examples.append(generated_text)

# Upload to firestore
cred = credentials.Certificate('firebase-credentials.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

for idx, row in df_test.iterrows():
	doc_ref = db.collection('generated_examples').document()
	doc_ref.set({
		'context': row['text'],
		'generated_text': generated_examples[idx]
	})