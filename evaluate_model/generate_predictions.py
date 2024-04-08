import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--model_path', type=str, default='Qwen/Qwen1.5-0.5B-Chat')

args = arg_parser.parse_args()

df = pd.read_csv('data/datasets/RogueEval/predictions.csv')
df = df.sample(100)
df = df.reset_index(drop=True)


def load_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    padding_side = 'left' if model_path == 'Qwen/Qwen1.5-0.5B-Chat' else 'right'
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=padding_side)
    tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token

    model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_predictions(model, tokenizer, device, texts, batch_size=4):
    message_batch = []
    for i in range(batch_size):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": texts[i]}
        ]
        message_batch.append(messages)
    texts = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in message_batch]

    model_inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=1024, 
        add_special_tokens=True if model.config.architectures == ['Qwen2ForCausalLM'] else False
    )
    model_input_ids = model_inputs.input_ids.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            model_input_ids,
            max_new_tokens=512
        )

    # Decode the trimmed output token IDs to text for the entire batch
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Remove text from texts from the generated texts
    for i in range(len(generated_texts)):
        text_to_replace = texts[i].replace('<|im_start|>', '').replace('<|im_end|>', '')
        generated_texts[i] = generated_texts[i].replace(text_to_replace, '').strip()

    return generated_texts


def run_generation(text_list, num_batches=25, batch_size=4, model_path='Qwen/Qwen1.5-0.5B-Chat'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, tokenizer = load_model(model_path, device)
    all_generated_texts = []
    num_rows = len(text_list)
    for batch_num in tqdm(range(num_batches)):
        # Calculate start and end index of the current batch
        print(f'Batch: {batch_num}/{num_batches}')
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, num_rows)
        # Extract the texts for the current batch from the DataFrame
        current_batch_texts = text_list[start_idx:end_idx]

        # Adjust the batch size if we're on the last batch, and it's smaller than the batch_size
        current_batch_size = len(current_batch_texts)
        if current_batch_size < batch_size:
            # Update the function call to handle a smaller batch if necessary
            # This might involve modifying your function to dynamically adjust to batch size
            generated_texts = generate_predictions(model, tokenizer, device, current_batch_texts, current_batch_size)
        else:
            generated_texts = generate_predictions(model, tokenizer, device, current_batch_texts)

        # Append the generated texts from the current batch to the all_generated_texts list
        all_generated_texts.extend(generated_texts)

        return all_generated_texts


if __name__ == '__main__':
    comparison_text = run_generation(df['Input'].tolist())
    df['Comparison_Predictions'] = comparison_text
    df.to_csv('data/datasets/RogueEval/predictions_with_comparison.csv', index=False)