import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

df_comparison = pd.read_csv('data/datasets/RogueEval/predictions.csv')
df_comparison = df_comparison.sample(100)
df_comparison = df_comparison.reset_index(drop=True)

comparison_path = 'Qwen/Qwen1.5-0.5B-Chat'
comparison_model = AutoModelForCausalLM.from_pretrained(comparison_path)
comparison_tokenizer = AutoTokenizer.from_pretrained(comparison_path)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
comparison_model = comparison_model.to(device)
comparison_model.eval()
num_batches = 25
batch_size = 4
num_rows = len(df_comparison)


def calculate_comparison_predictions(texts, batch_size=4):
    message_batch = []
    for i in range(batch_size):
        print(f'Adding message {texts[i]}')
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": texts[i]}
        ]
        message_batch.append(messages)
    texts = [comparison_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in message_batch]

    model_inputs = comparison_tokenizer(texts, return_tensors="pt", padding=True)
    model_input_ids = model_inputs.input_ids.to(device)
    with torch.no_grad():
        generated_ids = comparison_model.generate(
            model_input_ids,
            max_new_tokens=512
        )

    # Decode the trimmed output token IDs to text for the entire batch
    generated_texts = comparison_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_texts


if __name__ == '__main__':
    all_generated_texts = []
    for batch_num in range(num_batches):
        # Calculate start and end index of the current batch
        print(f'Batch: {batch_num}/{num_batches}')
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, num_rows)
        print(f'Start: {start_idx}, End: {end_idx}')
        # Extract the texts for the current batch from the DataFrame
        current_batch_texts = df_comparison['Input'][start_idx:end_idx].tolist()

        # Adjust the batch size if we're on the last batch and it's smaller than the batch_size
        current_batch_size = len(current_batch_texts)
        if current_batch_size < batch_size:
            # Update the function call to handle a smaller batch if necessary
            # This might involve modifying your function to dynamically adjust to batch size
            generated_texts = calculate_comparison_predictions(current_batch_texts, current_batch_size)
        else:
            generated_texts = calculate_comparison_predictions(current_batch_texts)

        # Append the generated texts from the current batch to the all_generated_texts list
        all_generated_texts.extend(generated_texts)
        print(f'Texts: {[text[:50] for text in generated_texts]}')
    df_comparison['Comparison_Predictions'] = all_generated_texts
    df_comparison.to_csv('data/datasets/RogueEval/predictions_with_comparison.csv', index=False)