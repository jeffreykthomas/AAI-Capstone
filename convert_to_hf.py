import os
import torch
from transformers import LlamaConfig
from llama import model as base_model


def hf_export(llama_model, filepath, step_num, group_size=64, dtype=torch.float32):
    """ Generate the pytorch_model.bin state_dict and config.json for HuggingFace """

    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    # Generate LlamaModel state_dict
    hf_state_dict = {}
    # Sometimes we have repeated key values for the heads
    dim = llama_model.params.model_dimension
    num_key_value_heads = llama_model.params.n_kv_heads
    n_rep = llama_model.params.num_heads // num_key_value_heads
    key_value_dim = dim // n_rep

    # HuggingFace needs the weights permuted.
    # See: https://github.com/huggingface/transformers/blob/b132c1703eb1c8bd9dfa4ad6a9be2bfd6ef819e9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122
    def permute_original(w, n_heads=llama_model.params.num_heads, dim1=dim, dim2=dim):
        return w.view(dim1, dim2).reshape(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # Transfer weights from llama model to the HF state dictionary format
    hf_state_dict['model.embed_tokens.weight'] = llama_model.tok_embeddings.weight.clone().to(dtype)
    hf_state_dict['model.norm.weight'] = llama_model.norm.weight.clone().to(dtype)

    # Add each layer's weights to the HF state dictionary
    for i, layer in enumerate(llama_model.layers):
        layer_id = layer.layer_id
        hf_state_dict[f'model.layers.{i}.input_layernorm.weight'] = llama_model.layers[layer_id].attention_norm.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.q_proj.weight'] = permute_original(llama_model.layers[layer_id].attention.wq.weight.clone()).to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.k_proj.weight'] = permute_original(llama_model.layers[layer_id].attention.wk.weight.clone(), num_key_value_heads, key_value_dim, dim).to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.v_proj.weight'] = llama_model.layers[layer_id].attention.wv.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.self_attn.o_proj.weight'] = llama_model.layers[layer_id].attention.wo.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.post_attention_layernorm.weight'] = llama_model.layers[layer_id].ffn_norm.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.mlp.gate_proj.weight'] = llama_model.layers[layer_id].feed_forward.w1.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.mlp.down_proj.weight'] = llama_model.layers[layer_id].feed_forward.w2.weight.clone().to(dtype)
        hf_state_dict[f'model.layers.{i}.mlp.up_proj.weight'] = llama_model.layers[layer_id].feed_forward.w3.weight.clone().to(dtype)

    # llama2.c usually uses tied weights -> reference the embed_tokens.weights instead
    hf_state_dict['lm_head.weight'] = hf_state_dict['model.embed_tokens.weight']

    # We check that the embeddings are tied, else use manual output weights
    _embeddings_are_tied: bool = torch.equal(llama_model.tok_embeddings.weight, llama_model.output.weight)
    if not _embeddings_are_tied:
        hf_state_dict['lm_head.weight'] = llama_model.output.weight.clone().to(dtype)

    # Generate LlamaConfig (seen in transformers.models.llama.configuration_llama)

    # Extract necessary attributes from llama.c model
    print(f'Saving config, vocab size: {llama_model.params.vocab_size}, '
          f'hidden_size: {llama_model.params.model_dimension}')
    vocab_size = llama_model.params.vocab_size
    hidden_size = llama_model.params.model_dimension
    intermediate_size = llama_model.layers[0].feed_forward.w1.weight.shape[0]
    num_hidden_layers = llama_model.params.num_layers
    num_attention_heads = llama_model.params.num_heads
    num_key_value_heads = llama_model.params.n_kv_heads
    max_position_embeddings = llama_model.params.max_seq_len
    rms_norm_eps = llama_model.params.norm_eps

    # TODO check values for:
    # pretraining_tp, initializer_range, use_cache,
    # rope_theta, and rope_scaling.

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        tie_word_embeddings=_embeddings_are_tied,
        # Manual
        architectures=["LlamaForCausalLM"],
        hidden_act="silu",
    )

    # Save files in directory filepath
    # First make the directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)

    # Save the state dictionary in .bin format, and config as .json
    torch.save(hf_state_dict, os.path.join(filepath, f"pytorch_model_step_{step_num}.bin"))
    config.save_pretrained(filepath)


if __name__ == "__main__":
    model_config = base_model.TransformerConfig()
    model_config.model_dimension = 2048
    model_config.num_layers = 16
    model_config.num_heads = 16
    model_config.n_kv_heads = 16
    model_config.vocabulary_size = 32000
    model_config.multiple_of = 32
    model_config.norm_eps = 1e-5
    model_config.max_seq_len = 1024
    model_config.dropout = 0.1

    print(model_config)

    model = base_model.Transformer(model_config)

    step_num = 11001

    ckpt_path = f'/data/models/llama_health/model_step_{step_num}.pt'
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    state_dict = checkpoint['model']

    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

    hf_export(model, '/data/models/llama-mental-health/', step_num)
