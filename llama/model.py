from typing import Optional, Tuple
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
import inspect


# Configuration for the Transformer model, specifying model dimensions, layer counts, etc.
# Based on LlaMA Architecture: https://arxiv.org/abs/2302.13971v1
# Git repo: https://github.com/facebookresearch/llama/blob/main/llama/model.py

@dataclass
class TransformerConfig:
    model_dimension: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    num_kv_heads: Optional[int] = None
    vocabulary_size: int = 32000
    hidden_dim: Optional[int] = None
    swiglu_multiple: int = 256  # Determines the size of the hidden layer in SwiGLU activation
    normalization_epsilon: float = 1e-5
    max_sequence_length: int = 2048
    dropout_rate: float = 0.0


# Layer normalization using Root Mean Square, for stabilizing the neural network's output
class RMSLayerNorm(nn.Module):
    """
    Layer normalization using Root Mean Square, for stabilizing the neural network's output.

    Args:
        dimension (int): The dimension of the input tensor.
        epsilon (float): A small value to prevent division by zero.

    Attributes:
        epsilon (float): A small value to prevent division by zero.
        scale (nn.Parameter): A learnable scale parameter for the layer normalization.
    """

    def __init__(self, dimension: int, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(dimension))

    def _norm(self, inputs):
        # Compute the variance of the input tensor
        variance = inputs.pow(2).mean(-1, keepdim=True)
        # Normalize the input tensor
        return inputs * torch.rsqrt(variance + self.epsilon)

    def forward(self, inputs):
        # Normalize the input features
        normalized_mean_square = self._norm(inputs.float()).type_as(inputs)
        return normalized_mean_square * self.scale


# Generates frequencies for rotary position embeddings
def compute_rotary_embedding_frequencies(dimension: int, sequence_end: int, scale_factor: float = 10000.0):
    """
    Generates frequencies for rotary position embeddings.

    Args:
        dimension (int): The dimension of the input tensor.
        sequence_end (int): The maximum sequence length.
        scale_factor (float): The scaling factor for the frequencies.

    Returns:
        torch.Tensor: A tensor containing the rotary position embeddings.
    """
    # Calculate frequency for each position in the input sequence
    position_indices = torch.arange(0, dimension, 2).float()[:dimension // 2]
    frequencies = 1.0 / (scale_factor ** (position_indices / dimension))
    time_steps = torch.arange(sequence_end)
    encoded_frequencies = torch.outer(time_steps, frequencies)
    freqs_cos = torch.cos(encoded_frequencies)
    freqs_sin = torch.sin(encoded_frequencies)
    return freqs_cos, freqs_sin


# Adjusts frequencies for broadcasting over input tensors
def adjust_for_broadcast(rotary_freq: torch.Tensor, inputs: torch.Tensor):
    """
    Adjusts frequencies for broadcasting over input tensors.

    Args:
        rotary_freq (torch.Tensor): The rotary frequencies to adjust.
        inputs (torch.Tensor): The input tensor to adjust for.

    Returns:
        torch.Tensor: The adjusted rotary frequencies.
    """
    num_dim = inputs.ndim
    assert 0 <= 1 < num_dim, "Input tensor must have at least 2 dimensions."
    assert rotary_freq.shape == (inputs.shape[1], inputs.shape[-1]), ("Rotary frequencies must have the same "
                                                                      "shape as the input tensor.")
    # Prepare the shape for broadcasting
    broadcast_shape = [dim if idx == 1 or idx == num_dim - 1 else 1 for idx, dim in enumerate(inputs.shape)]
    return rotary_freq.view(broadcast_shape)


# Applies rotary position embeddings to queries and keys
def apply_rotary_embeddings(
        query: torch.Tensor,
        key: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary position embeddings to queries and keys.

    Args:
        query (torch.Tensor): The query tensor to apply rotary embeddings to.
        key (torch.Tensor): The key tensor to apply rotary embeddings to.
        freqs_cos (torch.Tensor): The cosine rotary frequencies to apply.
        freqs_sin (torch.Tensor): The sine rotary frequencies to apply.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The query and key tensors with rotary embeddings applied.
    """
    # Convert queries and keys to complex numbers for rotation
    real_query, complex_query = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    real_key, complex_key = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    adjusted_frequencies_cos = adjust_for_broadcast(freqs_cos, real_query)
    adjusted_frequencies_sin = adjust_for_broadcast(freqs_sin, real_query)
    # Apply rotation and convert back to real numbers
    rotated_query_real = real_query * adjusted_frequencies_cos - complex_query * adjusted_frequencies_sin
    rotated_query_imaginary = real_query * adjusted_frequencies_sin + complex_query * adjusted_frequencies_cos
    rotated_key_real = real_key * adjusted_frequencies_cos - complex_key * adjusted_frequencies_sin
    rotated_key_imaginary = real_key * adjusted_frequencies_sin + complex_key * adjusted_frequencies_cos
    # Combine the real and imaginary parts of the rotated queries and keys
    rotated_query = torch.stack([rotated_query_real, rotated_query_imaginary], dim=-1).flatten(3)
    rotated_key = torch.stack([rotated_key_real, rotated_key_imaginary], dim=-1).flatten(3)
    return rotated_query.type_as(query), rotated_key.type_as(key)


def repeat_kv(inputs: torch.Tensor, num_rep: int) -> torch.Tensor:
    """
    Repeats each key-value pair in the input tensor a specified number of times.

    Args:
        inputs (torch.Tensor): The input tensor with dimensions [batch_size, seq_len, num_kv_heads, head_dim],
                          where:
                          - batch_size is the size of the batch,
                          - seq_len is the sequence length,
                          - num_kv_heads is the number of key-value pairs (attention heads),
                          - head_dim is the dimension of each head.
        num_rep (int): The number of times to repeat each key-value pair.

    Returns:
        torch.Tensor: The tensor after repeating key-value pairs, with the updated dimension for num_kv_heads.

    If num_rep is 1, the function returns the input tensor unchanged. Otherwise, it repeats each key-value pair
    'num_rep' times along the num_kv_heads dimension, effectively increasing the number of key-value pairs by 'num_rep' times.
    """

    # Extract the dimensions of the input tensor
    batch_size, seq_len, num_kv_heads, head_dim = inputs.shape

    # If no repetition is needed, return the input tensor as is
    if num_rep == 1:
        return inputs

    # Repeat each key-value pair 'num_rep' times:
    # 1. Add an extra dimension to make room for the repetitions.
    # 2. Use .expand() to repeat the elements across the new dimension without copying data (for efficiency).
    # 3. Reshape the tensor back to a 4D tensor, combining the repeated key-value pairs into the num_kv_heads dimension.
    return (
        inputs.unsqueeze(3)  # Insert a new dimension for repetition
        .expand(batch_size, seq_len, num_kv_heads, num_rep, head_dim)  # Repeat KV pairs 'num_rep' times
        .reshape(batch_size, seq_len, num_kv_heads * num_rep, head_dim)  # Merge the repeated KV pairs into num_kv_heads
    )


# Multi-head attention mechanism for processing inputs in parallel
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for processing inputs in parallel.

    Args:
        config (TransformerConfig): The configuration for the Transformer model.

    Attributes:
        num_heads (int): The number of attention heads.
        dimension_per_head (int): The dimension per attention head.
        query_weight (nn.Linear): The weight matrix for the query projection.
        key_weight (nn.Linear): The weight matrix for the key projection.
        value_weight (nn.Linear): The weight matrix for the value projection.
        output_weight (nn.Linear): The weight matrix for the output projection.
        key_cache (torch.Tensor): The cache for the key tensors.
        value_cache (torch.Tensor): The cache for the value tensors.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_kv_heads = config.num_attention_heads if config.num_kv_heads is None else config.num_kv_heads
        assert config.num_attention_heads % self.num_kv_heads == 0, (
            "The number of attention heads must be divisible by "
            "the number of key-value heads.")
        model_parallel_size = 1
        self.num_heads = config.num_attention_heads
        self.num_local_heads = config.num_attention_heads // model_parallel_size
        self.num_local_kv_heads = self.num_kv_heads // model_parallel_size
        self.num_rep = self.num_local_heads // self.num_local_kv_heads
        self.dimension_per_head = config.model_dimension // self.num_heads

        # Initialize weights for query, key, value, and output projections
        self.query_weight = nn.Linear(config.model_dimension, self.num_heads * self.dimension_per_head, bias=False)
        self.key_weight = nn.Linear(config.model_dimension, self.num_kv_heads * self.dimension_per_head, bias=False)
        self.value_weight = nn.Linear(config.model_dimension, self.num_kv_heads * self.dimension_per_head, bias=False)
        self.output_weight = nn.Linear(self.num_heads * self.dimension_per_head, config.model_dimension, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout_rate)
        self.residual_dropout = nn.Dropout(config.dropout_rate)
        self.dropout = config.dropout_rate

        # Pre-allocate memory for key and value caches
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print('WARNING: Using PyTorch native attention. This is not recommended for large models. '
                  'Upgrade to PyTorch >= 2.0')
        mask = torch.full((1, 1, config.max_sequence_length, config.max_sequence_length), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(
        self,
        inputs: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        # Compute query, key, and value projections
        batch_size, sequence_length, _ = inputs.shape
        query, key, value = self.query_weight(inputs), self.key_weight(inputs), self.value_weight(inputs)
        query = query.view(batch_size, sequence_length, self.num_local_heads, self.dimension_per_head)
        key = key.view(batch_size, sequence_length, self.num_local_kv_heads, self.dimension_per_head)
        value = value.view(batch_size, sequence_length, self.num_local_kv_heads, self.dimension_per_head)
        # Apply rotary embeddings to query and key
        query, key = apply_rotary_embeddings(query, key, freqs_cos, freqs_sin)

        # apply the repeat_kv function to the key and value tensors
        key = repeat_kv(key, self.num_rep)
        value = repeat_kv(value, self.num_rep)

        # make head dimension into batch dimension
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # flash attention implementation
        if self.flash:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # manual attention implementation
            scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.dimension_per_head)
            assert hasattr(self, 'mask'), 'Mask not found'
            scores = scores + self.mask[:, :, :sequence_length, :sequence_length]
            scores = F.softmax(scores.float(), dim=-1).type_as(query)
            scores = self.attn_dropout(scores)
            attn_output = torch.matmul(scores, value)

        # Merge the heads back into the batch dimension
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)

        # Apply output projection and residual dropout
        attn_output = self.output_weight(attn_output)
        attn_output = self.residual_dropout(attn_output)
        return attn_output


# Feed-forward network used within each Transformer block
class FeedForwardNetwork(nn.Module):
    """
    Feed-forward network used within each Transformer block.

    Args:
        dimension (int): The dimension of the input tensor.
        hidden_dimension (int): The dimension of the hidden layer.
        rounding_multiple (int): The rounding multiple for the hidden layer dimension.

    Attributes:
        first_linear (nn.Linear): The first linear layer for the feed-forward network.
        second_linear (nn.Linear): The second linear layer for the feed-forward network.
        third_linear (nn.Linear): The third linear layer for the feed-forward network.
    """

    def __init__(self, dimension: int, hidden_dimension: int, rounding_multiple: int, dropout_rate: float = 0.0):
        super().__init__()
        if hidden_dimension is None:
            hidden_dimension = dimension * 4
            hidden_dimension = int(2 * hidden_dimension / 3)
            hidden_dimension = rounding_multiple * ((hidden_dimension + rounding_multiple - 1) // rounding_multiple)

        # Initialize linear layers for feed-forward network
        self.first_linear = nn.Linear(dimension, hidden_dimension, bias=False)
        self.second_linear = nn.Linear(hidden_dimension, dimension, bias=False)
        self.third_linear = nn.Linear(dimension, hidden_dimension, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        # Apply SwiGLU activation function
        return self.dropout(self.second_linear(F.silu(self.first_linear(inputs)) * self.third_linear(inputs)))


# Represents a single Transformer block, combining attention and feed-forward networks
class TransformerLayer(nn.Module):
    """
    Represents a single Transformer block, combining attention and feed-forward networks.

    Args:
        config (TransformerConfig): The configuration for the Transformer model.

    Attributes:
        attention (MultiHeadAttention): The multi-head attention mechanism.
        attention_norm (RMSLayerNorm): The layer normalization for attention output.
        feed_forward (FeedForwardNetwork): The feed-forward network.
        feed_forward_norm (RMSLayerNorm): The layer normalization for feed-forward output.
    """

    def __init__(self, layer_id: int, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.dimension = config.model_dimension
        self.dimension_per_head = config.model_dimension // config.num_attention_heads
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForwardNetwork(
            config.model_dimension,
            config.hidden_dim,
            config.swiglu_multiple,
            config.dropout_rate)
        self.layer_id = layer_id
        self.attention_norm = RMSLayerNorm(config.model_dimension, epsilon=config.normalization_epsilon)
        self.feed_forward_norm = RMSLayerNorm(config.model_dimension, epsilon=config.normalization_epsilon)

    def forward(self, inputs, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        # Apply multi-head attention and layer normalization
        h = inputs + self.attention.forward(self.attention_norm(inputs), freqs_cos, freqs_sin)
        # Apply feed-forward network and layer normalization
        attention_output = self.feed_forward.forward(self.feed_forward_norm(h))
        return attention_output


# Complete Transformer model, stacking multiple Transformer layers
class TransformerModel(nn.Module):
    """
    Complete Transformer model, stacking multiple Transformer layers.

    Args:
        config (TransformerConfig): The configuration for the Transformer model.

    Attributes:
        embedding (nn.Embedding): The input token embedding layer.
        positional_embedding (torch.Tensor): The positional embedding tensor.
        layers (nn.ModuleList): The list of Transformer layers.
        final_norm (RMSLayerNorm): The final layer normalization.
        output_projection (nn.Linear): The output projection layer.
    """
    last_loss: Optional[torch.Tensor]

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.vocabulary_size = config.vocabulary_size
        self.num_layers = config.num_layers

        self.tok_embeddings = nn.Embedding(config.vocabulary_size, config.model_dimension)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.num_layers):
            self.layers.append(TransformerLayer(layer_id, config))
        self.norm = RMSLayerNorm(config.model_dimension, epsilon=config.normalization_epsilon)
        self.output = nn.Linear(config.model_dimension, config.vocabulary_size, bias=False)

        # Share weights between token embeddings and output projection
        self.tok_embeddings.weight = self.output.weight  # https://paperswithcode.com/method/weight-tying

        # Precompute frequencies for rotary position embeddings
        freqs_cos, freqs_sin = compute_rotary_embedding_frequencies(
            self.config.model_dimension // self.config.num_attention_heads, self.config.max_sequence_length)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # Apply weight initialization to the output layer
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

        # Initialize attribute for the loss of the last forward call.
        # This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _batch_size, seq_length = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seq_length]
        freqs_sin = self.freqs_sin[:seq_length]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :])  # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
