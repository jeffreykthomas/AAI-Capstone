from typing import Optional, Tuple
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F


# Configuration for the Transformer model, specifying model dimensions, layer counts, etc.
# Based on LlaMA Architecture: https://arxiv.org/abs/2302.13971v1
# Git repo: https://github.com/facebookresearch/llama/blob/main/llama/model.py

@dataclass
class TransformerConfig:
	model_dimension: int = 512
	num_layers: int = 8
	num_attention_heads: int = 8
	vocabulary_size: int = -1  # To be updated based on the specific vocabulary size
	swiglu_multiple: int = 256  # Determines the size of the hidden layer in SwiGLU activation
	normalization_epsilon: float = 1e-5
	max_batch_size: int = 32
	max_sequence_length: int = 256


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

	def forward(self, inputs):
		# Normalize the input features
		mean_square = inputs.pow(2).mean(-1, keepdim=True)
		normalized_inputs = inputs * torch.rsqrt(mean_square + self.epsilon)
		return normalized_inputs * self.scale


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
	return torch.polar(torch.ones_like(encoded_frequencies), encoded_frequencies)


# Adjusts frequencies for broadcasting over input tensors
def adjust_for_broadcast(rotary_frequencies: torch.Tensor, inputs: torch.Tensor):
	"""
	Adjusts frequencies for broadcasting over input tensors.

	Args:
		rotary_frequencies (torch.Tensor): The rotary frequencies to adjust.
		inputs (torch.Tensor): The input tensor to adjust for.

	Returns:
		torch.Tensor: The adjusted rotary frequencies.
	"""
	# Prepare the shape for broadcasting
	broadcast_shape = [dimension if index in [1, -1] else 1 for index, dimension in enumerate(inputs.shape)]
	return rotary_frequencies.view(*broadcast_shape)


# Applies rotary position embeddings to queries and keys
def apply_rotary_embeddings(
		query: torch.Tensor,
		key: torch.Tensor,
		frequencies: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Applies rotary position embeddings to queries and keys.

	Args:
		query (torch.Tensor): The query tensor to apply rotary embeddings to.
		key (torch.Tensor): The key tensor to apply rotary embeddings to.
		frequencies (torch.Tensor): The rotary frequencies to apply.

	Returns:
		Tuple[torch.Tensor, torch.Tensor]: The query and key tensors with rotary embeddings applied.
	"""
	# Convert queries and keys to complex numbers for rotation
	complex_query = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
	complex_key = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
	adjusted_frequencies = adjust_for_broadcast(frequencies, complex_query)
	# Apply rotation and convert back to real numbers
	rotated_query = torch.view_as_real(complex_query * adjusted_frequencies).flatten(3)
	rotated_key = torch.view_as_real(complex_key * adjusted_frequencies).flatten(3)
	return rotated_query, rotated_key


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
		self.num_heads = config.num_attention_heads
		self.dimension_per_head = config.model_dimension // self.num_heads

		# Initialize weights for query, key, value, and output projections
		self.query_weight = nn.Linear(config.model_dimension, self.num_heads * self.dimension_per_head, bias=False)
		self.key_weight = nn.Linear(config.model_dimension, self.num_heads * self.dimension_per_head, bias=False)
		self.value_weight = nn.Linear(config.model_dimension, self.num_heads * self.dimension_per_head, bias=False)
		self.output_weight = nn.Linear(self.num_heads * self.dimension_per_head, config.model_dimension, bias=False)

		# Pre-allocate memory for key and value caches
		self.key_cache = torch.zeros(
			config.max_batch_size,
			config.max_sequence_length,
			self.num_heads,
			self.dimension_per_head).cuda()
		self.value_cache = torch.zeros(
			config.max_batch_size,
			config.max_sequence_length,
			self.num_heads,
			self.dimension_per_head).cuda()

	def forward(self, inputs, start_position, rotary_frequencies, attention_mask: Optional[torch.Tensor] = None):
		# Compute query, key, and value projections
		batch_size, sequence_length, _ = inputs.shape
		query, key, value = self.query_weight(inputs), self.key_weight(inputs), self.value_weight(inputs)
		query, key, value = [x.view(batch_size, sequence_length, self.num_heads, self.dimension_per_head) for x in
							[query, key, value]]

		# Apply rotary embeddings to query and key
		query, key = apply_rotary_embeddings(query, key, frequencies=rotary_frequencies)

		# Update caches with the current key and value tensors
		self.key_cache = self.key_cache.to(inputs.device)
		self.value_cache = self.value_cache.to(inputs.device)
		self.key_cache[:, start_position:start_position + sequence_length] = key
		self.value_cache[:, start_position:start_position + sequence_length] = value

		# Retrieve cached keys and values for attention calculation
		cached_keys = self.key_cache[:, :start_position + sequence_length]
		cached_values = self.value_cache[:, :start_position + sequence_length]

		# Transpose for batched matrix multiplication
		query, cached_keys, cached_values = [x.transpose(1, 2) for x in [query, cached_keys, cached_values]]
		# Compute attention scores and apply mask if provided
		attention_scores = torch.matmul(query, cached_keys.transpose(2, 3)) / math.sqrt(self.dimension_per_head)
		if attention_mask is not None:
			attention_scores += attention_mask
		attention_weights = F.softmax(attention_scores, dim=-1)

		# Aggregate attention and project output
		attention_output = torch.matmul(attention_weights, cached_values)
		attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
		return self.output_weight(attention_output)


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
	def __init__(self, dimension: int, hidden_dimension: int, rounding_multiple: int):
		super().__init__()

		# Adjust hidden dimension size based on rounding multiple
		adjusted_hidden_dimension = int(2 * hidden_dimension / 3)
		adjusted_hidden_dimension = rounding_multiple * (
				(adjusted_hidden_dimension + rounding_multiple - 1) // rounding_multiple)

		# Initialize linear layers for feed-forward network
		self.first_linear = nn.Linear(dimension, adjusted_hidden_dimension, bias=False)
		self.second_linear = nn.Linear(adjusted_hidden_dimension, dimension, bias=False)
		self.third_linear = nn.Linear(dimension, adjusted_hidden_dimension, bias=False)

	def forward(self, inputs):
		# Apply SwiGLU activation function
		return self.second_linear(F.silu(self.first_linear(inputs)) * self.third_linear(inputs))


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
	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.attention = MultiHeadAttention(config)
		self.attention_norm = RMSLayerNorm(config.model_dimension, epsilon=config.normalization_epsilon)
		self.feed_forward = FeedForwardNetwork(
			config.model_dimension,
			config.model_dimension * config.swiglu_multiple,
			256)
		self.feed_forward_norm = RMSLayerNorm(config.model_dimension, epsilon=config.normalization_epsilon)

	def forward(self, inputs, start_position, rotary_frequencies, attention_mask: Optional[torch.Tensor] = None):
		# Apply multi-head attention and layer normalization
		attention_output = self.attention.forward(inputs, start_position, rotary_frequencies, attention_mask)
		attention_output = self.attention_norm.forward(inputs + attention_output)
		# Apply feed-forward network and layer normalization
		feed_forward_output = self.feed_forward.forward(attention_output)
		return self.feed_forward_norm.forward(attention_output + feed_forward_output)


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
	def __init__(self, config: TransformerConfig):
		super().__init__()
		self.embedding = nn.Embedding(config.vocabulary_size, config.model_dimension)
		self.positional_embedding = compute_rotary_embedding_frequencies(
			config.model_dimension,
			config.max_sequence_length)
		self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
		self.final_norm = RMSLayerNorm(config.model_dimension, epsilon=config.normalization_epsilon)
		self.output_projection = nn.Linear(config.model_dimension, config.vocabulary_size, bias=False)

	def forward(self, inputs, start_position, attention_mask: Optional[torch.Tensor] = None):
		# Embed inputs and add rotary positional embeddings
		embedded_inputs = self.embedding(inputs)
		embedded_inputs += self.positional_embedding[:, :inputs.size(1)]
		# Process inputs through each Transformer layer
		for layer in self.layers:
			embedded_inputs = layer(embedded_inputs, start_position, self.positional_embedding, attention_mask)
		# Apply final normalization and project to output vocabulary
		return self.output_projection(self.final_norm.forward(embedded_inputs))
