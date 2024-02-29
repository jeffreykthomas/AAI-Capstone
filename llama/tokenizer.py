from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List
import os

# Initialize the logger to output information during the process
logger = getLogger()


class Tokenizer:
	def __init__(self, model_path: str):
		"""
		Initializes the tokenizer with a pre-trained SentencePiece model.

		Args:
			model_path (str): The file path to the SentencePiece model.
		"""
		# Ensure the model file exists
		assert os.path.isfile(model_path), f"Model path {model_path} does not exist."

		# Load the SentencePiece model
		self.sp_model = SentencePieceProcessor(model_file=model_path)
		logger.info(f"Reloaded SentencePiece model from {model_path}")

		# Retrieve special token IDs: Beginning Of Sentence (BOS), End Of Sentence (EOS), and Padding (PAD)
		self.n_words = self.sp_model.vocab_size()  # Total vocabulary size
		self.bos_id = self.sp_model.bos_id()  # BOS token ID
		self.eos_id = self.sp_model.eos_id()  # EOS token ID
		self.pad_id = self.sp_model.pad_id()  # PAD token ID

		# Log the special token IDs and vocabulary size
		logger.info(f"#words: {self.n_words}, BOS ID: {self.bos_id}, EOS ID: {self.eos_id}")

		# Ensure that the reported vocabulary size matches the number of pieces in the model
		assert self.sp_model.vocab_size() == self.sp_model.get_piece_size(), (
			"Mismatch in reported vocabulary size and actual piece size.")

	def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
		"""
		Encodes a string into a list of token IDs, optionally adding BOS and/or EOS tokens.

		Args:
			s (str): The string to encode.
			bos (bool): If True, prepend the BOS token.
			eos (bool): If True, append the EOS token.

		Returns:
			List[int]: A list of token IDs representing the encoded string.
		"""
		# Validate input type
		assert isinstance(s, str), "Input must be a string."

		# Encode the string to a list of token IDs
		tokens = self.sp_model.encode(s)

		# Optionally add BOS and EOS tokens
		if bos:
			tokens = [self.bos_id] + tokens
		if eos:
			tokens += [self.eos_id]

		return tokens

	def decode(self, tokens: List[int]) -> str:
		"""
		Decodes a list of token IDs back into a string.

		Args:
			tokens (List[int]): The list of token IDs to decode.

		Returns:
			str: The decoded string.
		"""
		return self.sp_model.decode(tokens)
