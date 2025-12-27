import regex as re
from typing import Dict, Tuple

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        """

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle),
         return a generator that lazily yields token IDs. 
         This is required for memory-eﬀicient tokenization of large files that
          we cannot directly load into memory."""


    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
