from src.bpe_tokenizer.bpe_tokenizer import BBPE
import json
from typing import Iterable, Iterator

class Tokenizer(BBPE):
    def __init__(self, vocab, merges, special_tokens=None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, 
        and (optionally) a list of special tokens.

        Paramters:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        """
        super().__init__(max_vocab_size=len(vocab.keys()), special_tokens=special_tokens)
        # Overwrite
        self.id_2_bytes = vocab
        self.byte_2_id_size = len(list(vocab.keys()))
        self.byte_2_id = {v: k for k, v in self.id_2_bytes.items()}
        self.merge_sequence = merges
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized 
        vocabulary and list of merges (in the same format that your BPE training 
        code output) and (optionally) a list of special tokens.
        
        Paramters:
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        """
        # Read-In vocabulary
        with open(vocab_filepath, "r") as f:
            vocab = dict(json.load(f))

        # Read-In merge list
        with open(merges_filepath, "r") as f:
            merges = list(f.read())

        # Return a class object
        return cls(vocab, merges, special_tokens)
            

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        return self.encoding(text)
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs.
        """
        for string in iterable:
            for byte_id in self.encode(string):
                yield byte_id

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        return self.decoding(ids)
