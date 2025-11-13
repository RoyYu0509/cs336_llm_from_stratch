from numpy.conftest import dtype

from src.bpe_tokenizer.tokenizerBackEnd import BBPE
import json
from typing import Iterable, Iterator

class Tokenizer(BBPE):
    def __init__(self, vocab=None, merges=None, special_tokens=None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, 
        and (optionally) a list of special tokens.

        Paramters:
            vocab:  int | dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None

        For Special Tokens we keep it as one token: counter[(b"<|endoftext|>",)] = <count>
        """
        if isinstance(vocab, dict):
            # print("Load in tokenizer...")
            super().__init__(max_vocab_size=len(vocab.keys()), special_tokens=special_tokens)
            # Overwrite a trained tokenizer
            if vocab is not None:
                self.id_2_bytes = vocab
                self.byte_2_id_size = len(list(vocab.keys()))
                self.byte_2_id = {v: k for k, v in self.id_2_bytes.items()}
            if merges is not None:
                self.merge_sequence = merges

        elif isinstance(vocab, int):
            # print("Initiate raw tokenizer...")
            # Initiate untrained tokenizer
            super().__init__(max_vocab_size=vocab, special_tokens=special_tokens)
        else:
            # Invalid input
            raise ValueError("Invalid Input `vocab`, expected dict or int")


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        vocab = {int(k): bytes(v) if isinstance(v, list) else v.encode("latin-1")
                for k, v in vocab_json.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                left_str, right_str = line.split()
                left = left_str.encode("latin-1")
                right = right_str.encode("latin-1")
                merges.append((left, right))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

                

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
