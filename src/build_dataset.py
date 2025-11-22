"""
Utility script to convert raw text files into NumPy arrays of token IDs.

Example:
    uv run python src/train/build_dataset.py \
        --train-text data/TinyStoriesV2-GPT4-train.txt \
        --val-text data/TinyStoriesV2-GPT4-valid.txt \
        --vocab-path src/bpe_tokenizer/vocab_id2b_dict.pkl \
        --merges-path src/bpe_tokenizer/merges_seq.pkl \
        --out-dir data/tokenized
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.bpe_tokenizer.tokenizer import Tokenizer


def _encode_file(tokenizer: Tokenizer, text_path: Path, eos_token: str, max_size: int) -> np.ndarray:
    def text_gen():
        eos_count = 0
        line_count = 0
        with text_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_count += 1
                yield line

                # Count EOS lines
                if line.strip() == eos_token:
                    eos_count += 1
                    # Stop reading more lines if reached max_size
                    if max_size is not None and eos_count >= max_size:
                        break

                # Print progress every 10 lines
                if line_count % 100 == 0:
                    print(f"Processed {line_count} lines...")

    token_stream = tokenizer.encode_iterable(text_gen())

    return np.fromiter(token_stream, dtype=np.int32)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize text files into .npy buffers.")
    parser.add_argument("--train-size", type=int, required=True, help="Rows of training data")
    parser.add_argument("--valid-size", type=int, required=True, help="Rows of validation data")
    parser.add_argument("--train-text", type=Path, required=True, help="Path to training text file.")
    parser.add_argument("--val-text", type=Path, required=True, help="Path to validation text file.")
    parser.add_argument("--vocab-path", type=Path, required=True, help="Tokenizer vocab pickle/json.")
    parser.add_argument("--merges-path", type=Path, required=True, help="Tokenizer merges pickle/json.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write train/val .npy outputs.")
    return parser.parse_args()


def _load_tokenizer(vocab_path: Path, merges_path: Path, eos_token: str = "<|endoftext|>") -> Tokenizer:
    """Load a tokenizer directly from its serialized files."""
    return Tokenizer.from_files(
        str(vocab_path),
        str(merges_path),
        special_tokens=[eos_token],
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(args.vocab_path, args.merges_path)

    print("Encoding training data...")
    train_tokens = _encode_file(tokenizer, args.train_text, "<|endoftext|>", args.train_size)
    print("Encoding validation data...")
    val_tokens = _encode_file(tokenizer, args.val_text, "<|endoftext|>", args.valid_size)

    train_out = args.out_dir / "train_tokens.npy"
    val_out = args.out_dir / "val_tokens.npy"

    np.save(train_out, train_tokens)
    np.save(val_out, val_tokens)

    print(f"Wrote {train_out} ({train_tokens.shape[0]} tokens)")
    print(f"Wrote {val_out} ({val_tokens.shape[0]} tokens)")


if __name__ == "__main__":
    main()
