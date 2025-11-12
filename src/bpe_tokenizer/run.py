from src.bpe_tokenizer.tokenizerBackEnd import BBPE
from src.bpe_tokenizer.tokenizer import Tokenizer
import pickle
import random


def train_tokenizer(input_path, vocab_size, special_tokens,
              num_processes = 3, split_special_token = b"<|endoftext|>"
            ):
    bpe_tk = Tokenizer(vocab_size=vocab_size, special_tokens=special_tokens)
    vocab, merges = bpe_tk.train(input_path, vocab_size,
                                 num_processes, split_special_token)
    return vocab, merges


def train_encoder():
    ECHO = True
    special_tokens = ['<|endoftext|>', ]
    max_vocab_size = 10000
    fp = "/Users/yifanyu/Desktop/CS336 LLM/CS336 A1/data/TinyStoriesV2-GPT4-train.txt"
    num_processes = 10
    sp_tok = "<|endoftext|>".encode("utf-8")

    bpe_tk = Tokenizer(vocab=max_vocab_size, merges=None, special_tokens=special_tokens)
    vocab_id2b_dict, merges_seq = bpe_tk.train(fp, max_vocab_size,
                                 num_processes, sp_tok)

    # Save vocab and merges
    with open("vocab_id2b_dict.pkl", "wb") as f:
        pickle.dump(vocab_id2b_dict, f)

    with open("merges_seq.pkl", "wb") as f:
        pickle.dump(merges_seq, f)


def small_vocab_size_encode_decode_sample():
    with open("vocab_id2b_dict.pkl", "rb") as f:
        vocab_id2b_dict = pickle.load(f)

    with open("merges_seq.pkl", "rb") as f:
        merges_seq = pickle.load(f)

    # Create tokenizer
    special_tokens = ['<|endoftext|>', ]
    bpe_tk = Tokenizer(vocab_id2b_dict, merges_seq, special_tokens)

    # Sample a few lines from text corp

    def sample_lines(filename, k=5):
        samples = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if i <= k:
                    samples.append(line)
                else:
                    j = random.randint(1, i)
                    if j <= k:
                        samples[j - 1] = line
        return [s.strip() for s in samples]

    samples = sample_lines("/Users/yifanyu/Desktop/CS336 LLM/CS336 A1/data/TinyStoriesV2-GPT4-valid.txt", k=5)
    for i, s in enumerate(samples, 1):
        print(f"--- Raw Sample {i} ---\n{s}\n")

        encoded = bpe_tk.encode(s)
        decoded = bpe_tk.decode(encoded)

        # --- Compute statistics ---
        n_bytes = len(s.encode("utf-8"))  # raw text bytes
        n_tokens = len(encoded)  # number of BPE tokens
        ratio = n_bytes / n_tokens if n_tokens > 0 else float("inf")

        print(f"Encoded IDs: {encoded}")
        print(f"Decoded txt: {decoded}\n")
        print(f"Bytes: {n_bytes}, Tokens: {n_tokens}, Bytes/Token: {ratio:.2f}\n")


def large_vocab_size_encode_decode_sample():
    with open("vocab_id2b_dict.pkl", "rb") as f:
        vocab_id2b_dict = pickle.load(f)

    with open("merges_seq.pkl", "rb") as f:
        merges_seq = pickle.load(f)

    # Create tokenizer
    special_tokens = ['<|endoftext|>', ]
    bpe_tk = Tokenizer(vocab_id2b_dict, merges_seq, special_tokens)

    # Sample a few lines from text corp

    def sample_lines(filename, k=5):
        samples = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if i <= k:
                    samples.append(line)
                else:
                    j = random.randint(1, i)
                    if j <= k:
                        samples[j - 1] = line
        return [s.strip() for s in samples]

    samples = sample_lines("/Users/yifanyu/Desktop/CS336 LLM/CS336 A1/data/TinyStoriesV2-GPT4-valid.txt", k=5)
    for i, s in enumerate(samples, 1):
        print(f"--- Raw Sample {i} ---\n{s}\n")

        encoded = bpe_tk.encode(s)
        decoded = bpe_tk.decode(encoded)

        # --- Compute statistics ---
        n_bytes = len(s.encode("utf-8"))  # raw text bytes
        n_tokens = len(encoded)  # number of BPE tokens
        ratio = n_bytes / n_tokens if n_tokens > 0 else float("inf")

        print(f"Encoded IDs: {encoded}")
        print(f"Decoded txt: {decoded}\n")
        print(f"Bytes: {n_bytes}, Tokens: {n_tokens}, Bytes/Token: {ratio:.2f}\n")


if __name__ == "__main__":
    # train_encoder()
    encode_decode_sample()