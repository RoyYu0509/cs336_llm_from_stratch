from src.bpe_tokenizer.bpe_tokenizer import BBPE

def main():
    ECHO = True
    special_tokens = ['<|endoftext|>',]
    max_vocab_size = 270
    fp = "/Users/yifanyu/Desktop/CS336 LLM/CS336 A1/data/toy_text.txt"
    num_processes = 1
    sp_tok = "<|endoftext|>".encode("utf-8")

    bpe_tk = BBPE(max_vocab_size, special_tokens)
    vocab, merges = bpe_tk.train(fp, max_vocab_size, 
                                 num_processes, sp_tok)
    
    
    # print(bpe_tk.merge_sequence)

    encoded_text = bpe_tk.encoding(" low <|endoftext|> <|endoftext|> key <|endoftext|>")

    print(encoded_text)

    for bt in encoded_text:
        print(bpe_tk.id_2_bytes[bt])

    
def train_bpe(input_path, vocab_size, special_tokens, 
              num_processes = 3, split_special_token = b"<|endoftext|>"
            ):
    bpe_tk = BBPE(vocab_size, special_tokens)
    vocab, merges = bpe_tk.train(input_path, vocab_size, 
                                 num_processes, split_special_token)
    return vocab, merges

    


if __name__ == "__main__":
    main()