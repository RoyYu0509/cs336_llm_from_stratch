# Build the NumPy data from the raw text (train + valid)
```
uv run python src/build_dataset.py \
    --text-path data/TinyStoriesV2-GPT4-train.txt \
    --vocab-path src/bpe_tokenizer/vocab_id2b_dict.pkl \
    --merges-path src/bpe_tokenizer/merges_seq.pkl \
    --out data/tokenized/train_tokens.npy

uv run python src/build_dataset.py \
    --text-path data/TinyStoriesV2-GPT4-valid.txt \
    --vocab-path src/bpe_tokenizer/vocab_id2b_dict.pkl \
    --merges-path src/bpe_tokenizer/merges_seq.pkl \
    --out data/tokenized/val_tokens.npy
```

# Train the LM using the NumPy Data
```
uv run python src/trainer.py \
    --TRAIN_PATH data/tokenized/train_tokens.npy \
    --VAL_PATH data/tokenized/val_tokens.npy \
    --VOCAB_PATH src/bpe_tokenizer/vocab_id2b_dict.pkl \
    --MERGES_PATH src/bpe_tokenizer/merges_seq.pkl \
    --TR_BAT_SIZE 32 \
    --VAL_BAT_SIZE 32 \
    --CONTEXT_LENGTH 256 \
    --EPOCHES 1000 \
    --WANDB_PROJECT "MyLLMProject" \
    --DEVICE "mps" \
    --EVAL_INTERVAL 100 \
    --SAVE_INTERVAL 500
```
