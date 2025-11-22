# Build the NumPy data from the raw text
```
uv run python src/build_dataset.py \
    --train-text data/train.txt \
    --val-text data/valid.txt \
    --vocab-path src/bpe_tokenizer/vocab_id2b_dict.pkl \
    --merges-path src/bpe_tokenizer/merges_seq.pkl \
    --train-size 10000 \
    --valid-size 2000 \
    --out-dir data/tokenized
```

# Train the LM using the NumPy Data
```
uv run python src/trainer.py \
    --TRAIN_PATH data/tokenized/train_tokens.npy \
    --VAL_PATH data/tokenized/val_tokens.npy \
    --VOCAB_PATH src/bpe_tokenizer/vocab_id2b_dict.pkl \
    --MERGES_PATH src/bpe_tokenizer/merges_seq.pkl \
    --BATCH_SIZE 32 \
    --CONTEXT_LENGTH 256 \
    --EPOCHES 1000 \
    --WANDB_PROJECT "MyLLMProject" \
    --DEVICE "mps" \
    --EVAL_INTERVAL 100 \
    --SAVE_INTERVAL 500 \
```