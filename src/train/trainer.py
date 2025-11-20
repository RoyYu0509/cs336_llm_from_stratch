import argparse
import wandb
import os
import torch
DTYPE_DICT={
    "float32": torch.float32,
    "float16": torch.float16
}

parser = argparse.ArgumentParser(description="Training LLM")

# Logging
parser.add_argument("--WANDB_PROJECT", type=str, default=None, help="Weights & Biases project (optional).")
parser.add_argument("--WANDB_RUN_NAME", type=str, default=None, help="Weights & Biases run name.")
# Data / experiment setup.
parser.add_argument("--TRAIN_PATH", type=str, required=True, help="Path to tokenized training data file.")
parser.add_argument("--VAL_PATH", type=str, required=True, help="Path to tokenized validation data file.")
parser.add_argument("--VOCAB_PATH", type=str, required=True, help="Pickled tokenizer vocab data file.")
parser.add_argument("--MERGES_PATH", type=str, required=True, help="Pickled tokenizer merges data file.")
parser.add_argument("--BATCH_SIZE", type=int, default=32, help="Sequences per optimization step.")
parser.add_argument("--CONTEXT_LENGTH", type=int, default=256, help="Tokens per training sequence.")
parser.add_argument("--EPOCHES", type=int, default=500, help="Number of training epoches.")

# Model hyperparameters.
parser.add_argument("--VOCAB_SIZE", type=int, default=10_000, help="Tokenizer vocabulary size.")
parser.add_argument("--NUM_LAYERS", type=int, default=4, help="Transformer block count.")
parser.add_argument("--D_MODEL", type=int, default=512, help="Transformer embedding dimension.")
parser.add_argument("--NUM_HEADS", type=int, default=16, help="Attention head count.")
parser.add_argument("--D_FF", type=int, default=1_344, help="Point-wise FFN hidden size.")
parser.add_argument("--ROPE_THETA", type=float, default=10_000.0, help="RoPE theta parameter.")

# Optimization settings.
parser.add_argument("--LR", type=float, default=3e-4, help="AdamW learning rate.")
parser.add_argument("--WEIGHT_DECAY", type=float, default=0.01, help="AdamW weight decay.")
parser.add_argument("--BETA1", type=float, default=0.9, help="AdamW beta1.")
parser.add_argument("--BETA2", type=float, default=0.999, help="AdamW beta2.")
parser.add_argument("--ADAM_EPS", type=float, default=1e-8, help="AdamW epsilon.")
parser.add_argument("--GRAD_CLIP", type=float, default=1.0, help="Global gradient norm clip value.")
parser.add_argument("--MAX_ITERS", type=int, default=50_000, help="Number of optimizer steps.")
parser.add_argument("--WARMUP_ITERS", type=int, default=2_000, help="Linear warmup steps.")

# Device.
parser.add_argument("--DEVICE", type=str, default="cpu", help="Torch device string, e.g., 'cuda', 'cpu', 'mps'.")
parser.add_argument("--DTYPE", type=str, default="float32", help="Torch dtype string, e.g., 'float32', 'bfloat16'.")

# Checkpointing
parser.add_argument("--CHECKPOINT_DIR", type=str, default="checkpoints", help="Where to store checkpoints.")
parser.add_argument("--RESUME_FROM", type=str, default=None, help="Checkpoint file to resume from.")
parser.add_argument("--LOG_INTERVAL", type=int, default=50, help="Steps between training log prints.")
parser.add_argument("--EVAL_INTERVAL", type=int, default=500, help="Steps between validation runs.")
parser.add_argument("--SAVE_INTERVAL", type=int, default=1_000, help="Steps between checkpoint saves.")
parser.add_argument("--SEED", type=int, default=0, help="Random seed.")



args = parser.parse_args()

TRAIN_PATH = args.TRAIN_PATH
VAL_PATH = args.VAL_PATH
VOCAB_PATH = args.VOCAB_PATH
MERGES_PATH = args.MERGES_PATH
BATCH_SIZE = args.BATCH_SIZE
EPOCHES = args.EPOCHES

CONTEXT_LENGTH = args.CONTEXT_LENGTH
VOCAB_SIZE = args.VOCAB_SIZE
NUM_LAYERS = args.NUM_LAYERS
D_MODEL = args.D_MODEL
NUM_HEADS = args.NUM_HEADS
D_FF = args.D_FF
ROPE_THETA = args.ROPE_THETA

LR = args.LR
WEIGHT_DECAY = args.WEIGHT_DECAY
BETAS = (args.BETA1, args.BETA2)
ADAM_EPS = args.ADAM_EPS
GRAD_CLIP = args.GRAD_CLIP
MAX_ITERS = args.MAX_ITERS
WARMUP_ITERS = args.WARMUP_ITERS

DEVICE = args.DEVICE
DTYPE = DTYPE_DICT[args.DTYPE]

CHECKPOINT_DIR = args.CHECKPOINT_DIR
RESUME_FROM = args.RESUME_FROM
LOG_INTERVAL = args.LOG_INTERVAL
EVAL_INTERVAL = args.EVAL_INTERVAL
SAVE_INTERVAL = args.SAVE_INTERVAL
SEED = args.SEED

WANDB_PROJECT = args.WANDB_PROJECT
WANDB_RUN_NAME = args.WANDB_RUN_NAME

print("==== Trainer arguments ====")
for name in sorted(vars(args)):
    print(f"{name}: {getattr(args, name)}")
print("===========================")

# Prepared the logging 
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
if not WANDB_PROJECT:
    raise RuntimeError("Provide a WanDB Project to log the results.")
run = wandb.init(project= WANDB_PROJECT, name=f"seed{SEED}-epoches{EPOCHES}")


import argparse
from src.lm import TransformerLM
from src.train.optimizer import AdamW
from src.train.checkpointing import load_checkpoint, save_checkpoint, save_checkpoint_and_log
from src.train.data_loader import data_loading
from src.train.loss import cross_entropy, perplexity
from src.bpe_tokenizer.tokenizer import Tokenizer
import numpy as np

# Initialize Modules
lm_model = TransformerLM(VOCAB_SIZE, CONTEXT_LENGTH, NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, ROPE_THETA,
                         device=DEVICE, dtype=DTYPE)
opt = AdamW(lm_model.parameters(), LR, WEIGHT_DECAY, BETAS)
toeknizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"])

# Collect Data
train_data = np.load(TRAIN_PATH, mmap_mode="r")
valid_data = np.load(VAL_PATH, mmap_mode="r")

# Pad a batch dimension for correct iterating
if train_data.ndim == 1:
    train_data = np.expand_dims(train_data, axis=0)
if valid_data.ndim == 1:
    valid_data = np.expand_dims(valid_data, axis=0)

# Training Loop
for iter in range(EPOCHES):
    for x in train_data:
        # Reset the gradients for all learnable parameters.
        opt.zero_grad() 
        inputs, targets = data_loading(x, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
        prediction = lm_model.forward(inputs)
        tr_loss = cross_entropy(prediction, targets)
        tr_loss.backward() # The returned loss is a Tensor, with a computational graph attached, so that we can bp
        opt.step() # After bp, all parameters' tensors have collect grad values
    
    if iter % EVAL_INTERVAL == 0:
        # Compute the Validation Loss (Perplexity)
        val_loss = 0
        for x in valid_data:
            inputs, targets = data_loading(x, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
            prediction = lm_model.forward(inputs)
            val_loss += perplexity(prediction, targets)
        print(f"Training Loss: {tr_loss} | Validation Loss: {val_loss}")

    if iter % SAVE_INTERVAL == 0:
        # Compute the Validation Loss (Perplexity)
        val_loss = 0
        for x in valid_data:
            inputs, targets = data_loading(x, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
            prediction = lm_model.forward(inputs)
            val_loss += perplexity(prediction, targets)
        print(f"Training Loss: {tr_loss} | Validation Loss: {val_loss}")

        # Log into WanDB
        local_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"iter_{iter}.pt")
        save_checkpoint_and_log(lm_model, opt, iter, out=local_checkpoint_path)
    
    
        

    
    

        

