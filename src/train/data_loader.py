from src.transfromer.embedding import Embedding
from src.transfromer.linear_module import Linear
from src.transfromer.multiheads_attention import MultiHeadsAttention
from src.transfromer.pointwise_ffn import PointwiseSGLUactFFN
from src.transfromer.positionalNencoding import PosEncod
from src.transfromer.rmsnorm import Rmsnorm
from src.transfromer.transformer import PreNormTransformer
from src.transfromer.scaled_dot_prod_attention import softmax
from einops import reduce
from jaxtyping import Float, Array, Int
import numpy as np
import torch.nn as nn
import torch
from torch import logsumexp

def data_loading(x, batch_size, context_length, device="cpu"):
    """
    Sample `batch_size` number of data from the input sequence x.
    """
    x: Int[np.array, f"Seq"]
    seq_len = x.shape[0]
    sampled_input = []
    sampled_output = []

    if context_length > seq_len:
        raise("Insufficient raw text length for the given context_length.")

    # Sample the starting index from 0 to (seq_len - batch_size - 1)
    start_idx = torch.randperm(seq_len - context_length)[:batch_size]

    for start in start_idx:
        end = start + context_length
        inputs = x[start:end]
        target = x[start+1:end+1] # Build the target as shift right by one place.
        sampled_input.append(list(inputs))
        sampled_output.append(list(target))
    
    input_seq = torch.tensor(sampled_input, device=device)
    target_seq = torch.tensor(sampled_output, device=device)
    return input_seq, target_seq