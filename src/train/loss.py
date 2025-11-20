from src.transfromer.embedding import Embedding
from src.transfromer.linear_module import Linear
from src.transfromer.multiheads_attention import MultiHeadsAttention
from src.transfromer.pointwise_ffn import PointwiseSGLUactFFN
from src.transfromer.positionalNencoding import PosEncod
from src.transfromer.rmsnorm import Rmsnorm
from src.transfromer.transformer import PreNormTransformer
from src.transfromer.scaled_dot_prod_attention import softmax
from einops import reduce
from jaxtyping import Float, Array
import torch.nn as nn
import torch
from torch import logsumexp



def cross_entropy(out_logits, reference):
    """
    Compute the cross-entropy (negative log-likelihood) on the LM's output

    Inputs:
        - out_logits: [Batch, ..., Seq_len] The output logits for each position of the sentence.
        - reference: [Batch, ..., Seq_len, Vocab_dist] The reference tokenized sentence.
    
    Output:
        - nll: A scalar loss value
    """
    d = reference.shape[0]
    seq_l = reference.shape[-1]
    
    # Subtract max for numerical stability
    max_val, _ = torch.max(out_logits, dim=-1, keepdim=True) # Keep dim for Broadcasting
    out_logits = out_logits - max_val

    # Build the words retrieveing mask
    out_logits: Float[torch.Tensor, f"Batch, ..., Seq_len, Vocab_dist"]
    ref_words: Float[torch.Tensor, f"Batch, ..., Seq_len, 1"] = reference.unsqueeze(-1)

    # Cancel out log & exp to change the computation structure.
    # o[xi+1] - log(exp(sum(o)))
    nll = -(out_logits.gather(-1, ref_words) - logsumexp(out_logits, dim=-1))
    loss = 1/(d*seq_l) * nll.sum()
    return loss


def perplexity(logits, targets):
    loss = cross_entropy(logits, targets)
    return torch.exp(loss)
