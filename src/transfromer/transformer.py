impot jaxtyping

from src.transfromer.embedding import Embedding
from src.transfromer.linear_module import Linear
from src.transfromer.multiheads_attention import MultiHeadsAttention
from src.transfromer.pointwise_ffn import PointwiseSGLUactFFN
from src.transfromer.positionalNencoding import PosEncod
from src.transfromer.rmsnorm import Rmsnorm
import torch.nn as nn
import torch


class PreNormTransformer(nn.Module):
    def __init__(self,
                 d_model: int, heads_num: int, # Multi-Head Attention args
                 pos_encod=None, token_positions=None, # Multi-Head Attention kwargs
                 eps: float = 1e-5, # rmsnorm kwargs
                 dim_ff=None, latent_exp_factor = 8/3, # FNN kwargs
                 device=None, dtype=torch.float16  # general kwargs
                 ):
        super().__init__()
        self.d_model = d_model
        self.heads_num = heads_num
        self.RMSN = Rmsnorm(d_model, eps, device, dtype)
        self.MHA = MultiHeadsAttention(d_model, heads_num, pos_encod, token_positions, device, dtype)
        self.FNN = PointwiseSGLUactFFN(d_model, dim_ff, latent_exp_factor, device, dtype)

    def forward(self,x:torch.Tensor):
        """
        Return the output from a transformer block.

        Parameter:
            x: (batch, seq, d_model)
        """
        

        
