from src.transfromer.embedding import Embedding
from src.transfromer.linear_module import Linear
from src.transfromer.multiheads_attention import MultiHeadsAttention
from src.transfromer.pointwise_ffn import PointwiseSGLUactFFN
from src.transfromer.positionalNencoding import PosEncod
from src.transfromer.rmsnorm import Rmsnorm
from src.transfromer.transformer import PreNormTransformer
from src.transfromer.scaled_dot_prod_attention import softmax
import torch.nn as nn
import torch


class TransformerLM(nn.Module):
    def __init__(self, 
                 vocab_size: int, context_length: int, num_layers: int, # LM args
                 d_model: int, heads_num: int, # Multi-Head Attention args
                 d_ff: int, # FNN args
                 theta: float, # Encoder para
                 pos_encoder=None, # Multi-Head Attention kwargs
                 eps: float = 1e-5, # rmsnorm kwargs
                 latent_exp_factor = 8/3, # FNN kwargs
                 device=None, dtype=torch.float32,  # general kwargs
                 ):
        """
        A transformer block.

        Parameters:
            vocab_size:     int | The size of the vocabulary, necessary for determining the 
                            dimensionality of the token embedding matrix.

            context_length: int | The maximum context length, necessary for determining 
                            the dimensionality of the position embedding matrix.

            num_layers:     int | The number of Transformer blocks to use.

        
        Component:
            
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        # Input Embedding Block
        self.in_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model,
                                   device=device, dtype=dtype)

        # Positional Encoder
        self.pos_encoder = pos_encoder if pos_encoder is not None else PosEncod(theta, d_model//heads_num, context_length, device=device)
        
        # Defining token position with maximum length 
        token_positions = torch.arange(0, context_length)
        self.register_buffer("token_positions", token_positions)
        
        # Transformer Blocks
        self.tf_layers = nn.ModuleList([
            PreNormTransformer(
                d_model, heads_num,
                d_ff,
                pos_encod=self.pos_encoder, token_positions=None,
                latent_exp_factor=latent_exp_factor,
                device = device, dtype=dtype
            ) for _ in range(num_layers)
        ])

        # Normalization Layer
        self.norm = Rmsnorm(d_model, eps, device, dtype)

        # Output Linear Block
        self.head = Linear(d_model, vocab_size, device=device, dtype=dtype)


    def forward(self, x):
        """
        Input: 
            - x: A batched sequence of integer token IDs, (batch_size, sequence_length)

        Return a raw logits distribution over the vocabulary
        with shape (batch_size, sequence_length, vocab_size).
        """
        # MatMul: (batch_size, sequence_length) . (vocab_size, embedding_dim) 
        x = self.in_embedding.forward(x)
        
        batch_size, seq_len = x.shape[0], x.shape[1]
        positions = self.token_positions[:seq_len].unsqueeze(0).expand(batch_size, -1)
        positions = positions.to(x.device)
        for tf_block in self.tf_layers:
            x = tf_block.forward(x, token_positions=positions)

        x = self.norm.forward(x)

        x = self.head.forward(x)
        
        # softmax(x, -1)  # Softmax muted, Return raw logit
        return x

        
        
