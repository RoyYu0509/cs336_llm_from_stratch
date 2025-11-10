import torch
from jaxtyping import Float, Array
from torch import Tensor
from einops import rearrange, reduce, repeat, einsum


class RoPE:
    """
    A R^{2x2} Rotation Matrix
    """
    def __init__(self, 
                 theta: float, 
                 d_k: int,
                 device=None
        ):
        """
        Create the rotation matrix
        """
        self.d_k = d_k
        self.device = device
        self.theta = torch.tensor(theta, device=device, dtype = torch.float32)
    
    def getRotMat(self, i:int, k:int):
        theta_ik = i / (self.theta.clone()**((2*k)/self.d_k))
        theta_ik = theta_ik.to(self.device)
        _cos = torch.cos(theta_ik)
        _sin = torch.sin(theta_ik)
        
        rotMat_ik: Float[Tensor, "out_vec in_vec"]
        rotMat_ik = torch.tensor(
            [
             [_cos, -_sin],
             [_sin,  _cos]
            ],
            dtype=torch.float64
        )
        return rotMat_ik


class PosEncod:
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        
        Parameters:
            - theta: float      Constant value for computing frequency
            - d_k: int          Dimension of the token vectors.
            - max_seq_len: int  Maximum sequence length of our input tokens.
        """
        self.theta = theta
        self.d_k = d_k
        self.device = device

        # Buffer the rotation matrix for reuse
        self.Rs:Float[Tensor, "max_seq_l, half_d, 2out_vec, 2in_vec"]
        if d_k % 2 != 0:
            d = d_k+1   # Pad for odd d_k
        else:
            d = d_k
        self.Rs = torch.stack(
            [
                torch.stack(
                    [
                        RoPE(theta, d_k).getRotMat(pos, k) for k in range(0, d//2)
                    ],
                    dim=0
                ) for pos in range(max_seq_len)
            ], 
            dim = 0
        )
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an in_vecput tensor of shape (..., seq_len, d_k) and 
        return a tensor of the same shape.

        Parameters:
            - x: the input batch data
            - token_positions: (..., seq_len) specifying the token positions of x along the sequence dimension.
        """
        token_positions:Float[Tensor, "seq"]
        x, token_positions = x.to(self.device), token_positions.to(self.device)
        # Pad a dummy dimension if the d_k is odd
        d = self.d_k
        is_odd_d = (d % 2 == 1)
        if is_odd_d:
            x = torch.cat([x, torch.zeros(*x.shape[:-1], 1, device=self.device, dtype=x.dtype)], dim=-1)
            d += 1
        
        # Get the Rotation Matrix
        Rs: Float[Tensor, f"seq, half_d, in_vec, out_vec"]
        position = token_positions[:x.shape[-2]] # Select the same len as the input sequence.
        Rs = self.Rs[position].to(x.dtype) # Retrieve only the relavent position
        
        # Unpack x into several 2-dim vec
        reshape_x = rearrange(x, "... seq (half_d in_vec) -> ... seq half_d in_vec", half_d = d//2, in_vec = 2)
        print(f"Shape of Rs: {Rs.shape}")
        print(f"Shape of reshape_x: {reshape_x.shape}")
        Rx = einsum(Rs, reshape_x, "... seq half_d out_vec in_vec, ... seq half_d in_vec -> ... seq half_d out_vec")
        reshape_Rx = rearrange(Rx, "... seq half_d out_vec -> ... seq (half_d out_vec)")
        # print(reshape_Rx.shape, x.shape)
        if is_odd_d:
            return reshape_Rx[..., :self.d_k]
        else:
            return reshape_Rx
    

        
    