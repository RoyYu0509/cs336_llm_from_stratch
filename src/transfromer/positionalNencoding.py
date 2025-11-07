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
        self.theta = torch.tensor(theta, device=device)
    
    def getRotMat(self, i:int, k:int):
        theta_ik = i / (self.theta**((2*k)/self.d_k))
        rotMat_ik = torch.tensor(
            [
             [torch.cos(theta_ik), -torch.sin(theta_ik)],
             [torch.sin(theta_ik),  torch.cos(theta_ik)]
            ],
            dtype=torch.float32
        )
        return rotMat_ik


class PosEncod:
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        """
        self.theta = theta
        self.d_k = d_k
        self.rotMatGen = RoPE(theta, d_k)
        self.max_seq_len = max_seq_len
        self.device = device
        
    def _Ri(self, i, d):
        """
        Create the R^{dxd} Rotation Matrix for the R^{d} token vec
        """
        return torch.stack(
            [
             self.rotMatGen.getRotMat(i, k) for k in range(0, d//2)
            ]
        )

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and 
        return a tensor of the same shape.
        """
        x, token_positions = x.to(self.device), token_positions.to(self.device)
        # Pad a dummy dimension if the d_k is odd
        d = self.d_k
        id_odd_d = d % 2 == 1
        if id_odd_d:
            x = torch.cat([x, torch.zeros(*x.shape[:-1], 1, device=self.device, dtype=x.dtype)], dim=-1)
            d += 1

        Rs: Float[Tensor, f"seq, half_d, two_in, two_out"]
        Rs = torch.stack(
            [self._Ri(i, d) for i in range(self.max_seq_len)]
        )
        
        # Unpack x into several 2-dim vec
        reshape_x = rearrange(x, "... seq (half_d sub_vec) -> ... seq half_d sub_vec", half_d = d//2, sub_vec = 2)
        Rx = einsum(reshape_x, Rs, "... seq half_d sub_vec, seq half_d two_in two_out -> ... seq half_d two_out")
        reshape_Rx = rearrange(Rx, "... seq half_d two_out -> ... seq (half_d two_out)")
        # print(reshape_Rx.shape, x.shape)
        if id_odd_d:
            return reshape_Rx[..., :-1]
        else:
            return reshape_Rx
    

        
    