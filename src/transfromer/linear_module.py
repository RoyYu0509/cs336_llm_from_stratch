from para_init import trunct_normal_para_init
import torch
from jaxtyping import Float, Int
from einops import rearrange, einsum

class Linear(torch.nn.Moudule):
    def __init__(self, in_feats, out_feats, device=None, dtype=None, mu=0, var=1, truc_magnitude=3):
        """
        linear transformation module. This function should accept the following parameters:
        
        Parameters:
            in_features:    int final dimension of the input
            out_features:   int final dimension of the output
            device:         torch.device | None = None Device to store the parameters on
            dtype:          torch.dtype | None = None Data type of the parameters
        """
        super.__init__()
        self.device = device
        self.dtype = dtype
        weightMat = trunct_normal_para_init(in_feats, out_feats, mu, var, truc_magnitude, device, dtype)
        self.weightMat: Float[torch.Tensor, "dim_in, dim_out"] = weightMat
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        """
        return 


