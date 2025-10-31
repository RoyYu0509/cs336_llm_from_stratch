import torch
import torch.nn as nn

from jaxtyping import Float, Int

def trunct_normal_para_init(dim_in, dim_out, mu=0, var=1, truc_magnitude=3, device="cpu", dtype=torch.float16):
    """
    Return the initialized parameters value using truncated normal.
    """
    tensor = torch.empty(dim_in, dim_out, dtype = dtype, device = device)
    weight_mat: Float[torch.Tensor, "dim_in, dim_out"] = tensor 
    nn.init.trunc_nrmal_(
        weight_mat, mean=mu, std=torch.sqrt(var), 
        a=mu-truc_magnitude, b=mu+truc_magnitude,
    )

    return weight_mat
    