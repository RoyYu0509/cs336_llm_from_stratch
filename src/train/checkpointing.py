import torch

def save_checkpoint(model, optimizer, iteration, out): 
    """
    Save the states of `model`, `optimizer` and `iteration` into
    one file. 

    Inputs:
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        iteration: int
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    """
    torch.save(
        {"model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": iteration},
        f = out
    )

def load_checkpoint(src, model, optimizer):
    """
    Load a checkpoint from src, then recover the model and optimizer 
    states from that checkpoint.

    Inputs:
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
    """
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["iter"]