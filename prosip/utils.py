import torch.nn.functional as F
import torch.nn as nn

def get_activation_fn(activation: str, return_module: bool = False):
    """Convert activation function name to the corresponding PyTorch implementation.
    
    Args:
        activation: String name of the activation function
        
    Returns:
        PyTorch activation function
        
    Raises:
        ValueError: If activation name is not recognized
    """
    activation = activation.lower()
    if activation == "relu":
        return F.relu if not return_module else nn.ReLU
    elif activation == "gelu":
        return F.gelu if not return_module else nn.GELU
    elif activation == "silu" or activation == "swish":
        return F.silu if not return_module else nn.SiLU
    elif activation == "tanh":
        return F.tanh if not return_module else nn.Tanh
    elif activation == "sigmoid":
        return F.sigmoid if not return_module else nn.Sigmoid
    else:
        raise ValueError(f"Unknown activation function: {activation}")
