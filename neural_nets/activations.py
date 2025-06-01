import torch
from torch.nn import (
    Parameter,
    Module,
    Sigmoid,
    Tanh,
    Hardsigmoid,
    ReLU,
    ReLU6, 
    SELU,
    GELU,
)
import torch.nn.functional as F

class Swish(Module):
    def __init__(
        self,
        device=None,
        dtype=None
    ):
        r"""
        Implement of Swish activation with learnable parameter

        ``Swish(x) = x * Sigmoid(beta*x)``
        
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.beta = Parameter(torch.ones((), **factory_kwargs))
        
    def forward(self, input):
        return input * F.sigmoid(self.beta*input)
    
    
class HardSwish(Module):
    def __init__(
        self,
        device=None,
        dtype=None
    ):
        r"""
        Implement of HardSwish activation with learnable parameter

        ``HardSwish(x) = x * HardSigmoid(beta*x)``
        
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.beta = Parameter(torch.ones((), **factory_kwargs))
        
    def forward(self, input):
        return input * F.hardsigmoid(self.beta*input)
    
    
ACTIVATIONS = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "hardsigmoid": Hardsigmoid,
    "relu": ReLU,
    "relu6": ReLU6,
    "selu": SELU,
    "gelu": GELU,
    "swish": Swish,
    "hardswish": HardSwish
}


def get_activation(name: str):
    try:
        return ACTIVATIONS[name]()
    except KeyError:
        raise KeyError(
            f"Activation must be one of these {list(ACTIVATIONS.keys)}, got {name} instead"
        )