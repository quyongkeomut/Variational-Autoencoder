import torch
import torch.nn as nn
from torch.nn import (
    Parameter,
    Module,
)
import torch.nn.functional as F


class Sigmoid(nn.Sigmoid):
    def __init__(self, *args, **kwargs):
        super().__init__()
        

class Hardsigmoid(nn.Hardsigmoid):
    def __init__(self, inplace = False, *args, **kwargs):
        super().__init__(inplace)


class Tanh(nn.Tanh):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        
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
        
        
class ReLU(nn.ReLU):
    def __init__(self, inplace = False, *args, **kwargs):
        super().__init__(inplace)
        

class ReLU6(nn.ReLU6):
    def __init__(self, inplace = False, *args, **kwargs):
        super().__init__(inplace)
        
    
class LeakyReLU(nn.LeakyReLU):
    def __init__(
        self, 
        negative_slope = 0.05, 
        inplace = False, 
        *args, 
        **kwargs
    ):
        super().__init__(negative_slope, inplace)


class SELU(nn.SELU):
    def __init__(self, inplace = False, *args, **kwargs):
        super().__init__(inplace)
        
        
class GELU(nn.GELU):
    def __init__(self, approximate = "none", *args, **kwargs):
        super().__init__(approximate)
        
        
class SiLU(nn.SiLU):
    def __init__(self, inplace = False, *args, **kwargs):
        super().__init__(inplace)
    
    
ACTIVATIONS = {
    "sigmoid": Sigmoid,
    "hardsigmoid": Hardsigmoid,
    "tanh": Tanh,
    "swish": Swish,
    "hardswish": HardSwish,
    "relu": ReLU,
    "relu6": ReLU6,
    "leaky_relu": LeakyReLU,
    "selu": SELU,
    "gelu": GELU,
    "silu": SiLU,
}


def get_activation(
    name: str,
    *args,
    **kwargs
):
    try:
        return ACTIVATIONS[name](*args, **kwargs)
    except KeyError:
        raise KeyError(
            f"Activation must be one of these {list(ACTIVATIONS.keys)}, got {name} instead"
        )