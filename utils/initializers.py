from typing import Callable

from torch import Tensor

from torch.nn.init import (
    xavier_normal_,
    xavier_uniform_,
    kaiming_normal_,
    kaiming_uniform_,
    zeros_,
    ones_
)

_INITIALIZERS = {
    "glorot_normal": xavier_normal_,
    "glorot_uniform": xavier_uniform_,
    "he_normal": kaiming_normal_,
    "he_uniform": kaiming_uniform_,
    "zeros": zeros_,
    "ones": ones_
}


def _get_initializer(name: str | Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    if isinstance(name, str):
        try:
            initializer = _INITIALIZERS[name]
            return initializer
        except KeyError as e:
            raise ValueError(
                f"Valid initializers are {list(_INITIALIZERS.keys())}",
                f"Got {name} instead."
            )
    return name