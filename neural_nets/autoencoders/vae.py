from typing import Tuple, List, Any
    
import torch
from torch import Tensor, randn
from torch.nn import (
    Sequential,
    Linear,
)
from torch.nn.functional import sigmoid

from neural_nets.activations import get_activation
from neural_nets.autoencoders.base_ae import Encoder, Decoder

from utils.initializers import ones_, zeros_
from utils.other_utils import to_image


class VAEEncoder(Encoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # append the µ and log(σ2) layers
        post_last_dim = self.invert_residual_kwargs["expand_factor"]*self.last_dim
        self.mean = Sequential(
            Linear(
                in_features=self.last_dim,
                out_features=post_last_dim,
                **self.factory_kwargs
            ),
            get_activation(self.activation, **self.factory_kwargs),
            Linear(
                in_features=post_last_dim,
                out_features=self.latent_dim,
                **self.factory_kwargs
            ),
        )
        
        self.log_var = Sequential(
            Linear(
                in_features=self.last_dim,
                out_features=post_last_dim,
                **self.factory_kwargs
            ),
            get_activation(self.activation, **self.factory_kwargs),
            Linear(
                in_features=post_last_dim,
                out_features=self.latent_dim,
                **self.factory_kwargs
            ),
            get_activation("leaky_relu", **self.factory_kwargs)
        )
        self._reset_parameters()
    
    
    def _reset_parameters(self):
        super()._reset_parameters()
        self.initializer(self.mean[0].weight)
        self.initializer(self.mean[2].weight)
        self.initializer(self.log_var[0].weight)
        self.initializer(self.log_var[2].weight)
    
        ones_(self.mean[0].bias)
        ones_(self.mean[2].bias)
        ones_(self.log_var[0].bias)
        ones_(self.log_var[2].bias)
    

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        Z: Tensor = self.layers(input) # (N, C, H, W)
        Z = Z.mean((-1, -2)) # (N, C), global avg pool
        mean, log_var = self.mean(Z), self.log_var(Z)
        return mean, log_var
    
    
class VAEDecoder(Decoder):
    def __init__(
        self, 
        reconstruction_method: str = "mse",
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.reconstruction_method = reconstruction_method
        
        self.decoder_input = Sequential(
            Linear(self.latent_dim, self.latent_dim*2, **self.factory_kwargs),
            get_activation(self.activation, **self.factory_kwargs),
            Linear(self.latent_dim*2, self.latent_dim, **self.factory_kwargs),
        )
        self._reset_parameters()
        
        
    def _reset_parameters(self):
        super()._reset_parameters()
        self.initializer(self.decoder_input[0].weight)
        ones_(self.decoder_input[0].bias)
        self.initializer(self.decoder_input[2].weight)
        ones_(self.decoder_input[2].bias)
        
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Forward method of VAE Decoder

        Args:
            input (Tensor): Latent vector

        Shape:
            input: (N, d)
        
        Returns:
            Tensor: Reconstructed image
        """
        Z = self.decoder_input(input) # (N, d)
        Z = Z.view(-1, self.latent_dim, *self.latent_shape) # (N, C, H, W)
        return self.layers(Z)
    
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
    ) -> List[Any]:
        """
        Samples from the latent space and return the synthetic image

        Args:
            num_samples (int): Number of samples
        """
        latent_shape = (num_samples, self.latent_dim)
        Z = randn(*latent_shape, **self.factory_kwargs) # (N, d)
        
        outputs = self.forward(Z).detach().cpu()
        if self.reconstruction_method == "bce":
            outputs = sigmoid(outputs)
        return [to_image(output) for output in outputs]
    
    
    def generate(self) -> Any:
        return self.sample(1)[0]