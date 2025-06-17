from typing import Tuple
    
import torch
from torch import randn, Tensor
from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
    ModuleList,
    Embedding
)

from neural_nets.activations import get_activation
from neural_nets.autoencoders.base_ae import Encoder, Decoder, LATENT_H, LATENT_W

from utils.initializers import ones_
from utils.other_utils import REVERSE_TRANSFORMS
    
    
class VQ_VAEEncoder(Encoder):
    ...
    
    
class VQ_VAEDecoder(Decoder):
    def __init__(
        self, 
        codebook_size: int,
        codebook_dim: int,
        retriever_depth: int = 3,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        factory_kwargs = {"device": kwargs["device"], "dtype": kwargs["dtype"]}
        self.codebook = Embedding(
            num_embeddings=codebook_size,
            embedding_dim=codebook_dim
        ) # Parameter(randn(codebook_size, codebook_dim, **factory_kwargs))
    
        dim_feedforward=2*codebook_dim
        self.retriever = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=codebook_dim,
                    nhead=kwargs["nhead"],
                    dim_feedforward=dim_feedforward,
                    batch_first=True,
                    **factory_kwargs,
                )
                for _ in range(retriever_depth)
            ]
        )

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
        Z = input.view(-1, self.latent_channels, LATENT_H, LATENT_W) # (N, C, H, W)
        return self.layers(Z)
    
    def sample(
        self,
        num_samples: int,
    ):
        """
        Samples from the latent space and return the synthetic image

        Args:
            num_samples (int): Number of samples
        """
        Z = torch.randn(num_samples, self.latent_channels, LATENT_H, LATENT_W, **self.factory_kwargs)
        return self.layers(Z).detach()
    
    def generate(self):
        return REVERSE_TRANSFORMS(self.sample(1)[0].cpu())