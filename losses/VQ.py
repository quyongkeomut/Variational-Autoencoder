from typing import (
    Any,
    Tuple
)

import torch
from torch import (
    Tensor, 
    exp, 
    randn_like
)

from torch.nn import Module, Embedding
from torch.nn.functional import mse_loss


class VQ_VAELoss(Module):
    def __init__(
        self, 
        commitment_weight: float = 1.0,
        vqloss_weight: float = 1.0,
        *args, 
        **kwargs
    ) -> None:
        """
        Implementation of ELBO (Evidence Lower-Bound) and it variant beta-VAE (β-VAE)
        
        Args:
            prior_weight (float, optional): Weight of Prior Matching loss. 
                Defaults to 1.
            beta (float, optional): Scaling scalar for the Prior matching part of the 
                total loss function. Defaults to 1.0.
        """
        super().__init__()
        self.commitment_weight = commitment_weight
        self.vqloss_weight = vqloss_weight
    
    def _VQ_loss(
        self,
        decoder: Any,
        latents: Tensor,
    ) -> Tensor:
        """
        Calculate the VQ Loss part of VQ-VAE

        Args:
            decoder (Any): The Decoder.
            latents (Tensor): Output from Encoder.
            
        Shape:
            latent: (N, C, H, W)

        Returns:
            Tensor: VQ Loss component of ELBO as a scalar
        """
        # ----> 
        def _commitment_loss(
            latents: Tensor,
            quantized_latents: Tensor
        ):
            return mse_loss(latents, quantized_latents.detach())
            
        def _codebook_loss(
            latents: Tensor,
            quantized_latents: Tensor
        ):
            return mse_loss(quantized_latents, latents.detach())
            
        quantized_latents = self._quantize(decoder.codebook, latents)
        return (
            _codebook_loss(latents, quantized_latents) 
            + self.commitment_weight*_commitment_loss(latents, quantized_latents)
        ), quantized_latents
        
    def _quantize(
        self, 
        codebook: Embedding, 
        latents: Tensor,
    ) -> Tensor:
        """
        _summary_

        Args:
            codebook (Embedding): The codebook.
            latents (Tensor): Output from the Encoder.

        Shape:
            latents: (N, d, H, W)
        
        Returns:
            Tensor: Quantized latents that looked-up from the codebook.
        """
        distances: Tensor = torch.linalg.vector_norm(
            latents.movedim(1, -1).unsqueeze(-2) # (N, H, W, 1, d)
            - codebook.weight, # (K, d)
            dim=-1
        ) # (N, H, W, K)
        
        # get the closest centroids
        encoding_indices = distances.argmin(-1) # (N, H, W)
        Q_Z = codebook(encoding_indices) # (N, H, W, d)
        Q_Z = Q_Z.movedim(-1, 1) # (N, d, H, W)
        return Q_Z
    
    def _reconstruction(
        self, 
        decoder: Any,
        quantized_latents: Tensor,
        target: Tensor
    ) -> Tensor:
        """
        Calculate the Reconstruction part of VQ-VAE.

        Args:
            decoder (Any): The Decoder
            quantized_latents: Quantized version of latents from the Encoder.
            target (Tensor): Input data 

        Shape:
            quantized_latents: (N, C, H, W), where `C` is the dimensions of latent space
            target: (N, C, H, W), where `C` is the dimensions of image space
        
        Returns:
            Tensor: A scalar value of Reconstruction Loss
        """
        # get the statistics from the Encoder
        output: Tensor = decoder(quantized_latents) # (N, C, H, W)        
        return mse_loss(output, target)
    
    def forward(
        self, 
        decoder: Any,
        latents: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        VA-VAE Loss Forward method

        Args:
            decoder (Any): The Decoder
            latents (Tensor): Output from the Encoder
            target (Tensor): Input data

        Returns:
            Tensor: Scalar value of VQ-VAE Loss
        """
        VQ_loss, quantized_latents = self._VQ_loss(decoder, latents)
        # copy gradient, as if the latents if the quantized version rather than the original
        quantized_latents = latents + (quantized_latents - latents).detach() 
        reconstruction_loss = self._reconstruction(decoder, quantized_latents, target)
        return {
            "Loss": self.vqloss_weight*VQ_loss + reconstruction_loss, 
            "VQ Loss": VQ_loss, 
            "Reconstruction": reconstruction_loss
        }