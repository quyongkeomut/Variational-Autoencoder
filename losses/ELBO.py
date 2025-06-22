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

from torch.nn import (Module)
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits

class ELBOLoss(Module):
    def __init__(
        self, 
        prior_weight: float = 1e-4,
        reconstruction_method: str = "mse",
        beta: float = 1.0,
        epsilon: float = 1e-5,
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
            epsilon (float, optional): Smoothing value. Defaults to 1e-5.
        """
        super().__init__()
        self.prior_weight = float(prior_weight)
        assert reconstruction_method in ["mse", "bce"], (
            f"reconstruction_method must be one of ['mse', 'bce'], got {reconstruction_method} instead."
        )
        self.reconstruction_method = reconstruction_method
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        
        
    def _prior_matching(
        self,
        mean: Tensor,
        log_var: Tensor
    ) -> Tensor:
        """
        Calculate the Prior matching part of ELBO

        Args:
            mean (Tensor): Mean of latent space
            log_var (Tensor): Log of variance vector of latent space

        Shape:
            mean: (N, d)
            log_var: (N, d)
        
        Returns:
            Tensor: Prior matching component of ELBO as a scalar
        """
        # return 1./(2. * N) * (
        #     var.sum(-1) # (N,)
        #     - dim  # (d,)
        #     + mean.square().sum(-1) # (N,)
        #     - log(var.prod(-1) + self.epsilon) # (N,)
        # ).sum() / dim # for a proper ratio with the Reconstruction Loss
        
        # ----> 
        return -0.5 * (
            1
            + log_var # (N, d)
            - log_var.exp() # (N, d)
            - (mean ** 2) # (N, d)
        ).sum(-1).mean(0) 
    
    
    def _reconstruction(
        self, 
        decoder: Any,
        stats: Tuple[Tensor, Tensor],
        target: Tensor
    ) -> Tensor:
        """
        Calculate the Reconstruction part of ELBO.

        Args:
            decoder (Any): The Decoder
            stats (Tuple[Tensor, Tensor]): Ouputs of the Encoder, consists of Mean and Log 
                variance vectors
            target (Tensor): Input data 

        Shape:
            stats: [(N, d), (N, d)], where `d` is the dimensions of latent space
            target: (N, C, H, W)
        
        Returns:
            Tensor: A scalar value of Reconstruction Loss
        """
        # get the statistics from the Encoder
        mean, log_var = stats # (N, d)
        latent = self.reparameterize(mean, log_var) # (N, d)
        output: Tensor = decoder(latent) # (N, C, H, W)       
        if self.reconstruction_method == "bce": 
            return binary_cross_entropy_with_logits(output, target)
        elif self.reconstruction_method == "mse":
            return mse_loss(output, target)
    
    
    def forward(
        self, 
        decoder: Any,
        stats: Tuple[Tensor],
        X_input: Tensor,
        
    ) -> Tensor:
        """
        ELBO Loss Forward method

        Args:
            decoder (Any): The Decoder
            stats (Tensor): Statistic outputs from the Encoder
            X_input (Tensor): Input data

        Returns:
            Tensor: Scalar value of ELBO Loss
        """
        prior_loss = self._prior_matching(*stats)
        recon_loss = self._reconstruction(decoder, stats, X_input)
        return {
            "Loss": self.prior_weight*self.beta*prior_loss + recon_loss, 
            "Prior Loss": prior_loss, 
            "Recon Loss": recon_loss,
            "Mean": stats[0].detach().mean((0, 1)), # estimation of current output mean
            "Var": stats[1].detach().exp().mean((0, 1)) # estimation of current output mean
        }
        
    
    def reparameterize(
        self,
        mean: Tensor,
        log_var: Tensor
    ) -> Tensor:
        std = exp(0.5 * log_var)
        noise = randn_like(std)
        return std * noise + mean