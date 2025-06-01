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
from torch.nn.functional import binary_cross_entropy_with_logits

class ELBOLoss(Module):
    def __init__(
        self, 
        M: int = 16,
        epsilon: float = 1e-5,
        *args, 
        **kwargs
    ):
        """
        Implementation of ELBO (Evidence Lower-Bound)

        Args:
            M (int): Number of samples for the Monte-Carlo estimation of Reconstruction part
        """
        super().__init__(*args, **kwargs)
        self.M = M
        self.epsilon = epsilon
        
        
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
        N, dim = mean.shape
        # return 1./(2. * N) * (
        #     var.sum(-1) # (N,)
        #     - dim  # (d,)
        #     + mean.square().sum(-1) # (N,)
        #     - log(var.prod(-1) + self.epsilon) # (N,)
        # ).sum() / dim # for a proper ratio with the Reconstruction Loss
        
        # ----> 
        return 1./(2. * N * dim) * ( # for a proper ratio with the Reconstruction Loss
            exp(log_var) # (N, d)
            - 1  # ()
            + mean.square() # (N, d)
            - log_var # (N, d)
        ).sum() 
    
    
    def _reconstruction(
        self, 
        decoder: Any,
        stats: Tuple[Tensor, Tensor],
        target: Tensor
    ) -> Tensor:
        """
        Calculate the Reconstruction part of ELBO. This part of ELBO loss will use Binary 
        Entropy Loss, instead of L2 Norm

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
        mean, log_var = mean.unsqueeze(0), log_var.unsqueeze(0) # (1, N, d), (1, N, d) 
                
        N, C, H, W = target.shape # C = 3
        shape_MC = (self.M, N, C, H, W)        
        target = target.unsqueeze(0).expand(shape_MC).reshape(self.M*N, C, H, W) # (M*N, C, H, W)
        
        # create Monte-Carlo latent vectors
        noise_MC = randn_like(mean.expand(self.M, -1, -1)) # (M, N, d)
        latent_MC = mean + exp(log_var / 2)*noise_MC # (M, N, d)
        latent_MC = latent_MC.view(self.M*N, latent_MC.size(-1)) # (M*N, d)
        
        output_MC: Tensor = decoder(latent_MC) # (M*N, C, H, W)        
        return 1./(2. * decoder.std_dec**2) * binary_cross_entropy_with_logits(output_MC, target)
    
    
    def forward(
        self, 
        decoder: Any,
        stats: Tensor,
        X_input: Tensor,
    ) -> Tensor:
        """
        ELBO Loss Forward method

        Args:
            decoder (Any): The Decoder
            stats (Tensor): Statistic outputs from the Encoder
            X_input (Tensor): Input data

        Returns:
            Tensor: _description_
        """
        return self._prior_matching(*stats) + self._reconstruction(decoder, stats, X_input)