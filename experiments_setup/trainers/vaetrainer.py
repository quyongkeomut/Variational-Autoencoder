from typing import Dict, Tuple, Any
import tqdm
import os
from contextlib import nullcontext

import torch
from torch import Tensor

from .base_trainer import BaseTrainer

      
class VAETrainer(BaseTrainer):
    METRIC_KEYS = ["Loss", "Prior Loss", "Recon Loss", "Mean", "Var"]
    
    
    def __init__(
        self, 
        num_recon_ahead: int = 1,
        *arg, 
        **kwargs
    ):
        super().__init__(*arg, **kwargs)
        self.num_recon_ahead = int(num_recon_ahead)
    
    
    def _optimize_step(
        self, 
        data_idx: int,
        data: Tuple[Tensor, Any]
    ) -> Dict[str, Tensor]:
        # Forward pass
        _losses = self._forward_pass(data)
        
        # Backward pass
        if (data_idx + 1) % self.num_recon_ahead == 0:
            _loss = _losses["Loss"]
        else:
            _loss = _losses["Recon Loss"]
        _loss.backward(retain_graph=True)
        
        # Accumulate gradient and Optimize step
        if self._current_gradient_step % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            
            # clear gradient
            self.decoder.zero_grad(set_to_none=True)
            self.encoder.zero_grad(set_to_none=True)
            
        self._current_gradient_step += 1
        return _losses