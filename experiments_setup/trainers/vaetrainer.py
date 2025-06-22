from typing import Dict
import tqdm
import os
from contextlib import nullcontext
import torch

from .base_trainer import BaseTrainer

      
class VAETrainer(BaseTrainer):
    METRIC_KEYS = ["Loss", "Prior Loss", "Recon Loss", "Mean", "Var"]
    
    
    def __init__(
        self, 
        num_recon_ahead: int = 5,
        *arg, 
        **kwargs
    ):
        super().__init__(*arg, **kwargs)
        self.num_recon_ahead = int(num_recon_ahead)
    
    
    def _on_train_epoch(
        self, 
        epoch: int, 
    ) -> Dict[str, float]:
        
        _metrics = {
            metric_key: None for metric_key in self.METRIC_KEYS
        }
        pb = tqdm.tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch')
        
        with pb if self.gpu_id == 0 else nullcontext() as pbar:
            for i, data in enumerate(self.train_loader):
                # Forward pass
                _losses = self._run_one_step(data)
                if (i + 1) % self.num_recon_ahead == 0:
                    _loss = _losses["Loss"]
                else:
                    _loss = _losses["Recon Loss"]
                
                # Optimization step
                _loss.backward(retain_graph=True)
                self.optimizer.step()
                
                # Update metrics and progress bar
                if self.gpu_id == 0:
                    # update progress bar
                    for metric_key, current_metric in _metrics.items():
                        if current_metric is not None:
                            numeric_metric = float(current_metric)
                            _metrics[metric_key] = f"{0.9*numeric_metric + 0.1*_losses[metric_key].detach().item():.4f}"
                        else:
                            _metrics[metric_key] = f"{_losses[metric_key].detach().item():.4f}"
                    pbar.set_postfix(_metrics)
                    pbar.update(1)
                    
                    # Update EMA modules 
                    if self.gpu_id == 0:
                        if self.is_ddp: 
                            self.ema_decoder.update_parameters(self.decoder.module)
                            if self.save_encoder:
                                self.ema_encoder.update_parameters(self.encoder.module)
                        else:
                            self.ema_decoder.update_parameters(self.decoder)
                            if self.save_encoder:
                                self.ema_encoder.update_parameters(self.encoder)
                    
                    # Save this step for backup...
                    save_path = os.path.join(self.decoder_out_path, "decoder_last.pth")
                    state_dicts = (
                        self.optimizer.state_dict(), 
                        self.lr_scheduler.state_dict()
                    )
                    if self.is_ddp:
                        decoder_state_dicts = (self.decoder.module.state_dict(),) + state_dicts
                    else:
                        decoder_state_dicts = (self.decoder.state_dict(),) + state_dicts
                    self._save_modules(epoch, save_path, *decoder_state_dicts)

                    if self.save_encoder:
                        save_path = os.path.join(self.encoder_out_path, "encoder_last.pth")
                        if self.is_ddp:
                            encoder_state_dicts = (self.encoder.module.state_dict(),) + state_dicts
                        else:
                            encoder_state_dicts = (self.encoder.state_dict(),) + state_dicts
                        self._save_modules(epoch, save_path, *encoder_state_dicts)             
                
                # clear cache
                torch.cuda.empty_cache()

        # calculate averaged loss
        return _metrics