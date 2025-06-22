from typing import Optional, Any, Dict, Tuple
import tqdm
from datetime import datetime
import os
import csv
import time
import random
from pathlib import Path
from contextlib import nullcontext

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.other_utils import save_animation

TODAY = datetime.today()


class BaseTrainer:

    METRIC_KEYS = []
    AVG_WEIGHT = 0.99
    ANIMATION_SIZE = 32
    FORMATTED_TODAY = TODAY.strftime('%Y_%m_%d_%H_%M')
    
    def __init__(
        self,
        is_ddp: bool,
        encoder: Module,
        decoder: Module,
        base_lr: float,
        criterion: Any,
        optimizer: Any,
        num_epochs: int,
        train_loader: Any,
        out_dir: str | Path,
        save_encoder: bool = True,
        start_epoch: int = 0,
        lr_scheduler: Optional[Any] = None,
        gpu_id: int = 0,
        *args,
        **kwargs
    ) -> None:
        """
        Generic Variational Autoencoder Trainer

        Args:
            is_ddp (bool): Decide whether the training setting is DDP or normal criteria. 
            encoder (Type[Module]): The encoder.
            decoder (Type[Module]): The decoder.
            base_lr (float): Baseline learning rate.
            criterion (_type_): Loss function.
            optimizer (_type_): Parameter optimizer.
            num_epochs (int): Number of training epochs.
            train_loader (_type_, optional): Decide whether saving the decoder. Defaults to None.
            out_dir (str, optional): Folder to save the weights.
            save_encoder (bool, optional): Decide whether saving the encoder. Defaults to True.
            start_epoch (int, optional): Start epoch index. Defaults to 0.
            lr_scheduler (Optional[CosineAnnealingLR], optional): Learning rate scheduler. 
                Defaults to None.
            gpu_id (int, optional): GPU index. Defaults to 0.
        """
        self.is_ddp = is_ddp 
        self.gpu_id = gpu_id
        self.base_lr = base_lr
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.save_encoder = save_encoder
        
        # setup model and ema model
        self.encoder = encoder.to(gpu_id)
        self.decoder = decoder.to(gpu_id)
        self._setup_encoder_decoder()
        
        # setup criterion, optimizer and lr schedulers
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self._setup_lr_scheduler()
        
        # setup dataloader(s)
        self.train_loader = train_loader
        
        # setup paths to save model
        self.out_path = os.path.join(out_dir, self.FORMATTED_TODAY)
        self.decoder_out_path = os.path.join(self.out_path, "decoder")
        os.makedirs(self.decoder_out_path, exist_ok=True)
        if save_encoder:
            self.encoder_out_path = os.path.join(self.out_path, "encoder")
            os.makedirs(self.encoder_out_path, exist_ok=True)
        
        # setup path to save results
        self.val_out_path = os.path.join(self.out_path, "val_results")
        os.makedirs(self.val_out_path, exist_ok=True)
            
        
    def _setup_encoder_decoder(self) -> None:
        if self.is_ddp:
            self.encoder: DDP = DDP(
                self.encoder,
                device_ids=[self.gpu_id]
            )
            self.decoder: DDP = DDP(
                self.decoder,
                device_ids=[self.gpu_id]
            )
            
        if self.gpu_id == 0:
            self.ema_encoder = AveragedModel(
                model=self.encoder.module if self.is_ddp else self.encoder, 
                multi_avg_fn=get_ema_multi_avg_fn(self.AVG_WEIGHT)
            )
            self.ema_decoder = AveragedModel(
                model=self.decoder.module if self.is_ddp else self.decoder, 
                multi_avg_fn=get_ema_multi_avg_fn(self.AVG_WEIGHT)
            )
    
    
    def _setup_lr_scheduler(self) -> None:
        if self.lr_scheduler is None:
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.num_epochs,
                eta_min=self.base_lr/2
            )
                
    
    def _save_modules(
        self, 
        epoch,
        save_path: str, 
        *state_dicts
    ):
        model_state_dict, optim_state_dict, scheduler_state_dict = state_dicts
        torch.save({
            "num_epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optim_state_dict,
            "lr_scheduler_state_dict": scheduler_state_dict,
        }, save_path)
    
    
    def fit(self):
        r"""
        Perform fitting loop and validation
        """
        # FOR LOGGING
        train_csv_path = os.path.join(self.out_path, "train_metrics.csv")
        with open(train_csv_path, mode='w+', newline='') as train_csvfile:
            train_writer = csv.writer(train_csvfile)
            train_writer.writerow(["Epoch"] + self.METRIC_KEYS)
        self._on_train_begin()
        
        # FITTING LOOP
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            # Train
            self._train(epoch, train_csv_path)
            # validating
            if (epoch + 1) % 5 == 0:
                self._validate(epoch)
                
        # AT THE END OF TRAINING
        self._on_train_end()


    def _on_train_begin(self):
        # FOR SAVING RESULTS FROM BEGINING TO THE END
        self.all_results = []
        
        
    def _on_train_end(self):
        if self.gpu_id == 0:
             # save EMA model
            save_path = os.path.join(self.decoder_out_path, f"decoder_ema_model.pth")
            decoder_state_dicts = (
                self.ema_decoder.module.state_dict(), 
                self.optimizer.state_dict(), 
                self.lr_scheduler.state_dict()
            )
            self._save_modules(self.num_epochs, save_path, *decoder_state_dicts)
            if self.save_encoder:
                save_path = os.path.join(self.encoder_out_path, f"encoder_ema_model.pth")
                encoder_state_dicts = (
                    self.ema_encoder.module.state_dict(), 
                    self.optimizer.state_dict(), 
                    self.lr_scheduler.state_dict()
                )
                self._save_modules(self.num_epochs, save_path, *encoder_state_dicts)
        
            # save all generated images
            res_save_path = os.path.join(self.out_path, f"All_Evaluation_Results.gif")
            save_animation(
                self.all_results, 
                res_save_path, 
                interval=100, 
                repeat_delay=3000
            )
        
    
    def _run_one_step(self, data: Tuple[Tensor, Any]) -> Dict[str, Tensor]:
        # clear gradient
        self.encoder.zero_grad(set_to_none=True)
        self.decoder.zero_grad(set_to_none=True)
        
        # get data
        inputs = data[0].to(self.gpu_id)
            
        # compute output, loss and metrics
        encoder_outputs = self.encoder(inputs)
        
        _losses = self.criterion(self.decoder, encoder_outputs, inputs)
        return _losses


    def _on_train_epoch_begin(self, epoch: int) -> None:
        self.decoder.train()
        self.encoder.train()
        
        if self.gpu_id == 0:
            print(f"TRAINING PHASE - EPOCH {epoch+1}")
        time.sleep(2)
    
    
    def _on_train_epoch(
        self, 
        epoch: int, 
    ) -> Dict[str, float]:
        
        _metrics = {
            metric_key: None for metric_key in self.METRIC_KEYS
        }
        pb = tqdm.tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch')
        
        with pb if self.gpu_id == 0 else nullcontext() as pbar:
            for data in self.train_loader:
                
                # Forward pass
                _losses = self._run_one_step(data)
                _loss = _losses["Loss"]
                
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
    
    
    def _on_train_epoch_end(
        self, 
        epoch: int,
        train_metrics: Dict[str, float| None],
        train_csv_path: str | Path
    ):
        if self.gpu_id == 0:
            _train_metrics_list = [epoch + 1]
            for metric_key in self.METRIC_KEYS:
                _train_metrics_list.append(
                    float(train_metrics[metric_key]) if train_metrics[metric_key] is not None else 0
                )
            
            # write results
            with open(train_csv_path, mode='a', newline='') as train_csvfile:
                train_writer = csv.writer(train_csvfile)
                train_writer.writerow(_train_metrics_list)
        # lr schedulers step
        self.lr_scheduler.step()
        torch.cuda.empty_cache()
        time.sleep(2)
    
    
    def _train(
        self, 
        epoch: int, 
        train_csv_path: str | Path
    ):
        self._on_train_epoch_begin(epoch)
        _metrics = self._on_train_epoch(epoch)
        self._on_train_epoch_end(epoch, _metrics, train_csv_path)

        
    def _on_val_epoch_begin(self):
        self.decoder.eval()
    
    
    def _on_val_epoch(self, epoch: int):
        save_path = os.path.join(self.val_out_path, f"Evaluation result - EPOCH {epoch+1}.gif")
        if self.gpu_id == 0:
            if self.is_ddp:
                results = self.decoder.module.sample(self.ANIMATION_SIZE)
            else:
                results = self.decoder.sample(self.ANIMATION_SIZE)
            self.all_results.extend(random.sample(results, self.ANIMATION_SIZE//2))
            return save_animation(results, save_path)
            
    
    def _on_val_epoch_end(self, epoch: int):
        if self.gpu_id == 0:
            print(f"EVALUATION STEP - EPOCH {epoch+1} - DONE")
        time.sleep(2)
    
    
    @torch.no_grad()
    def _validate(self, epoch):
        self._on_val_epoch_begin()
        self._on_val_epoch(epoch)
        self._on_val_epoch_end(epoch)
  