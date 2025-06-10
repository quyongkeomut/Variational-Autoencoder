import tqdm
from datetime import datetime
import os
import csv
import glob
from typing import Type, Optional
from contextlib import nullcontext

import torch
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

from utils.other_utils import show_animation

AVG_WEIGHT = 0.99
ANIMATION_SIZE = 32
TODAY = datetime.today()
FORMATTED_TODAY = TODAY.strftime('%Y_%m_%d_%H_%M')


class AETrainer:
    def __init__(
        self,
        is_ddp: bool,
        encoder: Type[Module],
        decoder: Type[Module],
        latent_dim: int,
        task: str,
        base_lr: float,
        criterion,
        optimizer,
        num_epochs: int,
        start_epoch: int = 0,
        save_encoder: bool = False,
        train_loader = None,
        out_dir = "./weights/AEweights",
        lr_scheduler_cosine: Optional[CosineAnnealingLR] = None,
        gpu_id: int = 0
    ) -> None:
        """
        Variational Autoencoder Trainer

        Args:
            is_ddp (bool): Decide whether the training setting is DDP or normal criteria. 
            encoder (Type[Module]): The encoder.
            decoder (Type[Module]): The decoder.
            latent_dim (int): Dimension of latent vector.
            task (str): Task to train the model on.
            base_lr (float): Baseline learning rate.
            criterion (_type_): Loss function.
            optimizer (_type_): Parameter optimizer.
            num_epochs (int): Number of training epochs.
            start_epoch (int, optional): Start epoch index. Defaults to 0.
            save_encoder (bool, optional): Decide whether saving the encoder. Defaults to False.
            train_loader (_type_, optional): Decide whether saving the decoder. Defaults to None.
            out_dir (str, optional): Folder to save the weights. Defaults to "./weights/AEweights".
            lr_scheduler_cosine (Optional[CosineAnnealingLR], optional): Learning rate scheduler. 
                Defaults to None.
            gpu_id (int, optional): GPU index. Defaults to 0.
        """
        self.task = task
        self.latent_dim = latent_dim
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
        self._setup_criterion_optim_scheduler(criterion, optimizer, lr_scheduler_cosine)
        
        # setup dataloader(s)
        self.train_loader = train_loader
        
        # setup paths to save model
        self.out_path = os.path.join(out_dir, FORMATTED_TODAY)
        self.decoder_out_path = os.path.join(out_dir, FORMATTED_TODAY, "decoder", )
        os.makedirs(self.decoder_out_path, exist_ok=True)
        if save_encoder:
            self.encoder_out_path = os.path.join(out_dir, FORMATTED_TODAY, "encoder", )
            os.makedirs(self.encoder_out_path, exist_ok=True)
        
    def _setup_encoder_decoder(self):
        if self.is_ddp:
            self.encoder: DDP = DDP(
                self.encoder,
                device_ids=[self.gpu_id]
            )
            self.decoder: DDP = DDP(
                self.decoder,
                device_ids=[self.gpu_id]
            )

            self.ema_encoder = AveragedModel(
                model=self.encoder.module, 
                multi_avg_fn=get_ema_multi_avg_fn(AVG_WEIGHT)
            )
            self.ema_decoder = AveragedModel(
                    model=self.decoder.module, 
                    multi_avg_fn=get_ema_multi_avg_fn(AVG_WEIGHT)
                )
            
        else:
            self.ema_encoder = AveragedModel(
                model=self.encoder, 
                multi_avg_fn=get_ema_multi_avg_fn(AVG_WEIGHT)
            )
            self.ema_decoder = AveragedModel(
                    model=self.decoder, 
                    multi_avg_fn=get_ema_multi_avg_fn(AVG_WEIGHT)
                )
    
    def _setup_criterion_optim_scheduler(
        self,
        criterion,
        optimizer,
        lr_scheduler_cosine
    ):
        self.criterion = criterion
        self.optimizer = optimizer

        if lr_scheduler_cosine is None:
            self.lr_scheduler_cosine = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.num_epochs,
                eta_min=self.base_lr/2
            )
        else:
            self.lr_scheduler_cosine = lr_scheduler_cosine
    
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
            "lr_cosine_state_dict": scheduler_state_dict,
        }, save_path)
    
    def fit(self):
        r"""
        Perform fitting loop and validation
        """
        # For logging
        if self.gpu_id == 0:
            train_csv_path = os.path.join(self.out_path, "train_metrics.csv")
            with open(train_csv_path, mode='w+', newline='') as train_csvfile:
                train_writer = csv.writer(train_csvfile)
                train_writer.writerow(['Epoch', 'Loss'])
        # Fitting loop
        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_metrics = self.train(epoch=epoch)
            # self.eval(epoch=epoch)
            if self.gpu_id == 0:
                with open(train_csv_path, mode='a', newline='') as train_csvfile:
                    train_writer = csv.writer(train_csvfile)
                    train_writer.writerow(train_metrics)
            # lr schedulers step
            self.lr_scheduler_cosine.step()
            torch.cuda.empty_cache()

        # save EMA model
        if self.gpu_id == 0:
            save_path = os.path.join(self.decoder_out_path, f"decoder_ema_model.pth")
            decoder_state_dicts = (
                self.ema_decoder.module.state_dict(), 
                self.optimizer.state_dict(), 
                self.lr_scheduler_cosine.state_dict()
            )
            self._save_modules(self.num_epochs, save_path, *decoder_state_dicts)
            if self.save_encoder:
                save_path = os.path.join(self.encoder_out_path, f"encoder_ema_model.pth")
                encoder_state_dicts = (
                    self.ema_encoder.module.state_dict(), 
                    self.optimizer.state_dict(), 
                    self.lr_scheduler_cosine.state_dict()
                )
                self._save_modules(self.num_epochs, save_path, *encoder_state_dicts)
     
    def _run_one_step(self, data):
        # clear gradient
        self.encoder.zero_grad(set_to_none=True)
        self.decoder.zero_grad(set_to_none=True)
        
        # get data
        inputs = data[0].to(self.gpu_id)
        if self.task != "flowers102":
            targets = inputs
        else:
            targets = data[1].to(self.gpu_id)
            
        # compute output, loss and metrics
        with nullcontext() if self.save_encoder else torch.no_grad():
            stats = self.encoder(inputs)
        
        _loss = self.criterion(self.decoder, stats, targets)
        return _loss
        
    def train(self, epoch):
        self.decoder.train()
        if self.save_encoder:
            self.encoder.train()
        else:
            self.encoder.eval()
        ema_loss = 0.0
        
        if self.gpu_id == 0:
            print(f"TRAINING PHASE EPOCH: {epoch+1}")

        pb = tqdm.tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch')
        with pb if self.gpu_id == 0 else nullcontext() as pbar:
            for data in self.train_loader:
                _loss = self._run_one_step(data)
                
                # optimization step
                _loss.backward(retain_graph=True)
                self.optimizer.step()
                
                if self.gpu_id == 0:
                    # update progress bar
                    ema_loss = 0.9*ema_loss + 0.1*_loss.item()
                    pbar.set_postfix(loss=ema_loss)
                    pbar.update(1)  # Increase the progress bar
                    
                    # update ema model 
                    if self.is_ddp:
                        self.ema_decoder.update_parameters(self.decoder.module)
                        if self.save_encoder:
                            self.ema_encoder.update_parameters(self.encoder.module)
                    else:
                        self.ema_decoder.update_parameters(self.decoder)
                        if self.save_encoder:
                            self.ema_encoder.update_parameters(self.encoder)
                    
                    # save this step for backup...
                    save_path = os.path.join(self.decoder_out_path, "decoder_last.pth")
                    state_dicts = (
                        self.optimizer.state_dict(), 
                        self.lr_scheduler_cosine.state_dict()
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
                # break

        # calculate averaged loss
        loss = ema_loss
        if self.gpu_id == 0:
            print(f'Epoch {epoch+1} Loss: {loss:4f}')
            print()
        return [epoch + 1, f"{loss:4f}"]

    @torch.no_grad()
    def eval(self, epoch):
        if self.gpu_id == 0:
            print(f"EVALUATION RESULTS EPOCH: {epoch+1}")
            self.decoder.eval()
            results = []
            for i in range(ANIMATION_SIZE):
                latent = torch.randn(1, self.latent_dim, device=0)
                results.append(self.decoder(latent))
            return show_animation(results)