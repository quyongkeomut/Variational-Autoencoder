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


today = datetime.today()
formatted_today = today.strftime('%Y_%m_%d_%H_%M')


class AETrainer:
    def __init__(
        self,
        encoder: Type[Module],
        decoder: Type[Module],
        task: str,
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
        
        r"""
        _summary_

        Args:
            model (Type[Module]): _description_
            model_name (str): _description_
            criterion (_type_): _description_
            optimizer (_type_): _description_
            num_epochs (int): _description_
            start_epoch (int, optional): _description_. Defaults to 0.
            train_loader (_type_, optional): _description_. Defaults to None.
            out_dir (str, optional): _description_. Defaults to "./weights/AEweights".
            lr_scheduler_cosine (Optional[CosineAnnealingLR], optional): _description_. Defaults to None.
        """
        
        # setup model and ema model
        self.encoder = encoder.to(gpu_id)
        self.decoder = decoder.to(gpu_id)
        # self.decoder: DDP = DDP(
        #     self.decoder,
        #     device_ids=[gpu_id]
        # )
        
        self.ema_decoder = AveragedModel(
                model=self.decoder, #self.decoder.module, 
                multi_avg_fn=get_ema_multi_avg_fn(0.999)
            )
        self.save_encoder = save_encoder
        if save_encoder:
            # self.encoder: DDP = DDP(
            #     self.encoder,
            #     device_ids=[gpu_id]
            # )
            self.ema_encoder = AveragedModel(
                model=self.encoder, #self.encoder.module, 
                multi_avg_fn=get_ema_multi_avg_fn(0.999)
            )
        
        # setup criterion, optimizer and lr schedulers
        self.criterion = criterion
        self.optimizer = optimizer

        if lr_scheduler_cosine is None:
            self.lr_scheduler_cosine = CosineAnnealingLR(
                self.optimizer, 
                T_max=num_epochs,
                eta_min=5e-4
            )
        else:
            self.lr_scheduler_cosine = lr_scheduler_cosine
        
        # setup dataloaders
        self.train_loader = train_loader
        
        # setup training configs
        self.task = task
        self.num_epochs = num_epochs
        self.gpu_id = gpu_id
        self.start_epoch = start_epoch
        self.out_path = os.path.join(out_dir, formatted_today)
        self.decoder_out_path = os.path.join(out_dir, formatted_today, "decoder", )
        os.makedirs(self.decoder_out_path, exist_ok=True)
        if save_encoder:
            self.encoder_out_path = os.path.join(out_dir, formatted_today, "encoder", )
            os.makedirs(self.encoder_out_path, exist_ok=True)
        

    def fit(self):
        r"""
        Perform fitting loop and validation
        """
        if self.gpu_id == 0:
            train_csv_path = os.path.join(self.out_path, "train_metrics.csv")
            with open(train_csv_path, mode='w+', newline='') as train_csvfile:
                train_writer = csv.writer(train_csvfile)
                train_writer.writerow(['Epoch', 'Loss'])
        
        # Fitting loop
        for epoch in range(self.start_epoch, self.num_epochs):
            # Train
            train_metrics = self.train(epoch=epoch)
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
            torch.save({
                    "num_epoch": self.num_epochs,
                    "model_state_dict": self.ema_decoder.module.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                }, save_path)
            if self.save_encoder:
                save_path = os.path.join(self.encoder_out_path, f"encoder_ema_model.pth")
                torch.save({
                    "num_epoch": self.num_epochs,
                    "model_state_dict": self.ema_encoder.module.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                }, save_path)
        
        # train_csvfile.close()
        # val_csvfile.close()
     
     
    def _run_one_step(self, data) :
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
        ema_loss = 0
        
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
                    self.ema_decoder.update_parameters(self.decoder) #self.decoder.module)
                    if self.save_encoder:
                        self.ema_encoder.update_parameters(self.encoder) #self.encoder.module)
                    
                    # save this step for backup...
                    save_path = os.path.join(self.decoder_out_path, "decoder_last.pth")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.decoder.state_dict(), # self.decoder.module.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                    }, save_path)

                    if self.save_encoder:
                        save_path = os.path.join(self.encoder_out_path, "encoder_last.pth")
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": self.encoder.state_dict(), # self.encoder.module.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                        }, save_path)                    
                
                # clear cache
                # torch.cuda.empty_cache()
                # gc.collect()
                # break

        # calculate averaged loss
        loss = ema_loss
        print(f'Epoch {epoch+1} Loss: {loss:4f}')
        print()
        
        # also save model state dict along with optimizer's and scheduler's at the end of every epoch
        # os.makedirs(os.path.join(self.out_path, "epochs"), exist_ok=True)
        # save_path = os.path.join(self.out_path, "epochs", f"epoch_{epoch+1}.pth")
 
        # # os.makedirs(save_path, exist_ok=True)
        # torch.save({
        #     "epoch": epoch,
        #     "model_state_dict": self.model.state_dict(),
        #     "optimizer_state_dict": self.optimizer.state_dict(),
        #     "lr_increase_state_dict": self.lr_scheduler_increase.state_dict(),
        #     "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
        #     "loss": loss,
        #     "metrics": metrics
        #     }, save_path)
 
        # # reset metrics tracker after every training epoch
        # self.metrics.reset()

        return [epoch + 1, f"{loss:4f}"]


    @torch.inference_mode()
    def val(self, epoch):
        self.criterion.running_loss = 0.
        self.model.eval()
        ema_loss = 0
        print(F"VALIDATION PHASE EPOCH: {epoch+1}")
        with tqdm.tqdm(total=len(self.val_loader), desc=f'Epoch {epoch+1}/{self.num_epochs}', unit='batch') as pbar:
            for data in self.val_loader:
                # get data
                inputs = data[0].to(self.device)
                targets = data[1]
                targets = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)

                targets = targets.permute(0, 3, 1, 2)
                
                targets = targets.to(self.device)
                
                # compute output, loss and metrics
                outputs = self.model(inputs)

                _loss = self.criterion(outputs, targets)

                # Convert to numpy
                outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
                targets =  torch.argmax(targets, dim=1).cpu().detach().numpy()

                self.metrics.addBatch(outputs, targets)

                
                # calculate metrics of each task
                acc = self.metrics.pixelAccuracy()
                IoU = self.metrics.IntersectionOverUnion()
                mIoU = self.metrics.meanIntersectionOverUnion()


                metrics = {
                    "mIoU" : mIoU,
                    "IoU" : IoU,
                    "Acc" : acc
                }
                
                # update ema model 
                self.ema_model.update_parameters(self.model)

                # update progress bar
                ema_loss = 0.9*ema_loss + 0.1*_loss.item()
                pbar.set_postfix(loss=ema_loss, **metrics)
                pbar.update(1)  # Increase the progress bar
                
                # clear cache
                # torch.cuda.empty_cache()
                # gc.collect()
                # break

        # calculate averaged loss
        loss = ema_loss
        print(f'Epoch {epoch+1} Loss: {loss:4f}')
        print()
        
        # save the best model on IoU metric
        current_IoU = mIoU 
        if current_IoU >= self.best_IoU:
            files_to_delete = glob.glob(os.path.join(self.out_path, 'best_*'))
            for file_path in files_to_delete:
                os.remove(file_path)

            save_path = os.path.join(self.out_path, f"best_IoU_{round(current_IoU,4)}_epoch_{epoch + 1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_cosine_state_dict": self.lr_scheduler_cosine.state_dict(),
                "loss": loss,
                "metrics": metrics
                }, save_path)
            
            self.best_IoU = current_IoU

        # reset metrics tracker after every validating epoch
        self.metrics.reset()
        
        return [epoch + 1, f"{loss:4f}", f"{acc:4f}", f"{IoU:4f}", f"{mIoU:4f}"]