import os
import time
import random
import numpy as np

import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group

from datasets.utils import get_dataloader
from experiments_setup.trainers.base_trainer import BaseTrainer
from experiments_setup.utils import (
    load_configs,
    get_args_argparse,
    get_modules_criterion_trainer,
    get_optimizer,
    set_ddp,
    set_env,
    set_seed
)


NUM_DEVICE = torch.cuda.device_count()


def main(
    # general arguments
    rank: int,
    world_size: int,
    model_type: str,
    dataset: str,
    is_ddp: bool,
    img_size: int | tuple,
    num_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    kwargs: dict, # model/criterion/trainer-dependent keyword arguments
):
    """
    Main function for training.

    Args:
        rank (int): GPU id. In the case of DDP, this will be loaded automatically by the DPP setup.
            In the normal training criteria, GPU id should be loaded by the default value 0.
        world_size (int): Number of device.
        model_type (str): Type of model to train.
        dataset (str): Dataset to train the model on.
        is_ddp (bool): Whether the training setting is DDP or normal criteria.
        img_size (int | tuple): Image size.
        num_epochs (int): Number of training epoch.
        batch_size (int): Batch size.
        lr (float): Learning rate. 
        seed (int): Random seed.
        kwargs (dict): Model/Criterion/Trainer-dependent keyword arguments
    """
    # SET ENVIRONMENT
    if is_ddp:
        set_ddp(rank, world_size)
    set_env()
    set_seed(seed + rank)
    
    # LOAD MODEL-DEPENDENT CONFIG
    configs = load_configs(model_type)

    # INITIALIZE THE ENCODER, DECODER, CRITERION AND TRAINER
    encoder, decoder, criterion, trainer = get_modules_criterion_trainer(
        model_type,
        configs["ENCODER_CONFIGS"],
        configs["DECODER_CONFIGS"],
        img_channels=configs["IMG_CHANNELS"],
        latent_dim=configs["LATENT_DIM"],
        device=rank,
        **kwargs 
    )
    
    # INITIALIZE DATALOADER
    train_loader = get_dataloader(dataset, is_ddp, batch_size, img_size=img_size)

    # INITIALIZE THE CRITERION
    criterion = criterion(**kwargs) 
    
    # INITIALIZE THE OPTIMIZER
    optimizer = get_optimizer(
        configs["OPTIMIZER_NAME"], 
        encoder,
        decoder,
        lr=lr,
        **configs["OPTIM_KWARGS"]
    )
    
    # OUT DIR: TO SAVE CHECKPOINTS
    out_dir = os.path.join(configs["LOGGING_KWARGS"]["save_dir"], dataset)
    
    # SETUP THE TRAINER
    trainer: BaseTrainer = trainer(
        is_ddp=is_ddp,
        encoder=encoder,
        decoder=decoder,
        base_lr=lr,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        train_loader=train_loader,
        out_dir=out_dir,
        gpu_id=rank,
        **kwargs
    )
    
    # RUN TRAINING PROCEDURE
    trainer.fit()
    if is_ddp:
        destroy_process_group()
    

if __name__ == "__main__":    
    # arguments parser
    args, additional_args = get_args_argparse()
    
    world_size = NUM_DEVICE
    args_tuple = (world_size, ) + tuple(args.values()) + (additional_args, )

    if args["is_ddp"]:
        mp.spawn(main, args=args_tuple, nprocs=world_size)
    else:
        main(0, *args_tuple)