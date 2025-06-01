import os
import time
import random
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

from neural_nets.autoencoders.trainer import AETrainer
from neural_nets.autoencoders.ae import VAEEncoder, VAEDecoder

from optimizer.optimizer import OPTIMIZERS

from augmentation.augmentation import CustomAug


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    

def ddp_setup(rank: int, world_size: int):
    """
    Init DDP

    Args:
        rank (int): A unique identifier that is assigned to each process
        world_size (int): Total process in a group
    """
    # this machine coordinates the communication across all processes
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12397"
    init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    if rank == 0:
        time.sleep(30)


NUM_DEVICE = torch.cuda.device_count()


def main(
    rank: int,
    # world_size: int,
    task: str,
    img_size: int | tuple,
    std_dec: float,
    num_epochs: int,
    batch_size: int,
    ckpt_encoder,
    ckpt_decoder
):
    # ddp_setup(rank, world_size)
    
    if task == "flowers102":
        from datasets.flowers102 import Flowers102AE
        from experiments_setup.flowers102.backbone_config import get_ae_configs
        from experiments_setup.flowers102.experiment_config import (
            OPTIMIZER_NAME,
            OPTIM_ARGS
        )
        TRAIN_DS = Flowers102AE(transform=CustomAug(img_size), split="train")
        VAL_DS = Flowers102AE(transform=CustomAug(img_size), split="val")
        TEST_DS = Flowers102AE(transform=CustomAug(img_size), split="test")
        TRAIN_DS = torch.utils.data.ConcatDataset([TRAIN_DS, VAL_DS, TEST_DS])

    elif task == "fashion_mnist":
        from datasets.fashion_mnist import load_transformed_fashionMNIST
        from experiments_setup.mnist.backbone_config import get_ae_configs
        from experiments_setup.mnist.experiment_config import (
            OPTIMIZER_NAME,
            OPTIM_ARGS
        )
        TRAIN_DS = load_transformed_fashionMNIST(img_size)
        
    elif task == "mnist":
        from datasets.mnist import load_transformed_MNIST
        from experiments_setup.mnist.backbone_config import get_ae_configs
        from experiments_setup.mnist.experiment_config import (
            OPTIMIZER_NAME,
            OPTIM_ARGS
        )
        TRAIN_DS = load_transformed_MNIST(img_size)
    
    AE_CONFIGS = get_ae_configs()        
    IS_PIN_MEMORY = True
    NUM_WORKERS = 4
    
    out_dir = os.path.join("./weights/VAEweights", task)

    # these hyperparams depend on the dataset / experiment
    otim = OPTIMIZER_NAME
    optim_args = OPTIM_ARGS

    # initialize the encoder, decoder and optimizer
    latent_shape = _pair(img_size//8)
    decoder = VAEDecoder(
        std_dec=std_dec, 
        **AE_CONFIGS["decoder"], 
        device=rank
    )
    encoder = VAEEncoder(
        latent_shape=latent_shape,
        **AE_CONFIGS["encoder"], 
        device=rank
    )
    optimizer = OPTIMIZERS[otim](
        [
            {"params": encoder.parameters()},
            {"params": decoder.parameters()},
        ],
        **optim_args,
    )
    
    # print(optimizer)

    # load check point...
    if ckpt_encoder:
        ckpt_encoder = torch.load(ckpt_encoder, weights_only=False)
        encoder.load_state_dict(ckpt_encoder["model_state_dict"])        

    if ckpt_decoder:
        ckpt_decoder = torch.load(ckpt_decoder, weights_only=False)
        decoder.load_state_dict(ckpt_decoder["model_state_dict"])
        
    last_epoch = 0
    lr_scheduler_cosine = None
    
    # Compile modules
    encoder.compile(fullgraph=True, backend="cudagraphs")
    decoder.compile(fullgraph=True, backend="cudagraphs")   

    # setup dataloaders
    train_loader = DataLoader(
        TRAIN_DS, 
        batch_size=batch_size, 
        shuffle=True, 
        # sampler=DistributedSampler(TRAIN_DS),
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=IS_PIN_MEMORY
    )

    # call the trainer
    from losses.ELBO import ELBOLoss
    criterion = ELBOLoss(M=16)
    trainer = AETrainer(
        encoder=encoder,
        decoder=decoder,
        task=task,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        start_epoch=last_epoch,
        save_encoder=True,
        train_loader=train_loader,
        out_dir=out_dir,
        lr_scheduler_cosine=lr_scheduler_cosine,
        gpu_id=rank
    )
    trainer.fit()
    
    # destroy_process_group()


if __name__ == "__main__":

    # environment setup
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # os.environ["TORCH_LOGS"] = "+dynamo"
    # os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # for debugging
    os.environ["TORCHDYNAMO_DYNAMIC_SHAPES"] = "0"

    # The flags below controls whether to allow TF32 on cuda and cuDNN
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    import warnings
    warnings.filterwarnings("ignore")
    
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # arguments parser
    import argparse

    parser = argparse.ArgumentParser(description='Training args')

    parser.add_argument('--task', type=str, default="mnist", required=False, help='Dataset to train model on, valid values are one of [flowers102, mnist, fashion_mnist]')
    parser.add_argument('--img_size', type=int, default=32, help='Image size')
    # parser.add_argument('--scale', type=float, default=0.25, required=False, help='Model scale')
    parser.add_argument('--std_dec', type=float, default=1.25, help='Standard deviation of decoder')
    parser.add_argument('--epochs', type=int, default=10, help='Num epochs')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='seed for training')
    parser.add_argument('--ckpt_encoder', nargs='?', default = None,
                        help='Checkpoint of pretrained encoder for coutinue training')
    parser.add_argument('--ckpt_decoder', nargs='?', default = None,
                        help='Checkpoint of pretrained decoder for coutinue training')

    args = parser.parse_args()
    
    # setup model hyperparameters and training parameters
    task = args.task
    img_size = args.img_size
    std_dec = args.std_dec
    num_epochs = args.epochs
    batch_size = args.batch
    seed = args.seed
    ckpt_encoder = args.ckpt_encoder
    ckpt_decoder = args.ckpt_decoder
    set_seed(seed)
    
    world_size = NUM_DEVICE
    args = (
        0, # device
        # world_size, 
        task, 
        img_size, 
        std_dec,
        num_epochs, 
        batch_size, 
        ckpt_encoder,
        ckpt_decoder
    )
    # mp.spawn(main, args=args, nprocs=world_size)
    main(*args)