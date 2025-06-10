import os
import time
import random
import numpy as np

import torch
import torch.multiprocessing as mp
from torch.nn.modules.utils import _pair
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from neural_nets.autoencoders.trainer import AETrainer
from neural_nets.autoencoders.vae import (
    VAEEncoder, 
    VAEDecoder,
    PRE_H,
    PRE_W
)

from optimizer.optimizer import OPTIMIZERS


NUM_DEVICE = torch.cuda.device_count()


def get_args() -> None:
    """
    Getting arguments parser method

    Returns:
        None
    """
    import argparse
    parser = argparse.ArgumentParser(description='Training args')

    # task
    parser.add_argument(
        '--task', type=str, default="mnist", required=False, 
        help='Dataset to train model on, valid values are one of [flowers102, mnist, fashion_mnist]'
    )
    
    # DDP (Distributed Data Parallel) option
    parser.add_argument(
        '--is_ddp', action="store_true", 
        help='Option for choosing training in DDP or normal training criteria'
    )
    
    # image size
    parser.add_argument('--img_size', type=int, default=32, help='Image size')
    
    # standard deviation of Decoder
    parser.add_argument(
        '--beta', type=float, default=1, 
        help='Beta coeficient of Beta-VAE'
    )
    
    # Learning rate
    parser.add_argument(
        '--lr', type=float, default=1e-3, 
        help='Learning rate'
    )
    
    # Number of Monte Carlo samples
    parser.add_argument(
        '--MC', type=int, default=1, 
        help='Number of Monte Carlo samples'
    )
    
    # Number of training epochs
    parser.add_argument('--epochs', type=int, default=50, help='Num epochs')
    
    # Batch size
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training')
    
    # Checkpoint of encoder to resume
    parser.add_argument(
        '--ckpt_encoder', nargs='?', default = None,
        help='Checkpoint of pretrained encoder for coutinue training'
    )
    
    # Checkpoint of decoder to resume
    parser.add_argument(
        '--ckpt_decoder', nargs='?', default = None,
        help='Checkpoint of pretrained decoder for coutinue training'
    )
    args = parser.parse_args()
    return args


def set_seed(seed: int) -> None:
    """
    Setting random seed method

    Args:
        seed (int): Seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    

def set_ddp(rank: int, world_size: int) -> None:
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
        

def set_env() -> None:
    """
    Method for setting other enviroment variables
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # os.environ["TORCH_LOGS"] = "+dynamo"
    # os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # for debugging
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


def main(
    rank: int,
    world_size: int,
    is_ddp: bool,
    task: str,
    img_size: int | tuple,
    beta: float,
    lr: float,
    MC: int,
    num_epochs: int,
    batch_size: int,
    ckpt_encoder,
    ckpt_decoder
):
    """
    Main function for training

    Args:
        rank (int): GPU id. In the case of DDP, this will be loaded automatically by the DPP setup.
            In the normal training criteria, GPU id should be loaded by the default value 0.
        world_size (int): Number of device.
        is_ddp (bool): Whether the training setting is DDP or normal criteria.
        task (str): Task to train the model on.
        beta (float): Beta coeficient for Beta-VAE.
        lr (float): Learning rate
        MC (int): Number of Monte Carlo samples for the Reconstruction loss.
        img_size (int | tuple): Image size.
        num_epochs (int): Number of training epoch.
        batch_size (int): Batch size.
        ckpt_encoder (_type_): Checkpoint of encoder to continue the training process.
        ckpt_decoder (_type_): Checkpoint of decoder to continue the training process.
    """
    def _get_configs(
        task: str,
        img_size: int | tuple,
    ):
        """
        Return dataset and configuration elements of model
        """
        if task == "flowers102":
            from datasets.flowers102 import Flowers102AE
            from experiments_setup.flowers102.backbone_config import get_ae_configs
            from experiments_setup.flowers102.experiment_config import (
                OPTIMIZER_NAME,
                OPTIM_ARGS
            )
            from augmentation.augmentation import CustomAug
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
            
        return TRAIN_DS, (get_ae_configs, OPTIMIZER_NAME, OPTIM_ARGS)
    
    def _get_modules(
        img_size,
        ckpt_encoder,
        ckpt_decoder,
        backend: str = "cudagraphs"
    ):
        latent_shape = _pair(img_size//8)
        decoder = VAEDecoder(
            **AE_CONFIGS["decoder"], 
            device=rank
        )
        encoder = VAEEncoder(
            latent_shape=latent_shape,
            **AE_CONFIGS["encoder"], 
            device=rank
        )
        optimizer = OPTIMIZERS[OPTIMIZER_NAME](
            [
                {"params": encoder.parameters()},
                {"params": decoder.parameters()},
            ],
            lr=lr,
            **OPTIM_ARGS,
        )

        # load check point...
        if ckpt_encoder:
            ckpt_encoder = torch.load(ckpt_encoder, weights_only=False)
            encoder.load_state_dict(ckpt_encoder["model_state_dict"])        

        if ckpt_decoder:
            ckpt_decoder = torch.load(ckpt_decoder, weights_only=False)
            decoder.load_state_dict(ckpt_decoder["model_state_dict"])

        # Compile modules
        encoder.compile(fullgraph=True, backend=backend)
        decoder.compile(fullgraph=True, backend=backend)
        return encoder, decoder, optimizer
        
    if is_ddp:
        set_ddp(rank, world_size)
    
    # these hyperparams depend on the dataset / experimen
    TRAIN_DS, (get_ae_configs, OPTIMIZER_NAME, OPTIM_ARGS) = _get_configs(task, img_size)
    AE_CONFIGS = get_ae_configs()        
    IS_PIN_MEMORY = True
    NUM_WORKERS = 4
    
    # out_dir = os.path.join("./weights/VAEweights", task)
    out_dir = os.path.join("/kaggle/working/weights/VAEweights", task)

    # initialize the encoder, decoder and optimizer
    encoder, decoder, optimizer = _get_modules(img_size, ckpt_encoder, ckpt_decoder)
    
    # setup dataloaders
    train_loader_args = {
        "dataset": TRAIN_DS, 
        "batch_size": batch_size, 
        "num_workers": NUM_WORKERS,
        "drop_last": True,
        "pin_memory": IS_PIN_MEMORY
    }
    if is_ddp:
        train_loader = DataLoader(
            shuffle=False, 
            sampler=DistributedSampler(TRAIN_DS),
            **train_loader_args
        )
    else:
        train_loader = DataLoader(
            shuffle=True, 
            **train_loader_args
        )

    # setup the trainer
    from losses.ELBO import ELBOLoss
    
    latent_dim = AE_CONFIGS["decoder"]["latent_channels"] * PRE_H * PRE_W
    last_epoch = 0
    lr_scheduler_cosine = None
    criterion = ELBOLoss(M=MC, beta=beta)
    trainer = AETrainer(
        is_ddp=is_ddp,
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        task=task,
        base_lr=lr,
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
    
    if is_ddp:
        destroy_process_group()


if __name__ == "__main__":
    # environment setup
    set_env()
    
    # arguments parser
    args = get_args()
    
    # setup model hyperparameters and training parameters
    task = args.task
    is_ddp = args.is_ddp
    img_size = args.img_size
    beta = args.beta
    lr = args.lr
    MC = args.MC
    num_epochs = args.epochs
    batch_size = args.batch
    seed = args.seed
    ckpt_encoder = args.ckpt_encoder
    ckpt_decoder = args.ckpt_decoder
    set_seed(seed)
    
    world_size = NUM_DEVICE
    args = (
        world_size, 
        is_ddp,
        task, 
        img_size, 
        beta,
        lr,
        MC,
        num_epochs, 
        batch_size, 
        ckpt_encoder,
        ckpt_decoder
    )

    if is_ddp:
        mp.spawn(main, args=args, nprocs=world_size)
    else:
        main(0, *args)