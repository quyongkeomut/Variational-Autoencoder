from typing import Optional, Dict, Tuple
import argparse
import os
import time
import random
import yaml
import numpy as np

import torch
from torch.nn import Module
from torch.distributed import init_process_group

from neural_nets.autoencoders import *
from losses import *
from optimizer.optimizer import OPTIMIZERS
from .trainers import *
import datasets


VAE_CRITERION_TRAINER = {
    "VAE": (
        vae.VAEEncoder, 
        vae.VAEDecoder, 
        ELBO.ELBOLoss,
        vaetrainer.VAETrainer
    ),
    "VQ-VAE": (
        vq_vae.VQ_VAEEncoder, 
        vq_vae.VQ_VAEDecoder, 
        VQ.VQ_VAELoss,
        vq_vaetrainer.VQ_VAETrainer
    ),
}


def get_args_argparse() -> Tuple[Dict, Dict]:
    """
    Getting arguments parser method

    Returns:
        Dict: A dictionary contains all keyword arguments for experiment.
    """
    def parse_kwargs(args):
        kwargs = {}
        key = None
        for arg in args:
            if arg.startswith('--'):
                key = arg.lstrip('--')
                kwargs[key] = True  # default to flag
            elif key:
                kwargs[key] = arg
                key = None
        return kwargs
    
    parser = argparse.ArgumentParser(description='Training args')

    # model type
    parser.add_argument(
        '--model', type=str, default="VAE", required=False, 
        help=f'Type of model, valid values are one of {list(VAE_CRITERION_TRAINER.keys())}'
    )
    
    # dataset
    parser.add_argument(
        '--dataset', type=str, default="mnist", required=False, 
        help=f'Dataset to train model on, valid values are one of {datasets.__all__}'
    )
    
    # DDP (Distributed Data Parallel) option
    parser.add_argument(
        '--is_ddp', action="store_true", 
        help='Option for choosing training in DDP or normal training criteria'
    )
    
    # image size
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    
    
    # Number of training epochs
    parser.add_argument('--epochs', type=int, default=50, help='Num epochs')
    
    # Batch size
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    
    # Learning rate
    parser.add_argument(
        '--lr', type=float, default=1e-3, 
        help='Learning rate'
    )
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training')
    
    base_args, model_args = parser.parse_known_args()
    return vars(base_args), parse_kwargs(model_args)


def get_modules_criterion_trainer(
    model_type: str,
    encoder_configs: Dict,
    decoder_configs: Dict,
    backend: str = "cudagraphs",
    *args,
    **kwargs
) -> Tuple[base_ae.Encoder, base_ae.Decoder, base_trainer.BaseTrainer, Module]:
    # load type of VAE
    try:
        encoder, decoder, criterion, trainer = VAE_CRITERION_TRAINER[model_type]
        encoder = encoder(*args, **encoder_configs, **kwargs)
        decoder = decoder(*args, **decoder_configs, **kwargs)
    except KeyError:
        raise KeyError(
            f"Model must be one of {list(VAE_CRITERION_TRAINER.keys)}, got {model_type} instead"
        )
        
    # Compile modules
    encoder.compile(fullgraph=True, backend=backend)
    decoder.compile(fullgraph=True, backend=backend)
    return encoder, decoder, criterion, trainer
    
    
def get_optimizer(
    optim_name: str,
    *modules,
    **optim_kwargs
) -> ...:
    optimizer = OPTIMIZERS[optim_name](
        [
            {"params": module.parameters()}
            for module in modules
        ],
        **optim_kwargs,
    )
    return optimizer
    
    
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
        time.sleep(50)
        

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
    

def load_configs(model_type: str) -> Dict:
    filepath = f"./experiments_setup/configs/{model_type}.yaml"
    try:
        with open(filepath, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
    except:
        print(f"Model must be one of {list(VAE_CRITERION_TRAINER.keys)}, got {model_type} instead")
    return config