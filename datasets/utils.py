from typing import Tuple

from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from datasets import *
from augmentation.augmentation import *


NUM_WORKERS = 4
IS_DROP_LAST =  True
IS_PIN_MEMORY = True


def _get_dataset(
    dataset: str,
    *args,
    **kwargs
):
    AVAILABLE_DATASETS = {
        "flowers102": ConcatDataset([
            flowers102.Flowers102AE(transform=CustomAug(*args, **kwargs), split=split)
            for split in ["train", "val", "test"]
        ]),
        "mnist": mnist.load_transformed_MNIST(*args, **kwargs),
        "fashion_mnist": fashion_mnist.load_transformed_fashionMNIST(*args, **kwargs)
    }
    try:
        return AVAILABLE_DATASETS[dataset]
    except KeyError:
        raise KeyError(
            f"Dataset must be one of these {list(AVAILABLE_DATASETS.keys)}, got {dataset} instead"
        )

def get_dataloader(
    dataset: str,
    is_ddp: bool,
    batch_size: int,
    *args, 
    **kwargs
) -> DataLoader:
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False if is_ddp else True,
        "sampler": DistributedSampler(dataset) if is_ddp else None,
        "num_workers": NUM_WORKERS,
        "drop_last": IS_DROP_LAST,
        "pin_memory": IS_PIN_MEMORY,
    }
    dataset = _get_dataset(dataset, *args, **kwargs)
    return DataLoader(dataset=dataset, **dataloader_kwargs)