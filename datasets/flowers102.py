from typing import Tuple
import PIL.Image

import torch
from torch import Tensor

from torchvision import tv_tensors
from torchvision.datasets import Flowers102 
import torchvision.transforms as transforms
from torchvision.transforms.v2 import functional as F
import matplotlib.pyplot as plt


class BaseFlowers102(Flowers102):
    def __init__(
        self, 
        root = r"./data", 
        split = "train", 
        transform = None, 
        target_transform = None, 
        download = True,
        *args,
        **kwargs
    ):
        super().__init__(root, split, transform, target_transform, download)
    
    
class Flowers102AE(BaseFlowers102):
    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        image_file = self._image_files[idx]
        
        image = PIL.Image.open(image_file).convert("RGB")
        if self.transform:
            image = self.transform(image)    
        image: Tensor = transforms.ToTensor()(image) # [0, 1] range
        # image = tv_tensors.Image(image)
        return (image, 0.0)