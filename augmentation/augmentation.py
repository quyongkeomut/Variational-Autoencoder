from typing import Any, Tuple, Optional

import torch
from torch.nn.modules.utils import _pair

from torchvision.transforms import v2


_VALID_NORM = v2.Compose([
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def _WITH_MASK_AUG(img_size: int | Tuple[int, int]):
    img_size = _pair(img_size)
    transform = v2.Compose([
        v2.RandomResizedCrop(size=img_size, scale=(0.85, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
    ])
    return transform


class CustomAug:
    def __init__(self, img_size) -> None:
        self.with_mask_aug = _WITH_MASK_AUG(img_size) 

    def __call__(
        self, 
        image, 
        label: Optional[Any] = None, 
    ) -> Any:
        if label is not None:
            return self.with_mask_aug(image, label)
        else:
            return self.with_mask_aug(image)