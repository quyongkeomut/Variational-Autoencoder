from typing import List
from PIL import Image
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML

import torch
from torch import Tensor
from torch.nn.modules.utils import _pair
import torchvision.transforms as transforms


def to_image(
    tensor: Tensor, 
    to_pil: bool = True
) -> Tensor | Image.Image:
    """
    Transform a tensor into image

    Args:
        tensor (Tensor): Input tensor.
        to_pil (bool, optional): Whether transform it to PIL Image. Defaults to True.

    Shape:
        tensor: (C, H, W)
    
    Returns:
        Image: Output image
    """
    ones = torch.ones_like(tensor)
    tensor = torch.min(torch.stack([tensor, ones]), 0)[0]
    zeros = torch.zeros_like(tensor)
    tensor = torch.max(torch.stack([tensor, zeros]), 0)[0]
    if not to_pil:
        return tensor
    return transforms.functional.to_pil_image(tensor)


def show_animation(
    xs: List[Tensor], 
    interval=300, 
    repeat_delay=5000
):
    fig = plt.figure()
    plt.axis('off')
    imgs = []
    for x_t in xs:
        im = plt.imshow(x_t, animated=True)
        imgs.append([im])
    ani = animation.ArtistAnimation(fig, imgs, interval=interval, repeat_delay=repeat_delay, blit=True)
    plt.close(fig)  # Prevent static image from showing
    return HTML(ani.to_jshtml())  # Or use ani.to_html5_video()


def save_animation(
    xs, 
    save_path, 
    interval=300, 
    repeat_delay=5000,
    dpi: int = 200
):
    fig = plt.figure(dpi=dpi)
    plt.axis('off')
    imgs = []
    for x_t in xs:
        im = plt.imshow(x_t, animated=True)
        imgs.append([im])
    animate = animation.ArtistAnimation(fig, imgs, interval=interval, repeat_delay=repeat_delay)
    animate.save(save_path)


def show_tensor_image(
    image: Tensor,
):
    plt.imshow(image)


def plot_generated_images(
    results,
    figsize: int = 8
):
    plt.figure(figsize=_pair(figsize))
    nrows = 1
    ncols = len(results)
    for i, img in enumerate(results):
        ax = plt.subplot(nrows, ncols, i+1)
        ax.set_axis_off()
        show_tensor_image(img)
    plt.show()