import matplotlib.animation as animation
import matplotlib.pyplot as plt
# from IPython.display import Image
import torch
from torch import Tensor
from torch.nn.modules.utils import _pair
import torchvision.transforms as transforms



def save_animation(
    xs, 
    gif_name, 
    interval=300, 
    repeat_delay=5000
):
    fig = plt.figure()
    plt.axis('off')
    imgs = []

    for x_t in xs:
        im = plt.imshow(x_t, animated=True)
        imgs.append([im])

    animate = animation.ArtistAnimation(fig, imgs, interval=interval, repeat_delay=repeat_delay)
    animate.save(gif_name)

def show_tensor_image(
    image: Tensor,
):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: torch.minimum(torch.tensor([1]), t)),
        transforms.Lambda(lambda t: torch.maximum(torch.tensor([0]), t)),
        transforms.ToPILImage(),
    ])
    plt.imshow(reverse_transforms(torch.sigmoid(image[0]).detach().cpu()))

def to_image(tensor, to_pil=True):
    tensor = (tensor + 1) / 2
    ones = torch.ones_like(tensor)
    tensor = torch.min(torch.stack([tensor, ones]), 0)[0]
    zeros = torch.zeros_like(tensor)
    tensor = torch.max(torch.stack([tensor, zeros]), 0)[0]
    if not to_pil:
        return tensor
    return transforms.functional.to_pil_image(tensor)


def plot_generated_images(
    results,
    figsize: int = 8
):
    plt.figure(figsize=_pair(figsize))
    nrows = 1
    ncols = len(results)
  
    for i, (title, img) in enumerate(results.items()):
        ax = plt.subplot(nrows, ncols, i+1)
        ax.set_title(title)
        ax.set_axis_off()
        show_tensor_image(img)
    plt.show()