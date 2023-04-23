import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import kornia as K
import kornia.augmentation as KA


class DataPreprocess(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, x: np.ndarray) -> Tensor:
        """Transform from np.ndarray to PyTorch Tensors.

        Args:
            x (np.ndarray): np.uin8 grayscale array of shape BxHxW.

        Returns:
            Tensor: Unnormalized float tensor with values in [0, 1] and shape Bx3xHxW,
            where the grayscale image is repeated across the RGB channels.
        """
        images: np.ndarray = np.tile(x[..., None], (1, 1, 1, 3))
        tensors: Tensor = K.image_to_tensor(images, keepdim=False).float()
        tensors /= 255.
        return tensors


class DataAugmentation(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        # List of possible augmentations to be applied
        self.transforms = KA.ImageSequential(
            KA.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            KA.RandomBrightness(brightness=(0.85, 1.15), p=0.5),
            KA.RandomRotation(degrees=25., p=0.5),
            KA.RandomHorizontalFlip(),
            KA.RandomVerticalFlip(),
        )

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return self.transforms.forward(x)

