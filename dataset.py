import torch
from torch.utils.data import Dataset
import numpy as np
import imageio
from kornia import create_meshgrid
from einops import rearrange
from typing import Tuple


class ImageDataset(Dataset):
    def __init__(self, image_path: str, split: str):
        """
        split: 'train' or 'val'
        """
        image = imageio.imread(image_path)[..., :3]/255.
        c = [image.shape[0]//2, image.shape[1]//2]
        self.r = 256
        image = image[c[0]-self.r:c[0]+self.r,
                      c[1]-self.r:c[1]+self.r] # (512, 512, 3)

        self.uv = create_meshgrid(2*self.r, 2*self.r, True)[0] # (512, 512, 2)
        self.rgb = torch.FloatTensor(image) # (512, 512, 3)

        if split == 'train':
            self.uv = self.uv[::2, ::2] # (256, 256, 2)
            self.rgb = self.rgb[::2, ::2] # (256, 256, 3)

        self.uv = rearrange(self.uv, 'h w c -> (h w) c')
        self.rgb = rearrange(self.rgb, 'h w c -> (h w) c')

    def __len__(self):
        return len(self.uv)

    def __getitem__(self, idx: int):
        return {"uv": self.uv[idx], "rgb": self.rgb[idx]}