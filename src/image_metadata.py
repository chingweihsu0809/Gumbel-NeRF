from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ImageMetadata:
    def __init__(self, image_path: Path, c2w: torch.Tensor, W: int, H: int, intrinsics: torch.Tensor, image_index: int,
                 mask_path: Optional[Path], is_val: bool, crop_img: bool = False, instanceid = -1):
        self.image_path = image_path
        self.c2w = c2w.float()
        self.W = W
        self.H = H
        self.intrinsics = intrinsics
        self.image_index = image_index
        self._mask_path = mask_path
        self.is_val = is_val
        self.instanceid = instanceid

        if self.intrinsics.numel() == 2:
            # for dataset of waymo processed by LargeScaleNeRFPytorch
            intrinsics = torch.zeros([4])
            intrinsics[0] = self.intrinsics[0]
            intrinsics[1] = self.intrinsics[1]
            intrinsics[2] = self.W / 2.0
            intrinsics[3] = self.H / 2.0
            self.intrinsics = intrinsics
        
        self.crop_img = crop_img
        if self.crop_img:
            self.H, self.W = self.H // 2, self.W//2
            self.intrinsics[2] = self.intrinsics[2] // 2
            self.intrinsics[3] = self.intrinsics[3] // 2

    def load_image(self) -> torch.Tensor:
        rgbs = Image.open(self.image_path).convert('RGB')
        
        if self.crop_img:
            lt = rgbs.size[0] / 4
            rb = rgbs.size[0] * 3 / 4
            rgbs = rgbs.crop((lt, lt, rb, rb))
            
        size = rgbs.size

        if size[0] != self.W or size[1] != self.H:
            print("resizing")
            rgbs = rgbs.resize((self.W, self.H), Image.LANCZOS)

        return torch.ByteTensor(np.asarray(rgbs))

    def load_mask(self) -> Optional[torch.Tensor]:
        if self._mask_path is None:
            return None

        with ZipFile(self._mask_path) as zf:
            with zf.open(self._mask_path.name) as f:
                keep_mask = torch.load(f, map_location='cpu')

        if keep_mask.shape[0] != self.H or keep_mask.shape[1] != self.W:
            keep_mask = F.interpolate(keep_mask.unsqueeze(0).unsqueeze(0).float(),
                                      size=(self.H, self.W)).bool().squeeze()

        return keep_mask
