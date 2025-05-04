import cv2
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor,
)

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


class GRANDTOUR(Dataset):
    def __init__(self, filelist_path, mode, device, parent_data_dir=''):
        
        self.mode = mode
        self.parent_data_dir = parent_data_dir
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()

        precision: torch.dtype = torch.float32
        self.transform = Compose(
            [
                ToTensor(),
                Lambda(lambda x: x.to(device)),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ConvertImageDtype(precision),
            ]
        )
    
    def __getitem__(self, item):
        img_path = os.path.join(self.parent_data_dir, self.filelist[item].split(' ')[0])
        depth_path = os.path.join(self.parent_data_dir, self.filelist[item].split(' ')[1])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 1000.0  # cm to m
        if depth.ndim == 3:
            depth = depth[:, :, 0] 
        depth = depth.squeeze()
        print("gt dpeth min max", np.min(depth), np.max(depth))


        
        sample = {'image': image, 'depth': depth}

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        print(f"sample['depth'] shape: {sample['depth'].shape}")
        sample['valid_mask'] = (sample['depth'] <= 80) & (sample['depth'] > 0)
        
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample

    def __len__(self):
        return len(self.filelist)