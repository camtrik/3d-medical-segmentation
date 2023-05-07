import torch 
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BraTsDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        npimage = np.load(img_path)
        npmask = np.load(mask_path)
        # npimage = npimage.transpose((3, 0, 1, 2))
        # npmask = npmask.transpose((3, 0, 1, 2))
        # npmask = npmask.astype("float32")
        # npimage = npimage.astype("float32")
        
        # transform npimage and npmask to tensor
        npmask = npmask.astype("float32")
        npimage = npimage.astype("float32")
        return npimage,npmask
