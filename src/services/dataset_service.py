from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, split_ids=None, transform=None, ignore_index=255, unlabeled_value=0):
        self.split_ids = split_ids  # list of tile IDs for this split             

        all_image_folders = sorted(list(Path(images_dir).glob("*")))
        all_mask_folders = sorted(list(Path(masks_dir).glob("*")))

        # keep only folders in this split
        if split_ids is not None:
            self.mask_ids = [f"mask_{tile_id}" for tile_id in split_ids]
            self.image_folders = [f for f in all_image_folders if f.name in split_ids]
            self.mask_folders = [f for f in all_mask_folders if f.name in self.mask_ids]
        else:
            self.image_folders = all_image_folders
            self.mask_folders = all_mask_folders

        self.image_files = self.__get_tifs(type="image")
        self.mask_files = self.__get_tifs(type="mask")
        self.transform = transform
        self.ignore_index = ignore_index
        self.unlabeled_value = unlabeled_value

    def __get_tifs(self, type="image"):
        list_tifs = []
        if type == "image":
            for folder in self.image_folders:
                list_tifs.extend(sorted(list(folder.glob("*.tif"))))
        elif type == "mask":
            for folder in self.mask_folders:
                list_tifs.extend(sorted(list(folder.glob("*.tif"))))
        return list_tifs

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        mask = Image.open(self.mask_files[idx])

        img = np.array(img) / 255.0
        mask = np.array(mask).astype(np.int64)

        # replace unlabeled pixels with ignore_index
        mask[mask == self.unlabeled_value] = self.ignore_index

        img = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).long()

        return img, mask
