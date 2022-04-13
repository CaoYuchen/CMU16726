import glob
import os

import torch
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets
from torchvision import transforms


class CustomDataSet(Dataset):
    def __init__(self, main_dir, resolution, load_alpha):
        self.main_dir = main_dir
        self.resolution = (resolution, resolution)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.load_alpha = load_alpha
        self.total_imgs =  glob.glob(main_dir)
        self.total_imgs = sorted(self.total_imgs)
        print(len(self))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc)
        image = image.resize(self.resolution)

        rgb = image.convert("RGB")
        tensor_image = self.transform(rgb)

        if self.load_alpha:
            mask = image.split()[-1]
            mask = self.mask_transform(mask)
            # mask = (mask > 0.5).float()
        else:
            mask = 0.
        return tensor_image, mask


def get_data_loader(data_path, resolution, alpha=False, is_train=False):
    """Creates training and test data loaders.
    """
    dataset = CustomDataSet(data_path, resolution, alpha)
    dloader = DataLoader(dataset=dataset, batch_size=1, shuffle=is_train, drop_last=is_train,
                         num_workers=10)

    return dloader
