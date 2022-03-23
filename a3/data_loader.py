# CMU 16-726 Learning-Based Image Synthesis / Spring 2022, Assignment 3
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip

import glob
import os
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


class CustomDataSet(Dataset):
    """Load images under folders"""

    def __init__(self, main_dir, ext='*.png', transform=None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = glob.glob(os.path.join(main_dir, ext))
        self.total_imgs = all_imgs
        print(os.path.join(main_dir, ext))
        print(len(self))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, 0.


def get_data_loader(data_path, opts):
    """Creates training and test data loaders.
    """
    basic_transform = transforms.Compose([
        transforms.Resize(opts.image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if opts.data_preprocess == 'basic':
        train_transform = basic_transform
    elif opts.data_preprocess == 'deluxe':
        load_size = int(1.1 * opts.image_size)
        osize = [load_size, load_size]
        deluxe_transform = transforms.Compose([
            transforms.Resize(osize, Image.BICUBIC),
            transforms.RandomCrop(opts.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # transforms.RandomAutocontrast(0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_transform = deluxe_transform

    dataset = CustomDataSet(os.path.join('data/', data_path), opts.ext, train_transform)
    dloader = DataLoader(dataset=dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    # display augmented images
    # plot_image(dloader)
    return dloader


def plot_image(dloader):
    images, labels = iter(dloader).next()
    length = len(images)
    f, subplot = plt.subplots(length, 1)
    index = 0
    for image in images:
        image = image.permute(1, 2, 0)
        plt.figure()
        subplot[index].imshow(image)
        index += 1
