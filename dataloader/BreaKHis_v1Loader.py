import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(open(path, 'rb')).convert('RGB')


class ImageFolder(data.Dataset):
    def __init__(self, images, params, labels, training, loader=default_loader, transform=None):
        self.images = images
        self.params = params
        self.labels = labels
        self.training = training
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        # Pad images so they're all the same size (there's only 2-3 I think that actually need this).
        image = self.loader(self.images[index])
        if self.transform:
            image = self.transform(image)
        params = self.params[index]
        label = 0 if self.labels[index] == 'benign' else 1

        if self.training:
            return image, params, label
        else:
            return image, params

    def __len__(self):
        return len(self.images)
