import argparse
import Augmentor
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import BreaKHis_v1Lister as lister
from dataloader import BreaKHis_v1Loader as DA
from model import *
from torchvision.models import *
from torchvision import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--datapath', default='/home/dthiagar/datasets/BreaKHis_v1/histology_slides/breast/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=3000,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

images, all_params, labels = lister.dataloader(args.datapath)

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.7879, 0.6272, 0.7653),
                             (0.1215, 0.1721, 0.1058)),
])
TrainImgLoader = torch.utils.data.DataLoader(
    DA.ImageFolder(images, all_params, labels, True, transform=transform),
    batch_size= 16, shuffle= True, num_workers= 0, drop_last=False)

mean, var  = torch.zeros(3), torch.zeros(3)
if args.cuda:
    mean, var = mean.cuda(), var.cuda()
for image, _, _ in tqdm(TrainImgLoader):
    if args.cuda:
        image = image.cuda()
    b, c, h, w = image.shape
    image = image.permute((1, 0, 2, 3)).contiguous()
    mean += 16 * image.view(c, -1).mean(1)
    var += 16 * image.view(c, -1).var(1)
mean /= len(images)
var /= len(images)
var = torch.sqrt(var)
print(mean)
print(var)
