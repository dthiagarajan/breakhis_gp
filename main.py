import argparse
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
from torchvision import transforms

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--datapath', default='/scratch/datasets/BreaKHis_v1/histology_slides/breast/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
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
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
])
''' Need to split data into train, val, test.'''
TrainImgLoader = torch.utils.data.DataLoader(
    DA.ImageFolder(images, all_params, labels, True, transform=transform),
    batch_size= 8, shuffle= True, num_workers= 0, drop_last=False)


model = PreActResNet18(num_classes=2)
if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(list(model.parameters()), lr=0.001,
                      momentum=0.9, weight_decay=5e-4, nesterov=True)
model.train()
for epoch in range(args.epochs):
    total_train_loss = 0.
    for batch_idx, (image, params, label) in enumerate(TrainImgLoader):
        optimizer.zero_grad()
        start_time = time.time()
        image = Variable(torch.FloatTensor(image))
        label = Variable(torch.LongTensor(label))
        if args.cuda:
            image, label = image.cuda(), label.cuda()
        pred = model(image)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time), flush=True)
        total_train_loss += float(loss)
    print('Epoch %d total training loss = %.3f' % (epoch, total_train_loss/len(TrainImgLoader)))
