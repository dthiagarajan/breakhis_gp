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
from torchvision.models import *
from torchvision import transforms

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--datapath', default='/scratch/datasets/BreaKHis_v1/histology_slides/breast/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=300,
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

feature_extractor = ResNetFeatureExtractor(resnet18).cuda()
num_features = 1000 # from ImageNet
model = DKLModel(feature_extractor, num_dim=num_features).cuda()

if args.cuda:
    model.cuda()

lr = 0.1
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, n_classes=2).cuda()
optimizer = optim.SGD([
    {'params': model.feature_extractor.parameters()},
    {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * args.epochs, 0.75 * args.epochs], gamma=0.1)

def train(epoch):
    model.train()
    likelihood.train()
    
    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(TrainImgLoader))
    train_loss = 0.
    for batch_idx, (image, params, label) in enumerate(TrainImgLoader):
        start_time = time.time()
        if args.cuda:
            image, label = image.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(image)
        loss = -mll(output, label)
        loss.backward()
        optimizer.step()
        print('Train Epoch: %d [%03d/%03d], Loss: %.6f' % (epoch, batch_idx + 1, len(TrainImgLoader), loss.item()))

for epoch in range(1, args.epochs + 1):
    scheduler.step()
    with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(0):
        train(epoch)
    state_dict = model.state_dict()
    likelihood_state_dict = likelihood.state_dict()
    torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, 'dkl_cifar_checkpoint.dat')
