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

num_total = len(images)
indices = list(range(num_total))
split = int(np.floor(0.15 * num_total))

np.random.seed(3)
np.random.shuffle(indices)

train_idx, test_idx = indices[split:], indices[:split]
train_images, train_params, train_labels = [images[i] for i in train_idx], [all_params[i] for i in train_idx], [labels[i] for i in train_idx]
test_images, test_params, test_labels = [images[i] for i in test_idx], [all_params[i] for i in test_idx], [labels[i] for i in test_idx]

p = Augmentor.Pipeline()
p.rotate90(probability=1)
p.rotate270(probability=1)
p.flip_top_bottom(probability=0.8)
p.crop_random(probability=1, percentage_area=0.5)
p.resize(probability=1.0, width=120, height=120)


transform = transforms.Compose([
        p.torch_transform(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
])
print(len(train_images))
print(len(test_images))

TrainImgLoader = torch.utils.data.DataLoader(
    DA.ImageFolder(train_images, train_params, train_labels, True, transform=transform),
    batch_size= 16, shuffle= True, num_workers= 0, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.ImageFolder(test_images, test_params, test_labels, False, transform=transform),
    batch_size=1, shuffle=True, num_workers=0, drop_last=False)

feature_extractor = ResNetFeatureExtractor(resnet50).cuda()
num_features = 1000 # from ImageNet
model = DKLModel(feature_extractor, num_dim=num_features).cuda()

if args.cuda:
    model.cuda()

lr = 0.001
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, n_classes=2).cuda()
optimizer = optim.RMSprop([
    {'params': model.feature_extractor.parameters()},
    {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.1},
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr, weight_decay=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.75 * args.epochs], gamma=0.1)

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
        print('Train Epoch: %d [%03d/%03d], Loss: %.6f' % (epoch, batch_idx + 1, len(TrainImgLoader), loss.item()), flush=True)

def test():
    model.eval()
    likelihood.eval()

    correct = 0  
    for image, params, label in TestImgLoader:
        if args.cuda:
            image, label = image.cuda(), label.cuda()    
        with torch.no_grad():
            output = likelihood(model(image))
            pred = output.probs.argmax(1)
            correct += pred.eq(label.view_as(pred)).cpu().sum()
    print('Test set: Accuracy: {}/{} ({}%)'.format(
        correct, len(TestImgLoader.dataset), 100. * correct / float(len(TestImgLoader.dataset))
    ))

for epoch in range(1, args.epochs + 1):
    scheduler.step()
    with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(0):
        train(epoch)
        test()
    state_dict = model.state_dict()
    likelihood_state_dict = likelihood.state_dict()
    torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, 'dkl_cifar_checkpoint.dat')
