import argparse
import Augmentor
import logging
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
from resnet import *
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--base_dir', default='/home/dthiagar/datasets/',
                    help='base_dir')
parser.add_argument('--datapath', default='BreaKHis_v1/histology_slides/breast/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=3000,
                    help='number of epochs to train')
parser.add_argument('--checkpoints', default='models/',
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

images, all_params, labels = lister.dataloader(args.base_dir + args.datapath)

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


transform = transforms.Compose([
        p.torch_transform(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.7879, 0.6272, 0.7653),
                             (0.1215, 0.1721, 0.1058)),
])

TrainImgLoader = torch.utils.data.DataLoader(
    DA.ImageFolder(train_images, train_params, train_labels, True, transform=transform),
    batch_size= 16, shuffle= True, num_workers= 4, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.ImageFolder(test_images, test_params, test_labels, False, transform=transform),
    batch_size=1, shuffle=True, num_workers=1, drop_last=False)

feature_extractor = ResNetFeatureExtractor(resnet50).cuda()
num_features = feature_extractor.out_dim
model = DKLModel(feature_extractor, num_dim=num_features).cuda()

if args.cuda:
    logger.info("Using CUDA")
    model.cuda()

lr = 0.001
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=num_features, n_classes=2).cuda()
optimizer = optim.RMSprop([
    # {'params': model.feature_extractor.parameters()},
    {'params': model.gp_layer.hyperparameters()},
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr, weight_decay=0.9, centered=True)
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
        logger.info('Train Epoch: %d [%03d/%03d], Loss: %.6f, Time: %.3f' % (epoch, batch_idx + 1, len(TrainImgLoader), loss.item(), time.time() - start_time))

def test():
    model.eval()
    likelihood.eval()

    correct = 0  
    for image, params, label in tqdm(TestImgLoader):
        if args.cuda:
            image, label = image.cuda(), label.cuda()    
        with torch.no_grad():
            distr = model(image)
            output = likelihood(distr)
            pred = output.probs.argmax(1)
            correct += pred.eq(label.view_as(pred)).cpu().sum()
    logger.info('Test set: Accuracy: {}/{} ({}%)'.format(
        correct, len(TestImgLoader.dataset), 100. * correct / float(len(TestImgLoader.dataset))
    ))

for epoch in range(1, args.epochs + 1):
    scheduler.step()
    with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(0):
        train(epoch)
        test()
    state_dict = model.state_dict()
    likelihood_state_dict = likelihood.state_dict()
    torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, args.base_dir + args.checkpoints + 'dkl_breakhis_checkpoint_%d.dat' % epoch)
