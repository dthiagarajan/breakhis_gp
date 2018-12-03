import argparse
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

parser = argparse.ArgumentParser(description='BreakHis DKL')
parser.add_argument('--base_dir', default='/home/dthiagar/datasets/',
                    help='base_dir')
parser.add_argument('--datapath', default='BreaKHis_v1/histology_slides/breast/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=3000,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', type=int, default=None,
                    help='load model')
parser.add_argument('--checkpoints', default='models/BreaKHis_v1/',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()

resnet_type = resnet101
logger.info("ResNet Type: %s" % resnet_type.__name__)

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
train_images, train_params, train_labels = [images[i] for i in train_idx], [
    all_params[i] for i in train_idx], [labels[i] for i in train_idx]
test_images, test_params, test_labels = [images[i] for i in test_idx], [
    all_params[i] for i in test_idx], [labels[i] for i in test_idx]


transform = transforms.Compose([
    transforms.RandomRotation(90),
    transforms.RandomHorizontalFlip(0.8),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

TrainImgLoader = torch.utils.data.DataLoader(
    DA.ImageFolder(train_images, train_params,
                   train_labels, True, transform=transform),
    batch_size=10, shuffle=True, num_workers=4, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.ImageFolder(test_images, test_params, test_labels,
                   False, transform=transform),
    batch_size=1, shuffle=True, num_workers=1, drop_last=False)

feature_extractor = ResNetFeatureExtractor(resnet_type).cuda()
num_features = feature_extractor.out_dim
model = DKLModel(feature_extractor, num_dim=num_features).cuda()

if args.cuda:
    logger.info("Using CUDA")
    model.cuda()

lr = 0.0001
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
    num_features=num_features, n_classes=2).cuda()
optimizer = optim.RMSprop([
    {'params': model.feature_extractor.parameters(), 'lr': lr * 0.001},
    {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.001},
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', factor=0.9, patience=5, verbose=True)

completed_epochs = 0
if args.loadmodel:
    print("Loading model from %s" % args.loadmodel)
    state_file = args.base_dir + args.checkpoints + 'dkl_breakhis_%s_checkpoint_%d.dat' % (resnet_type.__name__, args.loadmodel)
    state = torch.load(state_file)
    model.load_state_dict(state['model'])
    likelihood.load_state_dict(state['likelihood'])
    completed_epochs = args.loadmodel

def train(epoch):
    model.train()
    likelihood.train()

    mll = gpytorch.mlls.VariationalELBO(
        likelihood, model.gp_layer, num_data=len(TrainImgLoader))
    total_loss = 0.
    for batch_idx, (image, params, label) in enumerate(TrainImgLoader):
        start_time = time.time()
        if args.cuda:
            image, label = image.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(image)
        loss = -mll(output, label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        logger.info('Train Epoch: %d [%03d/%03d], Loss: %.6f, Time: %.3f' % (
            epoch + completed_epochs, batch_idx + 1, len(TrainImgLoader), loss.item(), time.time() - start_time))
    return total_loss


def test():
    model.eval()
    likelihood.eval()
    train_correct = 0
    for image, params, label in tqdm(TrainImgLoader):
        if args.cuda:
            image, label = image.cuda(), label.cuda()
        with torch.no_grad():
            distr = model(image)
            output = likelihood(distr)
            pred = output.probs.argmax(1)
            train_correct += pred.eq(label.view_as(pred)).cpu().sum()
    correct = 0
    for image, params, label in tqdm(TestImgLoader):
        if args.cuda:
            image, label = image.cuda(), label.cuda()
        with torch.no_grad():
            distr = model(image)
            output = likelihood(distr)
            pred = output.probs.argmax(1)
            correct += pred.eq(label.view_as(pred)).cpu().sum()
    logger.info('Train_Accuracy: {}/{} ({}%), Test_Accuracy: {}/{} ({}%)'.format(
        train_correct, len(TrainImgLoader.dataset), 100. *
        train_correct / float(len(TrainImgLoader.dataset)),
        correct, len(TestImgLoader.dataset), 100. * correct /
        float(len(TestImgLoader.dataset))
    ))


for epoch in range(1, args.epochs - completed_epochs + 1):
    with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(0):
        loss = train(epoch)
        test()
        scheduler.step(loss)
    state_dict = model.state_dict()
    likelihood_state_dict = likelihood.state_dict()
    torch.save({'model': state_dict, 'likelihood': likelihood_state_dict},
               args.base_dir + args.checkpoints + 'dkl_breakhis_%s_checkpoint_%d.dat' % (resnet_type.__name__, epoch + completed_epochs))
