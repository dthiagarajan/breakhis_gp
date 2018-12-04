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
parser.add_argument('--resnet', type=int, default=18,
                    help='resnet model to use')
parser.add_argument('--datapath', default='BreaKHis_v1/histology_slides/breast/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=3000,
                    help='number of epochs to train')
parser.add_argument('--split', type=float, default=0.15,
                    help='percentage of data to be used for evaluation')
parser.add_argument('--loadmodel', type=int, default=None,
                    help='load model')
parser.add_argument('--checkpoints', default='models/BreaKHis_v1/',
                    help='save model')
parser.add_argument('--eval_train', type=bool, default=True,
                    help='evaluate train data every epoch')
parser.add_argument('--eval_test', type=bool, default=False,
                    help='evaluate test data every epoch')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()

resnet_mapping = {18: resnet18, 50: resnet50, 101: resnet101, 152: resnet152}

resnet_type = resnet_mapping[args.resnet]
logger.info("ResNet Type: %s" % resnet_type.__name__)

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

images, all_params, labels = lister.dataloader(args.base_dir + args.datapath)
benign_images, malignant_images = images
benign_params, malignant_params = all_params
benign_labels, malignant_labels = labels

logger.info("Using %.2f of data for evaluation" % args.split)
num_benign_total, num_malignant_total = len(benign_images), len(malignant_images)
benign_indices, malignant_indices = list(range(num_benign_total)), list(range(num_malignant_total))
benign_split, malignant_split = int(np.floor(args.split * num_benign_total)), int(np.floor(args.split * num_malignant_total))

np.random.seed(3)
np.random.shuffle(benign_indices)
np.random.shuffle(malignant_indices)

train_idx, test_idx = benign_indices[benign_split:], benign_indices[:benign_split]
malignant_train_idx, malignant_test_idx = [num_benign_total + i for i in malignant_indices[malignant_split:]], [num_benign_total + i for i in malignant_indices[:malignant_split]]
train_idx.extend(malignant_train_idx)
test_idx.extend(malignant_test_idx)

images[0].extend(images[1])
all_params[0].extend(all_params[1])
labels[0].extend(labels[1])
images, all_params, labels = images[0], all_params[0], labels[0]

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

test_transform = transforms.Compose([
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
                   False, transform=test_transform),
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
    logger.info("Finding model from at least epoch %s; if not that, closest to it" % args.loadmodel)
    checkpoint_dir = args.base_dir + args.checkpoints
    max_diff = 0
    max_epoch = args.loadmodel
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.dat'):
            l = file[:-4].split('_')
            rn_type, epoch = l[2], int(l[-1])
            diff = epoch - args.loadmodel
            if (rn_type == resnet_type.__name__) and (diff > max_diff):
                max_diff = diff
                max_epoch = epoch
    logger.info("Loading from model at epoch %s" % max_epoch)
    state_split_file = args.base_dir + args.checkpoints + 'dkl_breakhis_%s_checkpoint_%d_%d.dat' % (resnet_type.__name__, int(args.split * 100), max_epoch)
    state_file = args.base_dir + args.checkpoints + 'dkl_breakhis_%s_checkpoint_%d.dat' % (resnet_type.__name__, max_epoch)
    if os.path.isfile(state_split_file):
        logger.info("Loading from file with split in name")
        state_file = state_split_file
    assert os.path.isfile(state_file)

    state = torch.load(state_file)
    model.load_state_dict(state['model'])
    likelihood.load_state_dict(state['likelihood'])
    if 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    completed_epochs = max_epoch

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


def test(train=True, test=True):
    model.eval()
    likelihood.eval()
    train_correct = 0
    if train:
        for image, params, label in tqdm(TrainImgLoader):
            if args.cuda:
                image, label = image.cuda(), label.cuda()
            with torch.no_grad():
                distr = model(image)
                output = likelihood(distr)
                pred = output.probs.argmax(1)
                train_correct += pred.eq(label.view_as(pred)).cpu().sum()
        logger.info('Train Accuracy: {}/{} ({}%)'.format(train_correct, len(TrainImgLoader.dataset), 100. * train_correct / float(len(TrainImgLoader.dataset))))
    correct = 0
    if test:
        for image, params, label in tqdm(TestImgLoader):
            if args.cuda:
                image, label = image.cuda(), label.cuda()
            with torch.no_grad():
                distr = model(image)
                output = likelihood(distr)
                pred = output.probs.argmax(1)
                correct += pred.eq(label.view_as(pred)).cpu().sum()
        logger.info('Test_Accuracy: {}/{} ({}%)'.format(correct, len(TestImgLoader.dataset), 100. * correct / float(len(TestImgLoader.dataset))))


for epoch in range(1, args.epochs - completed_epochs + 1):
    true_epoch = epoch + completed_epochs
    with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(0):
        loss = train(epoch)
        test(train=args.eval_train, test=args.eval_test)
        scheduler.step(loss)
    if true_epoch % 25 == 0:
        state_dict = model.state_dict()
        likelihood_state_dict = likelihood.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        torch.save({'model': state_dict, 'likelihood': likelihood_state_dict, 'optimizer': optimizer_state_dict},
                args.base_dir + args.checkpoints + 'dkl_breakhis_%s_checkpoint_%d_%d.dat' % (resnet_type.__name__, int(args.split * 100), epoch + completed_epochs))
