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
from resnet import *
from torchvision import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--base_dir', default='/home/dthiagar/datasets/',
                    help='base_dir')
parser.add_argument('--datapath', default='BreaKHis_v1/histology_slides/breast/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=3000,
                    help='number of epochs to train')
parser.add_argument('--checkpoints', default='models/BreaKHis_v1/',
                    help='save model')
parser.add_argument('--load', default=1,
                    help='load model from end of that epoch')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()

resnet_type = resnet18
print("ResNet Type: %s" % resnet_type.__name__)

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

images, all_params, labels = lister.dataloader(args.base_dir + args.datapath)


transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.7879, 0.6272, 0.7653),
                             (0.1215, 0.1721, 0.1058)),
])
TrainImgLoader = torch.utils.data.DataLoader(
    DA.ImageFolder(images, all_params, labels, True, transform=transform),
    batch_size=32, shuffle= True, num_workers= 0, drop_last=False)

feature_extractor = ResNetFeatureExtractor(resnet_type).cuda()
num_features = feature_extractor.out_dim
model = DKLModel(feature_extractor, num_dim=num_features).cuda()
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=num_features, n_classes=2).cuda()

state_file = args.base_dir + args.checkpoints + 'dkl_breakhis_%s_checkpoint_%d.dat' % (resnet_type.__name__, args.load)
state = torch.load(state_file)
model.load_state_dict(state['model'])
likelihood.load_state_dict(state['likelihood'])

def analyze():
    model.eval()
    likelihood.eval()
    correct, tp, fp, tn, fn = 0, 0, 0, 0, 0
    for image, params, label in tqdm(TrainImgLoader):
        if args.cuda:
            image, label = image.cuda(), label.cuda()
        with torch.no_grad():
            distr = model(image)
            output = likelihood(distr)
            pred = output.probs.argmax(1)
            label_comp = label.view_as(pred)
            tp += (((pred == 1) + (label_comp == 1)) == 2).sum()
            fp += (((pred == 1) + (label_comp == 0)) == 2).sum()
            tn += (((pred == 0) + (label_comp == 0)) == 2).sum()
            fn += (((pred == 0) + (label_comp == 1)) == 2).sum()
            correct += pred.eq(label.view_as(pred)).cpu().sum()
    print('Accuracy: {}/{} ({}%)'.format(correct, len(TrainImgLoader.dataset), 100. * correct / float(len(TrainImgLoader.dataset))))
    print('Sensitivity Values: TP {}, FP {}, TN {}, FN {}'.format(tp, fp, tn, fn))

with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(0):
    analyze()
