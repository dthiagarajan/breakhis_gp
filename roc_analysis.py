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
import matplotlib.pyplot as plt
from dataloader import BreaKHis_v1Lister as lister
from dataloader import BreaKHis_v1Loader as DA
from model import *
from resnet import *
from sklearn.metrics import *
from torchvision import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--base_dir', default='/home/dthiagar/datasets/',
                    help='base_dir')
parser.add_argument('--datapath', default='BreaKHis_v1/histology_slides/breast/',
                    help='datapath')
parser.add_argument('--checkpoints', default='models/BreaKHis_v1/',
                    help='save model')
parser.add_argument('--load', type=int, default=None,
                    help='load model from end of that epoch')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size')
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
    transforms.RandomRotation(90),
    transforms.RandomHorizontalFlip(0.8),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

TrainImgLoader = torch.utils.data.DataLoader(
    DA.ImageFolder(images, all_params, labels, True, transform=transform),
    batch_size=args.batch_size, shuffle= True, num_workers= 0, drop_last=False)

feature_extractor = ResNetFeatureExtractor(resnet_type)
num_features = feature_extractor.out_dim
model = DKLModel(feature_extractor, num_dim=num_features)
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=num_features, n_classes=2)
if args.cuda:
    feature_extractor = ResNetFeatureExtractor(resnet_type).cuda()
    model = DKLModel(feature_extractor, num_dim=num_features).cuda()
    likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=num_features, n_classes=2).cuda()


state_file = args.base_dir + args.checkpoints + 'dkl_breakhis_%s_checkpoint_%d.dat' % (resnet_type.__name__, args.load)
if args.cuda:
    print("Using CUDA")
    state = torch.load(state_file)
else:
    state = torch.load(state_file, map_location='cpu')
model.load_state_dict(state['model'])
likelihood.load_state_dict(state['likelihood'])

def analyze():
    model.eval()
    likelihood.eval()
    correct, tp, fp, tn, fn = 0, 0, 0, 0, 0
    preds = torch.zeros(len(TrainImgLoader.dataset))
    labels = torch.zeros(len(TrainImgLoader.dataset))
    for i, (image, params, label) in tqdm(enumerate(TrainImgLoader), total=len(TrainImgLoader)):
        if args.cuda:
            image, label = image.cuda(), label.cuda()
        with torch.no_grad():
            distr = model(image)
            output = likelihood(distr)
            preds[i*args.batch_size:(i+1)*args.batch_size] = output.probs[:, 1]
            labels[i*args.batch_size:(i+1)*args.batch_size] = label
            pred = output.probs.argmax(1)
            label_comp = label.view_as(pred)
            tp += (((pred == 1) + (label_comp == 1)) == 2).sum()
            fp += (((pred == 1) + (label_comp == 0)) == 2).sum()
            tn += (((pred == 0) + (label_comp == 0)) == 2).sum()
            fn += (((pred == 0) + (label_comp == 1)) == 2).sum()
            correct += pred.eq(label.view_as(pred)).cpu().sum()
    print('Accuracy: {}/{} ({}%)'.format(correct, len(TrainImgLoader.dataset), 100. * correct / float(len(TrainImgLoader.dataset))))
    print('Sensitivity Values: TP {}, FP {}, TN {}, FN {}'.format(tp, fp, tn, fn))
    labels, preds = labels.cpu().numpy(), preds.cpu().numpy()
    fpr, tpr, thresholds = roc_curve(labels, preds)
    auc = roc_auc_score(labels, preds)
    return auc, fpr, tpr, thresholds

with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(0):
    auc, fpr, tpr, thresholds = analyze()
    print(auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve: %s, Epoch %d" % (resnet_type.__name__, args.load))
    plt.legend(loc="lower right")
    plt.savefig("dkl_breakhis_%s_%d_roc_curve.png" % (resnet_type.__name__, args.load))

