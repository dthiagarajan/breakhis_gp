import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import time
import math
from dataloader import BreaKHis_v1Lister as lister
from dataloader import BreaKHis_v1Loader as DA
from matplotlib import pyplot as plt
from model import *
from resnet import *
from sklearn.metrics import *
from torchvision import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--resnet', type=int, default=18,
                    help="resnet model to use")
parser.add_argument('--base_dir', default='/home/dthiagar/datasets/',
                    help='base_dir')
parser.add_argument('--datapath', default='BreaKHis_v1/histology_slides/breast/',
                    help='datapath')
parser.add_argument('--checkpoints', default='models/BreaKHis_v1/',
                    help='save model')
parser.add_argument('--split', type=float, default=0.15,
                    help='percentage of data to be used for evaluation')
parser.add_argument('--load', type=int, default=None,
                    help='load model from end of that epoch')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()

resnet_mapping = {18: resnet18, 50: resnet50, 101: resnet101, 152: resnet152}
resnet_type = resnet_mapping[args.resnet]
print("ResNet Type: %s" % resnet_type.__name__)

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


images, all_params, labels = lister.dataloader(args.base_dir + args.datapath)
benign_images, malignant_images = images
benign_params, malignant_params = all_params
benign_labels, malignant_labels = labels

print("Using %.2f of data for evaluation" % args.split)
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
#    transforms.RandomRotation(90),
#    transforms.RandomHorizontalFlip(0.8),
#    transforms.RandomResizedCrop(224),
#    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

print("Length of DataLoader: %d" % len(test_images))
print(set(test_labels))
TestImgLoader = torch.utils.data.DataLoader(
    DA.ImageFolder(test_images, test_params, test_labels,
                   False, transform=transform),
    batch_size=1, shuffle=False, num_workers=1, drop_last=False)

feature_extractor = ResNetFeatureExtractor(resnet_type)
num_features = feature_extractor.out_dim
model = DKLModel(feature_extractor, num_dim=num_features)
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=num_features, n_classes=2)
if args.cuda:
    feature_extractor = ResNetFeatureExtractor(resnet_type).cuda()
    model = DKLModel(feature_extractor, num_dim=num_features).cuda()
    likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=num_features, n_classes=2).cuda()


state_file = args.base_dir + args.checkpoints + 'dkl_breakhis_%s_checkpoint_%d_%d.dat' % (resnet_type.__name__, int(args.split * 100), args.load)
if args.cuda:
    print("Using CUDA")
    state = torch.load(state_file)
else:
    state = torch.load(state_file, map_location='cpu')
model.load_state_dict(state['model'])
likelihood.load_state_dict(state['likelihood'])

def analyze(data):
    model.eval()
    likelihood.eval()
    correct, tp, fp, tn, fn = 0., 0., 0., 0., 0.
    preds = torch.zeros(len(data.dataset))
    label_preds = torch.zeros(len(data.dataset))
    labels = torch.zeros(len(data.dataset))
    for i, (image, params, label) in enumerate(tqdm(data)):
        if args.cuda:
            image, label = image.cuda(), label.cuda()
        with torch.no_grad():
            distr = model(image)
            output = likelihood(distr)
            pred = output.probs.argmax(1)
            preds[i] = output.probs[:, 1]
            labels[i] = label
            label_preds[i] = pred
            label_comp = label.view_as(pred)
            tp += float((((pred == 1) + (label_comp == 1)) == 2).sum())
            fp += float((((pred == 1) + (label_comp == 0)) == 2).sum())
            tn += float((((pred == 0) + (label_comp == 0)) == 2).sum())
            fn += float((((pred == 0) + (label_comp == 1)) == 2).sum())
            correct += pred.eq(label.view_as(pred)).cpu().sum()
    print('Accuracy: {}/{} ({}%)'.format(correct, len(data.dataset), 100. * correct / float(len(data.dataset))))
    print('Sensitivity Values: TP {}, FP {}, TN {}, FN {}'.format(tp, fp, tn, fn))
    labels, preds = labels.cpu().numpy(), preds.cpu().numpy()
    print('Precision: %0.3f, Recall: %0.3f' % (float(tp*1.0/(tp+fp)), float(tp*1.0/(tp+fn))))
    fpr, tpr, thresholds = roc_curve(labels, preds)
    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    return fpr, tpr, thresholds, auc, ap

with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(0):
    fpr, tpr, thresholds, auc, ap  = analyze(TestImgLoader)
    print("Positive AUC: %0.4f" % auc)
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

