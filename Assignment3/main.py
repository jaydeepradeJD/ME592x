from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from train import train_model
from data import LeafImages
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='train', help='train/test mode')
parser.add_argument("--gpu", type=str, default='0', help='GPU option')
parser.add_argument("--batch_size", type=int, default=64, help='batch size')
parser.add_argument("--epochs", type=int, default=50, help='num epochs')
parser.add_argument("--lr", type=float, default=1e-3, help='learing rate')
parser.add_argument("--data_path", type=str, default='./meta_data.txt', help='data path')
parser.add_argument("--save_path", type=str, default='runs', help='save path')
parser.add_argument("--bbox_size", type=int, default=224, help='save path')
args = parser.parse_args()


projectPath     = '/data/Jaydeep/ME592x/Assignment3'
DataDir         = os.path.join(projectPath, args.data_path)

if not os.path.exists(os.path.join(projectPath, 'runs')):
	os.makedirs(os.path.join(projectPath, 'runs'))

ModelSaveDir    = os.path.join(projectPath, 'runs', args.save_path)
if not os.path.exists(ModelSaveDir):
	os.makedirs(ModelSaveDir)

with open(os.path.join(ModelSaveDir,'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

BatchSize   =   args.batch_size
learingRate =   args.lr     ## default = 0.001
BBSize      =   args.bbox_size
Num_epochs  =   args.epochs
device 		= 	args.gpu
device      =   torch.device('cuda:%s'%args.gpu if torch.cuda.is_available() else "cpu")
num_Classes =   7

# model_conv  = torchvision.models.resnet18(pretrained=True)
model_conv  = torchvision.models.resnet50(pretrained=True)
# for param in model_conv.parameters():
#     param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, num_Classes)

model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()
# Observe that only parameters of final layer are being optimized as opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=args.lr, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=20, gamma=0.1)

trainSet = LeafImages(DataDir, mode='train', augment=True, bbox_size=BBSize)
valSet = LeafImages(DataDir, mode='val', augment=False, bbox_size=BBSize)

datasets = {'train':trainSet, 'val':valSet}
dataset_sizes = {'train':trainSet.__len__(), 'val':valSet.__len__()}

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}


model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,  dataloaders, dataset_sizes, device, num_epochs=args.epochs)

torch.save(model_conv.state_dict(), os.path.join(ModelSaveDir, 'best.pt'))