from __future__ import print_function, division
import torch
import torch.nn as nn
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
from PIL import Image, ImageDraw, ImageFont 
import cv2

def get_leaf(img, xcenter, ycenter):
    xmin = xcenter - 112
    ymin = ycenter - 112
    xmax = xcenter + 112
    ymax = ycenter + 112

    img = img[:, ymin:ymax, xmin:xmax]
    return img, xmin, ymin, xmax, ymax  

device      =   torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

test_data_path = '/data/Jaydeep/ME592x/Assignment3/Data/idcdiseases'
test_images = os.listdir(test_data_path)

key_images = {'1001.jpg', '1002.jpg', '1003.jpg', '1004.jpg', '1005.jpg',
           '1006.jpg', '1007.jpg', '1008.jpg', '1009.jpg', '1010.jpg'}
centers = {}

for key in key_images:
    centers[key] = []

# centers['1001.jpg'] = []
# centers['1002.jpg'] = []
# centers['1003.jpg'] = []
# centers['1004.jpg'] = []
# centers['1005.jpg'] = []
# centers['1006.jpg'] = []
# centers['1007.jpg'] = []
# centers['1008.jpg'] = []
# centers['1009.jpg'] = []
# centers['1010.jpg'] = []


centers['1001.jpg'].append([1200, 1653])
centers['1001.jpg'].append([996, 2445])
centers['1001.jpg'].append([2286, 2343])
centers['1001.jpg'].append([2718, 2904])

centers['1002.jpg'].append([3120, 1728])
centers['1002.jpg'].append([1872, 2256])
centers['1002.jpg'].append([2796, 960])
centers['1002.jpg'].append([2736, 2520])
centers['1002.jpg'].append([1548, 1710])

centers['1003.jpg'].append([2538, 1164])
centers['1003.jpg'].append([1962, 1272])
centers['1003.jpg'].append([2592, 1968])
centers['1003.jpg'].append([3132, 2184])
centers['1003.jpg'].append([3120, 1200])

centers['1004.jpg'].append([2838, 1224])
centers['1004.jpg'].append([2154, 1026])
centers['1004.jpg'].append([2166, 1560])
centers['1004.jpg'].append([2760, 2436])
centers['1004.jpg'].append([3126, 2766])

centers['1005.jpg'].append([1776, 1290])
centers['1005.jpg'].append([2460, 1458])
centers['1005.jpg'].append([1938, 2082])
centers['1005.jpg'].append([3000, 1728])
centers['1005.jpg'].append([2724, 2370])

centers['1006.jpg'].append([3100, 1428])
centers['1006.jpg'].append([2658, 1440])
centers['1006.jpg'].append([2820, 912])
centers['1006.jpg'].append([3120, 1950])
centers['1006.jpg'].append([1986, 1656])

centers['1007.jpg'].append([2136, 1278])
centers['1007.jpg'].append([2262, 1566])
centers['1007.jpg'].append([2742, 1392])
centers['1007.jpg'].append([2670, 2502])
centers['1007.jpg'].append([2946, 2922])

centers['1008.jpg'].append([2610, 1800])
centers['1008.jpg'].append([1900, 2500])
centers['1008.jpg'].append([3486, 2298])
centers['1008.jpg'].append([3690, 1746])
centers['1008.jpg'].append([2292, 1668])

centers['1009.jpg'].append([2586, 708])
centers['1009.jpg'].append([2202, 996])
centers['1009.jpg'].append([4060, 1296])
centers['1009.jpg'].append([3138, 3090])
centers['1009.jpg'].append([1560, 1992])

centers['1010.jpg'].append([2160, 2400])
centers['1010.jpg'].append([2442, 2742])
centers['1010.jpg'].append([2200, 3048])
centers['1010.jpg'].append([1764, 1428])
centers['1010.jpg'].append([3126, 1152])



def annotate(image, centers, classes):
    img = cv2.imread(os.path.join(test_data_path, image))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 5
    thickness = 8
    color = (0, 0, 255)

    for i in range(len(classes)):
        xcenter, ycenter = centers[i][0], centers[i][1]
        cls = classes[i]
        img = cv2.putText(img, str(cls), (xcenter, ycenter), font, 
                       fontScale, color, thickness, cv2.LINE_AA)
    
    cv2.imwrite(image, img)   




weights = torch.load('/data/Jaydeep/ME592x/Assignment3/runs/3/best.pt')

model  = torchvision.models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)
model.load_state_dict(weights)

model = model.to(device)
model.eval()

for image in key_images:
    img = Image.open(os.path.join(test_data_path, image))
    img = transforms.ToTensor()(img)
    classes = []

    for i in range(len(centers[image])):

        xcenter, ycenter = centers[image][i][0], centers[image][i][1]
        print(xcenter, ycenter)
        img_crop, xmin, ymin, xmax, ymax = get_leaf(img, xcenter, ycenter)

        img_crop = torch.reshape(img_crop, (1, 3, 224, 224))
        img_crop = img_crop.to(device)
        outputs = model(img_crop)
        _, preds = torch.max(outputs, 1)
        preds = preds.detach().cpu().numpy()
        print('Output = ', preds)
        classes.append(preds[0])

    annotate(image, centers[image], classes)