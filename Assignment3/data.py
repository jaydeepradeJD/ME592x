import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image

class LeafImages(data.Dataset):
	def __init__(self, data_path='meta_data.txt', mode='train', augment=False, bbox_size=(500,500), **kwargs):
		self.data_path = data_path
		self.augment = augment
		self.mode = mode
		self.input_size = (224, 224)
		self.bbox_size = bbox_size
		self.image_paths, self.labels = [], []
		with open(self.data_path, 'r') as f:
			lines = f.readlines()
		for line in lines:
			self.image_paths.append(line.split()[0])
			self.labels.append(line.split()[1])
		self.ids = np.arange(len(self.labels))

		self.split = kwargs.get('split', 0.8)
		if self.mode == 'train':
			self.ids = self.ids[:int(self.split*len(self.ids))]
		elif self.mode == 'val' or self.mode == 'test':
			self.ids = self.ids[int(self.split*len(self.ids)):]	
		
	def __len__(self):
		return len(self.ids)


	
	def rotate(self, input_size, bbox_size):
		transform = transforms.Compose(
			    [   #transforms.ToPILImage(),
			    	transforms.CenterCrop(bbox_size),
			        transforms.RandomRotation(180),
			        torchvision.transforms.Resize(size=input_size),
			        transforms.ToTensor(),
			    ])
		return transform

	def hrflip(self, input_size, bbox_size):
		transform = transforms.Compose(
			    [   #transforms.ToPILImage(),
			        transforms.CenterCrop(bbox_size),
			        transforms.RandomHorizontalFlip(p=1.0),
			        torchvision.transforms.Resize(size=input_size),
			        transforms.ToTensor(),
			    ])
		return transform

	def vrflip(self, input_size, bbox_size):
		transform = transforms.Compose(
			    [   #transforms.ToPILImage(),
			        transforms.CenterCrop(bbox_size),
			        transforms.RandomVerticalFlip(p=1.0),
			        torchvision.transforms.Resize(size=input_size),
			        transforms.ToTensor(),
			    ])
		return transform

	def shear(self, input_size, bbox_size):
		transform = transforms.Compose(
			    [   #transforms.ToPILImage(),
			    	transforms.CenterCrop(bbox_size),
			        torchvision.transforms.RandomAffine(0, shear=(-45,45,-45,45)),
			        torchvision.transforms.Resize(size=input_size),
			        transforms.ToTensor(),
			    ])
		return transform

	def translate(self, input_size, bbox_size):
		transform = transforms.Compose(
			    [   #transforms.ToPILImage(),
			        torchvision.transforms.RandomAffine(0, translate=(0.5, 0.5)),
			        transforms.CenterCrop(bbox_size),
			        torchvision.transforms.Resize(size=input_size),
			        transforms.ToTensor(),
			    ])
		return transform

	def scale(self, input_size, bbox_size):
		transform = transforms.Compose(
			    [   #transforms.ToPILImage(),
			        torchvision.transforms.RandomAffine(0, scale=(0.1, 2.0)),
			        transforms.CenterCrop(bbox_size),
			        torchvision.transforms.Resize(size=input_size),
			        transforms.ToTensor(),
			    ])
		return transform


	def jitter(self, input_size, bbox_size):
		transform = transforms.Compose(
			    [   #transforms.ToPILImage(),
			        transforms.CenterCrop(bbox_size),
			        torchvision.transforms.ColorJitter(brightness=(0.0, 1.0), contrast=(0.0, 1.0), saturation=(0.0, 1.0), hue=(-0.5, 0.5)),
			        torchvision.transforms.Resize(size=input_size),
			        transforms.ToTensor(),
			    ])
		return transform

	def crop_bbox(self, input_size):
		transform = transforms.Compose(
			    [   
			        torchvision.transforms.Resize(size=input_size),
			        transforms.ToTensor(),
			    ])
		return transform


	def __getitem__(self, index):
		self.action_dict = {1:self.rotate, 2:self.hrflip, 3:self.vrflip, 4:self.shear, 5:self.translate,
							6:self.scale, 7:self.jitter}
		action = np.random.randint(1, 8)

		img = Image.open(self.image_paths[self.ids[index]])
		w, h = img.size
		if self.bbox_size>h or self.bbox_size>w:
			self.bbox_size = min(h, w)

		if self.augment:
			img = self.action_dict[action](self.input_size, self.bbox_size)(img)
		else:
			img = self.crop_bbox(self.input_size)(img)

		mean = img.view(img.size(0), -1).mean(1)
		std = img.view(img.size(0), -1).std(1) + 1e-08
		img = transforms.Normalize(mean, std)(img)
		label = int(self.labels[self.ids[index]])
		
		return img, label