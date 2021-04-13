import os
import numpy as np

path = os.getcwd() #'./Leaf_Images'
classes =  os.listdir(os.path.join(path, 'Data', 'Leaf_Images'))
img_paths = []
labels = []

for cls in classes:
	img_path = os.path.join(path, 'Data', 'Leaf_Images', cls, 'output')
	images = os.listdir(img_path)
	for img in images:
		img_paths.append(os.path.abspath(os.path.join(img_path, img)))
		if int(cls)==1:
			labels.append(str(int(cls)-1))
		else:
			labels.append(str(int(cls)-2))


with open('meta_data.txt', 'w') as f:
	for img_path, label in zip(img_paths, labels):
		f.write(f'{img_path}' + ' ' +f'{label}\n')


# image_paths, labels = [], []
# with open('meta_data.txt', 'r') as f:
# 	lines = f.readlines()

# for line in lines:
# 	image_paths.append(line.split()[0])
# 	labels.append(line.split()[1])

# print(image_paths[0])
# print(labels[0])
