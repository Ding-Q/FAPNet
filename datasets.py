#-----------------------------------AVD-v2------------------------------------
import os
import random
import numpy as np
import copy
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torchvision import transforms, utils, datasets, models
from torch.utils.data import DataLoader, Dataset, random_split
from utils import file_filter
# def file_filter(f):
#     if f[-4:] in ['.jpg', '.png', '.bmp']:
#         return True
#     else:
#         return False
def shadow_mask(crop_size, img, random_size):
    blur_img = img
    height = img.shape[0]
    width  = img.shape[1]

    for j in range(random_size):
        idx = np.random.randint(0, height-crop_size+1)
        idy = np.random.randint(0, width-crop_size+1)
        crop_img = img[idx:idx+crop_size, idy:idy+crop_size]
	if np.mean(crop_img) == 0:
	    j+=1
	    contiune
		
        # print('!!!!!!!!', crop_img.type)
        crop_blur_img = cv2.GaussianBlur(crop_img, (5, 5), 0)
        # crop_blur_img = crop_blur_img[ :, :, np.newaxis]
        blur_img[idx:idx+crop_size, idy:idy+crop_size] = crop_blur_img
       
    blur_img = np.where((blur_img>0) & (blur_img<255), 255, 0)        
    return blur_img

class kitti_Dataset(Dataset):
    def __init__(self, transform, data_path, label_path, image_scale_w, image_scale_h):
        self.transform = transforms.Compose([
    #transforms.Resize((image_scale_w, image_scale_h)),
    #transforms.RandomRotation(60),
    transforms.ToTensor()
    
])     
        self.data_path = data_path
        self.label_path = label_path
        self.image_scale_h = image_scale_h
        self.image_scale_w = image_scale_w
        
        # self.seed = np.random.randint(2147483647)  
    def __len__(self):
        return len(os.listdir(self.data_path))
    
    def __getitem__(self, idx):
        # random.seed(self.seed)
        img_name = os.listdir(self.data_path)
        img_name = list(filter(file_filter, img_name))[idx]
        imgA = cv2.imread(self.data_path + '/' + img_name)
        imgA = cv2.resize(imgA, (self.image_scale_w, self.image_scale_h))
        imgA = Image.fromarray(np.uint8(imgA)) 
        imgB = cv2.imread(self.label_path + '/' + img_name, 0)
        imgB = cv2.resize(imgB, (self.image_scale_w, self.image_scale_h))
        # imgB = imgB[ :, :, np.newaxis]
        # imgB = Image.fromarray(np.uint8(imgB)) 
        # imgC = cv2.imread(self.label_path + '/' + img_name, 0)               
        # imgC = cv2.resize(imgC, (self.image_scale_w, self.image_scale_h))
        imgC = copy.deepcopy(imgB)
        imgC = np.where(imgC>0, 255, 0)
        imgC = imgC.astype(np.uint8)
        imgC = shadow_mask(crop_size=32, img=imgC, random_size=10)
        imgC = imgC[np.newaxis, :, : ]
        imgC = torch.FloatTensor(imgC)

        imgD = copy.deepcopy(imgB)
        imgD = np.where(imgD>0, 255, 0)
        imgD = imgD.astype(np.uint8)
        imgD = cv2.Canny(imgD, 80, 200)
        imgD = imgD[np.newaxis, :, : ]
        imgD = torch.FloatTensor(imgD)
#------------------------------------------------------------------------------------
        # imgC = Image.fromarray(np.uint8(imgC))

        # degrees = np.random.randint(0, 180) 
        # imgA = transforms.functional.rotate(img=imgA, angle=degrees)
        # imgB = transforms.functional.rotate(img=imgB, angle=degrees)
        # imgC = transforms.functional.rotate(img=imgC, angle=degrees)

        if self.transform:
            # imgA, imgB, imgC = self.transform(imgA), self.transform(imgB), self.transform(imgC)
            imgA, imgB = self.transform(imgA), self.transform(imgB)
            
        return (imgA-0.5)/0.5, imgB, imgC/255, imgD/255


class test_Dataset(Dataset):

    def __init__(self, transform, data_path, label_path, image_scale_w, image_scale_h):
        self.transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
        self.test_data_path = data_path
        self.label_path    = label_path
        self.image_scale_h = image_scale_h
        self.image_scale_w = image_scale_w
    def __len__(self):
        return len(os.listdir(self.test_data_path))
    
    def __getitem__(self, idx):

        img_name = os.listdir(self.test_data_path)
        img_name = list(filter(file_filter, img_name))[idx]
        imgA = cv2.imread(self.test_data_path + '/' + img_name)
        height, width = imgA.shape[0], imgA.shape[1]
        imgA = cv2.resize(imgA, (self.image_scale_w, self.image_scale_h))
        imgB = cv2.imread(self.label_path + '/' + img_name)
        imgB = cv2.resize(imgB, (self.image_scale_w, self.image_scale_h))
        if self.transform:
            imgA, imgB = self.transform(imgA), self.transform(imgB)           
        return (imgA-0.5)/0.5, imgB, img_name[:-4], height, width
def init_train_data(data_path, label_path, image_scale_w, image_scale_h, batch_size):
	img_road = kitti_Dataset(transform=True, data_path=data_path, label_path=label_path, image_scale_w=image_scale_w, 
	    image_scale_h=image_scale_h)
	train_dataloader = DataLoader(img_road, batch_size=batch_size, shuffle=True, drop_last=True)
	print('train_data', len(train_dataloader.dataset), train_dataloader.dataset[7][0].shape, train_dataloader.dataset[7][1].shape)
	return train_dataloader

def init_test_data(data_path, label_path, image_scale_w, image_scale_h, batch_size):	
	img_road_test = test_Dataset(transform=True, data_path=data_path, label_path=label_path, image_scale_w=image_scale_w, 
	    image_scale_h=image_scale_h)
	test_dataloader = DataLoader(img_road_test, batch_size=batch_size, shuffle=False)
	print('test_data', len(test_dataloader.dataset), test_dataloader.dataset[7][0].shape)
	return test_dataloader	
