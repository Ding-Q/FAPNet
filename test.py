#-----------------------------------AVD-v2------------------------------------
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import time
import cv2


# from PIL import Image
from torchvision import utils, datasets#, models, transforms
# from torch.utils.data import DataLoader, Dataset, random_split
# from tqdm import tqdm

from datasets import *
# from nets import *
from utils import *




def test(device, test_dataloader, test_img_path, test_label_path, test_heatmap_img_path, fcrn_encode, fcrn_decode, w, h):
    fcrn_encode.eval()
    fcrn_decode.eval()
    # Gen.eval()
    begin_time = time.time()
    for batch_idx, (road, mask, img_name, height , width)in enumerate(test_dataloader):
        road, mask = road.to(device), mask.to(device)
        # z = torch.randn(road.shape[0], 1, IMAGE_SCALE, IMAGE_SCALE, device=device)
        # img_noise = torch.cat((road, z), dim=1)
        # fake_feature = Gen(img_noise)
        with torch.no_grad():
            feature, x2, x3, x4, x5, _ = fcrn_encode(road)
            det_road = fcrn_decode(feature, x2, x3, x4, x5)

            label = det_road.cpu()  
            label = np.where(label>0.5, 1, 0)      
            mask = mask.cpu()
            feature = feature.cpu()

            label = np.array(label)
            mask = np.array(mask)
            feature = np.array(feature)

            # label = np.where(label>0.5, 1, 0)
            label = np.squeeze(label)
            heatmap = heatmap_f(feature)
            heat_img = heat2img_test(heatmap, mask, w, h)
            

            cv2.imwrite(test_img_path + '/{}.png'.format(img_name[0]), label*255)
            cv2.imwrite(test_heatmap_img_path + '/{}.png'.format(img_name[0]), heat_img)
            print('testing...')
            print('{}/{}'.format(batch_idx, len(test_dataloader)))
        print('Done!')
    end_time = time.time()
    runtime = end_time - begin_time 
    print('running time:', runtime)
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print('Device:', device)

    parser = argparse.ArgumentParser(description="Super Parmaters")
    parser.add_argument('-image_scale_h', type=int, default=256)
    parser.add_argument('-image_scale_w', type=int, default=256)
    parser.add_argument('-sa_scale', type=float, default=8)
    parser.add_argument('-alpha', type=float, default=0.4)
    parser.add_argument('-dim', type=int, default=80)
    parser.add_argument('-model_num', type=int, default=540) 
    parser.add_argument('-test_data_path', type=str, default='../datasets/munich/test/img')
    parser.add_argument('-test_label_path', type=str, default='../datasets/munich/test/lab')
    parser.add_argument('-test_img_path', type=str, default='./test/V2_test')
    parser.add_argument('-test_heatmap_img_path', type=str, default='./test/kitti_test_heatmap_')
    parser.add_argument('-load_model', required=False, choices=['True', 'False'], default='True', help='choose True or False')
    opt = parser.parse_args()
    print(opt)

    auto_create_path(opt.test_img_path)
    auto_create_path(opt.test_heatmap_img_path)

    test_dataloader  = init_test_data(data_path=opt.test_data_path, label_path=opt.test_label_path, image_scale_w=opt.image_scale_w, image_scale_h=opt.image_scale_h, batch_size=1)
    fcrn_encode, fcrn_decode, _ = init_net(load_model=opt.load_model, dim=opt.dim, sa_scale=opt.sa_scale, alpha=opt.alpha, device=device, model_num=opt.model_num)
    test(device, test_dataloader, opt.test_img_path, opt.test_label_path, opt.test_heatmap_img_path, fcrn_encode, fcrn_decode, opt.image_scale_w, opt.image_scale_h)
    acc = iou(opt.test_img_path, opt.test_label_path, test, opt.image_scale_w, opt.image_scale_h, opt.alpha, opt.log) 