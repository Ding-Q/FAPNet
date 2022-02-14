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

from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms, utils, datasets, models
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from utils import *
from datasets import *
from nets import *
from test import *
# from test import test

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
# Image.MAX_IMAGE_PIXELS = None

writer = SummaryWriter('./runs/AVD')

parser = argparse.ArgumentParser(description="Choose mode")
parser.add_argument('-dim', type=int, default=96) 
parser.add_argument('-num_epochs', type=int, default=2000)
parser.add_argument('-model_num', type=int, default=180) 
parser.add_argument('-image_scale_h', type=int, default=256)
parser.add_argument('-image_scale_w', type=int, default=256)
parser.add_argument('-batch', type=int, default=4)
parser.add_argument('-img_cut', type=int, default=2)
parser.add_argument('-lr', type=float, default=2e-4)#1e-4
parser.add_argument('-lr_1', type=float, default=1e-4)#5e-5
parser.add_argument('-alpha', type=float, default=0.1)
parser.add_argument('-sa_scale', type=float, default=8)
parser.add_argument('-latent_size', type=int, default=100)
parser.add_argument('-data_path', type=str, default='../datasets/munich/train/img')
parser.add_argument('-label_path', type=str, default='../datasets/munich/train/lab')
parser.add_argument('-test_data_path', type=str, default='../datasets/munich/test/img')
parser.add_argument('-test_label_path', type=str, default='../datasets/munich/test/lab')
parser.add_argument('-save_img_path', type=str, default='./results')
parser.add_argument('-test_img_path', type=str, default='./test/V2_test_1')
parser.add_argument('-test_heatmap_img_path', type=str, default='./test/kitti_test_heatmap_')
parser.add_argument('-model_path', type=str, default='./model')
parser.add_argument('-log', type=str, default='./AVD_V2_crop64_res50_{}.txt')
parser.add_argument('-GPU', type=str, default='1')
parser.add_argument('-load_model', required=False, choices=['True', 'False'], help='choose True or False')
opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('using cuda:', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print('using CPU')
# print('Device:', device)

auto_create_path(opt.save_img_path)
auto_create_path(opt.test_img_path)
auto_create_path(opt.model_path)
auto_create_path(opt.test_heatmap_img_path)

train_dataloader = init_train_data(data_path=opt.data_path, label_path=opt.label_path, image_scale_w=opt.image_scale_w, image_scale_h=opt.image_scale_h, batch_size=opt.batch)
test_dataloader  = init_test_data(data_path=opt.test_data_path, label_path=opt.test_label_path, image_scale_w=opt.image_scale_w, image_scale_h=opt.image_scale_h, batch_size=1)

l1_loss = nn.L1Loss()
loss = nn.BCELoss()
smooth_loss = nn.SmoothL1Loss()

fcrn_encode, fcrn_decode, Gen = init_net(load_model=opt.load_model, dim=opt.dim, sa_scale=opt.sa_scale, alpha=opt.alpha, device=device, model_num=opt.model_num)
Fcrn_encode_optimizer, Fcrn_decode_optimizer, Gen_optimizer, encode_scheduler, decode_scheduler, Gen_scheduler = init_optimizer(fcrn_encode, fcrn_decode, Gen, opt.lr, opt.lr_1)

def train(device, train_dataloader, epoch):
    fcrn_encode.train()
    fcrn_decode.train()
    Gen.train()
    for batch_idx, (road, road_label, shadow_label, edge_lap)in enumerate(train_dataloader):
        road, road_label, shadow_label, edge_lap = road.to(device), road_label.to(device), shadow_label.to(device), edge_lap.to(device)

        feature, x2, x3, x4, x5, _= fcrn_encode(road)


        detect = fcrn_decode(feature, x2, x3, x4, x5)

        fcrn_loss = loss(detect, road_label)
        fcrn_loss += 2*smooth_loss(detect, road_label)
        Fcrn_encode_optimizer.zero_grad()
        Fcrn_decode_optimizer.zero_grad()
        fcrn_loss.backward()
        Fcrn_encode_optimizer.step()
        Fcrn_decode_optimizer.step() 
        
        z = torch.randn(road.shape[0], 1, opt.image_scale_h, opt.image_scale_w, device=device)
        G_input = torch.cat((road, z), dim=1)
        fake_feature = Gen(G_input)
        

        feature, x2, x3, x4, x5, _ = fcrn_encode(road)

        mix_feature = (1-opt.alpha)*feature.detach() + opt.alpha*fake_feature

        d_detect = fcrn_decode(mix_feature, x2, x3, x4, x5)

        # print('!!!!!!!!', shadow_label.dtype)
        g_loss = 5*loss(d_detect, shadow_label)
        Gen_optimizer.zero_grad()
        Fcrn_decode_optimizer.zero_grad()
        g_loss.backward()
        Gen_optimizer.step()

        z = torch.randn(road.shape[0], 1, opt.image_scale_h, opt.image_scale_w, device=device)
        G_input = torch.cat((road, z), dim=1)
        fake_feature = Gen(G_input)
        feature, x2, x3, x4, x5, _ = fcrn_encode(road)

        mix_feature = (1-opt.alpha)*feature + opt.alpha*fake_feature.detach()
        d_detect = fcrn_decode(mix_feature, x2, x3, x4, x5)

        fcrn_loss2 = loss(d_detect, road_label)
        fcrn_loss2 += 2*smooth_loss(d_detect, road_label)
        Fcrn_encode_optimizer.zero_grad()
        Fcrn_decode_optimizer.zero_grad()
        fcrn_loss2.backward()
        Fcrn_encode_optimizer.step()
        Fcrn_decode_optimizer.step()
        
        feature, x2, x3, x4, x5, edge_fea = fcrn_encode(road)

        # _, edge_fea = fcrn_decode(feature, x2, x3, x4, x5)
        edge_lap = transforms.functional.resize(edge_lap, [edge_fea.shape[2], edge_fea.shape[3]])
        # print('!!!!!!!!!!',edge_lap.shape, edge_fea.shape)
        edge_loss = 0.5*loss(edge_fea, edge_lap.detach())

        Fcrn_encode_optimizer.zero_grad()
        # Fcrn_decode_optimizer.zero_grad()
        edge_loss.backward()
        Fcrn_encode_optimizer.step()
        # Fcrn_decode_optimizer.step()

        writer.add_scalar('g_loss', g_loss.data.item(), global_step = batch_idx)
        writer.add_scalar('Fcrn_loss', fcrn_loss.data.item() , global_step = batch_idx)
        writer.add_scalar('Fcrn_loss', fcrn_loss2.data.item() , global_step = batch_idx)
        if batch_idx % 20 == 0:
            tqdm.write('[{}/{}] [{}/{}] Loss_Gen: {:.6f} Loss_Fcrn {:.6f} Loss_Fcrn_shadow {:.6f} Edge_loss {:.6f}'
                .format(epoch, num_epochs, batch_idx, len(train_dataloader), g_loss.data.item(), fcrn_loss.data.item(), fcrn_loss2.data.item(), edge_loss.data.item()))
        if batch_idx % 300 == 0:
            road_np = road.detach().cpu()
            

            # hmap = heatmap(fake_feature.detach().cpu())
            # heatmap_img = heat2img(hmap, road_np_, opt.image_scale_w, opt.image_scale_h, opt.batch)
            # heatmap_img = utils.make_grid(heatmap_img, nrow=opt.img_cut, padding=0)

            road_np = np.transpose(np.array(utils.make_grid(road_np, nrow=opt.img_cut, padding=0)), (1, 2, 0))
            road_label_np = road_label.detach().cpu()
            road_label_np = np.transpose(np.array(utils.make_grid(road_label_np, nrow=opt.img_cut, padding=0)), (1, 2, 0))
            detect_noise_np = d_detect.detach().cpu()
        
            detect_noise_np = np.transpose(np.array(utils.make_grid(detect_noise_np, nrow=opt.img_cut, padding=0)), (1, 2, 0))

            shadow_label_np = shadow_label.detach().cpu()
            shadow_label_np = np.transpose(np.array(utils.make_grid(shadow_label_np, nrow=opt.img_cut, padding=0)), (1, 2, 0))

            mix1 = np.concatenate(((road_np+1)*255/2, road_label_np*255), axis=1)
            mix2 = np.concatenate((shadow_label_np*255, detect_noise_np*255), axis=1)
            mix  = np.concatenate((mix1, mix2), axis=0) 


            # feature_np = cv2.resize((feature_np + 1)*255/2, (opt.image_scale_w, opt.image_scale_h))
            # fake_feature_np = cv2.resize((fake_feature_np + 1)*255/2, (opt.image_scale_w, opt.image_scale_h))
            # mix1 = np.concatenate((feature_np, fake_feature_np), axis=0)
            cv2.imwrite(opt.save_img_path + "/dete{}_{}.png".format(epoch, batch_idx), mix)
            # cv2.imwrite('./results_fcrn_noise/feature{}_{}.png'.format(epoch, batch_idx), mix1)
    #             cv2.imwrite("./results/feature{}_{}.png".format(epoch, batch_idx), (feature_img + 1)*255/2)
            # cv2.imwrite("./results9/label{}_{}.png".format(epoch, batch_idx), np.transpose(road_label.cpu().numpy(), (2, 0, 1))*255)
if __name__ == '__main__':
    with open(opt.log.format(opt.alpha),"a") as f:
        f.write('alpha='+'{}'.format(opt.alpha)+'\n')
    num_epochs = opt.num_epochs
    for epoch in tqdm(range(num_epochs)):  
        train(device, train_dataloader, epoch)
        Gen_scheduler.step()
        encode_scheduler.step()
        decode_scheduler.step()

        if epoch % 20 == 0:
            torch.save(Gen, opt.model_path + '/Gen_{}_{}_res.pkl'.format(opt.alpha,epoch))
            torch.save(fcrn_decode, opt.model_path + '/fcrn_decode_{}_{}_res.pkl'.format(opt.alpha,epoch))
            torch.save(fcrn_encode, opt.model_path + '/fcrn_encode_{}_{}_res.pkl'.format(opt.alpha,epoch))
            print('testing...')
            test(device, test_dataloader, opt.test_img_path, opt.test_label_path, opt.test_heatmap_img_path, fcrn_encode, fcrn_decode, opt.image_scale_w, opt.image_scale_h)
            acc = iou(opt.test_img_path, opt.test_label_path, epoch, opt.image_scale_w, opt.image_scale_h, opt.alpha, opt.log) 
