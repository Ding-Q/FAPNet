#-----------------------------------AVD-v2------------------------------------
import os
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# import torch.utils.data as data
from nets import *
import cv2


def file_filter(f):
    if f[-4:] in ['.jpg', '.png', '.bmp']:
        return True
    else:
        return False

def auto_create_path(FilePath):
    if os.path.exists(FilePath):   
    	print( FilePath + ' dir exists'  ) 
    else:
        print( FilePath + ' dir not exists')
        os.makedirs(FilePath)

def heatmap_f(fea):
    heatmap = np.squeeze(fea)
    heatmap = np.mean(np.array(heatmap), axis=0)
    heatmap = np.maximum(heatmap, 0)  # heatmap与0比较，取其大者
    heatmap /= np.max(heatmap)
    return heatmap

def heat2img_train(heatmap, img, w, h, batch):
    heatmapgroup = []
    heatmap = np.transpose(heatmap, (1, 2, 0))
    print('!!!!!!!!!!', heatmap.shape)
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.transpose(heatmap, (2, 0, 1))
    print('!!!!!!!!!!', heatmap.shape)
    for i in range(len(heatmap.shape[0])):
        heatmap_ = np.split(heatmap, [i+1, heatmap.shape[0]-i-1], axis=0)
        heatmap_ = np.squeeze(heatmap_)
        heatmap_ = np.uint8((heatmap_+1)*255/2)
        heatmap_ = cv2.applyColorMap(heatmap_, cv2.COLORMAP_JET)
        heatmap_ = heatmap_[np.newaxis, :, :, :]
    heatmapgroup.append(heatmap_)
    heatmapgroup = np.array(heatmapgroup)
    return heatmapgroup*0.4 + (img+1)*255/2

def heat2img_test(heatmap, img, w, h):
    img = np.squeeze(img)
    img = np.transpose(img, (1, 2, 0))
    # heatmap = np.transpose(heatmap, (1, 2, 0))
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(heatmap*255)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap*0.4 + (img+1)*255/2*0.6



def iou(path_img, path_lab, epoch, image_scale_w, image_scale_h, alpha, log):
    img_name = os.listdir(path_img)
    img_name = list(filter(file_filter, img_name))   
#     img_name.sort(key=lambda x:int(x[:-4]))
    iou_list = []
    for i in range(len(img_name)):
        det = img_name[i]
        det = cv2.imread(path_img + '/' + det, 0)
#         print(det.shape)
        lab = img_name[i]
        lab = cv2.imread(path_lab + '/' + lab, 0)
#         cv2.imwrite('./label/' + lab, lab*255)
#         print(lab.shape)
        lab = cv2.resize(lab, (image_scale_w, image_scale_h))
        count0, count1, a, count2 = 0, 0, 0, 0
        for j in range(det.shape[0]):
            for k in range(det.shape[1]):
                #TP
                if det[j][k] != 0 and lab[j][k] != 0:
                    count0 += 1
                elif det[j][k] == 0 and lab[j][k] != 0:
                    count1 += 1
                elif det[j][k] != 0 and lab[j][k] == 0:
                    count2 += 1
                #iou = (count1 + count2)/(det.shape[0] * det.shape[1])
                iou = count0/(count1 + count0 + count2 + 0.0001)
                    
#                 elif det[j][k] != 0 and lab[j][k] == 0:
#                     count1 += 1
                #FN
#                 elif det[j][k] == 0 and lab[j][k] != 0:
#                     count2 += 1
                #iou = (count0)/(count0 + count1)
#                 iou = count0/(count1 + count0 + 0.00001)
        iou_list.append(iou)
        print(img_name[i], ':', iou)
    print('mean_iou:', sum(iou_list)/len(iou_list))
    with open(log.format('edge_loss'),"a") as f:
        f.write("model_num" + " " + str(epoch) + " " + 'mean_iou:' + str(sum(iou_list)/len(iou_list)) + '\n')
    return sum(iou_list)/len(iou_list)  

def init_net(dim, sa_scale, alpha, model_num, device, load_model=False):

    fcrn_encode = Fcrn_encode(dim)
    fcrn_encode = fcrn_encode.to(device)
    fcrn_encode = nn.DataParallel(fcrn_encode)
    
    if load_model == 'True':
        fcrn_encode = torch.load('./model/fcrn_encode_{}_{}_res.pkl'.format(alpha,model_num),  map_location=torch.device(device))
        
    fcrn_decode = Fcrn_decode(sa_scale, dim)
    fcrn_decode = fcrn_decode.to(device)
    fcrn_decode = nn.DataParallel(fcrn_decode)

    if load_model == 'True':
        fcrn_decode = torch.load('./model/fcrn_decode_{}_{}_res.pkl'.format(alpha,model_num),  map_location=torch.device(device))

    Gen = Generator(dim)
    Gen = Gen.to(device)
    Gen = nn.DataParallel(Gen)

    if load_model == 'True':
        Gen = torch.load('./model/Gen_{}_{}_res.pkl'.format(alpha,model_num),  map_location=torch.device(device))

    # Dis = Discriminator(dim)
    # Dis = Dis.to(device)
    # Dis = nn.DataParallel(Dis)

    # if load_model == 'True':
    #     Dis = torch.load('./model/Dis{}_'.format(alpha) + str(model_num) + '_flex.pkl')
    return fcrn_encode, fcrn_decode, Gen#, Dis

def init_optimizer(fcrn_encode, fcrn_decode, Gen, lr, lr_1):
    # Dis_optimizer = optim.Adam(Dis.parameters(), betas=[0.5, 0.999], lr=lr_1)
    # Dis_scheduler = optim.lr_scheduler.MultiStepLR(Dis_optimizer, milestones=[150, 250, 350, 550], gamma=0.2)
    Fcrn_encode_optimizer = optim.Adam(fcrn_encode.parameters(), betas=[0.5, 0.999], lr=lr)
    encode_scheduler = optim.lr_scheduler.MultiStepLR(Fcrn_encode_optimizer, milestones=[200, 300, 400, 500, 600], gamma=0.2)
    Fcrn_decode_optimizer = optim.Adam(fcrn_decode.parameters(), betas=[0.5, 0.999], lr=lr)
    decode_scheduler = optim.lr_scheduler.MultiStepLR(Fcrn_decode_optimizer, milestones=[200, 300, 400, 500, 600], gamma=0.2)
    Gen_optimizer = optim.Adam(Gen.parameters(), betas=[0.5, 0.999], lr=lr_1)
    Gen_scheduler = optim.lr_scheduler.MultiStepLR(Gen_optimizer, milestones=[200, 300, 400, 500, 600], gamma=0.2)
    return Fcrn_encode_optimizer, Fcrn_decode_optimizer, Gen_optimizer, encode_scheduler, decode_scheduler, Gen_scheduler
