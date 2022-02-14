#-----------------------------------AVD-v2------------------------------------
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlockDown(nn.Module):
    def __init__(self, name, input_dim, output_dim, activate='relu'):
        super(ResidualBlockDown, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batchnormlize_1 = nn.BatchNorm2d(input_dim)
        self.activate = activate

        self.conv_0        = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv_1        = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1)
        self.conv_2        = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=2, padding=1)
        self.batchnormlize_2 = nn.BatchNorm2d(input_dim)
        if self.activate   == 'relu': 
            self.nonlinear = nn.ReLU()
        else:
            self.nonlinear = nn.LeakyReLU() 
    def forward(self, inputs):

        x1 = self.conv_shortcut(inputs)
        shortcut = self.conv_0(x1)

        x = inputs
        x = self.batchnormlize_1(x)
        x = self.nonlinear(x)
        x = self.conv_1(x)
        x = self.batchnormlize_2(x)
        x = self.nonlinear(x)
        x = self.conv_2(x) 
        return shortcut + x

class ResidualBlock(nn.Module):
    def __init__(self, name, input_dim, output_dim, activate='relu'):
        super(ResidualBlock, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batchnormlize_1 = nn.BatchNorm2d(input_dim)
        self.activate = activate

        self.conv_shortcut = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
        self.conv_1        = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1)
        self.conv_2        = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
        self.batchnormlize_2 = nn.BatchNorm2d(input_dim)
        if self.activate   == 'relu': 
            self.nonlinear = nn.ReLU()
        else:
            self.nonlinear = nn.LeakyReLU() 
        
    def forward(self, inputs):

        shortcut = inputs

        x = inputs
        x = self.batchnormlize_1(x)
        x = self.nonlinear(x)
        x = self.conv_1(x)
        x = self.batchnormlize_2(x)
        x = self.nonlinear(x)
        x = self.conv_2(x) 
        return shortcut + x 

class Self_Attn(nn.Module):
    
    def __init__(self, in_dim, sa_scale, activation=None):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
 
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//sa_scale, kernel_size = 3, stride=2, padding=1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//sa_scale, kernel_size = 3, stride=2, padding=1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 3, stride=2, padding=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.upsample = nn.Upsample(scale_factor = 2)
        self.conv_tail = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 3, stride=1, padding=1)
        self.softmax  = nn.Softmax(dim=-1) 
    def forward(self,x):
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height//4).permute(0,2,1) 
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height//4) 
        energy =  torch.bmm(proj_query,proj_key) 
        attention = self.softmax(energy) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height//4) 
 
        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width//2, height//2)
        
        out = self.upsample(out)
        out = F.leaky_relu(self.conv_tail(out))
        out = self.gamma*out + x
        return out 
class Self_Attn_shadow(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, ):
        super(Self_Attn_shadow, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation
 
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
 
        self.softmax  = nn.Softmax(dim=-1) 
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X (W*H) X C
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
 
        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)
 
        out = self.gamma*out + x
        return out
class UpProject(nn.Module):

    def __init__(self, in_channels, out_channels, activate='leaky_relu'):
        super(UpProject, self).__init__()
        self.activate = activate
        self.conv1_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv1_4 = nn.Conv2d(in_channels, out_channels, 2)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv2_3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv2_4 = nn.Conv2d(in_channels, out_channels, 2)

        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        # self.leaky_relu = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)
        if self.activate   == 'relu': 
            self.nonlinear = nn.ReLU()
        else:
            self.nonlinear = nn.LeakyReLU()
    def forward(self, x):
        # b, 10, 8, 1024
        batch_size = x.shape[0]
        out1_1 = self.conv1_1(nn.functional.pad(x, (1, 1, 1, 1)))
        out1_2 = self.conv1_2(nn.functional.pad(x, (1, 1, 0, 1)))
        out1_3 = self.conv1_3(nn.functional.pad(x, (0, 1, 1, 1)))
        out1_4 = self.conv1_4(nn.functional.pad(x, (0, 1, 0, 1)))
        

        out2_1 = self.conv2_1(nn.functional.pad(x, (1, 1, 1, 1)))
        out2_2 = self.conv2_2(nn.functional.pad(x, (1, 1, 0, 1))) 
        out2_3 = self.conv2_3(nn.functional.pad(x, (0, 1, 1, 1)))
        out2_4 = self.conv2_4(nn.functional.pad(x, (0, 1, 0, 1)))
       
        height = out1_1.size()[2]
        width = out1_1.size()[3]

        out1_1_2 = torch.stack((out1_1, out1_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out1_3_4 = torch.stack((out1_3, out1_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out1_1234 = torch.stack((out1_1_2, out1_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out2_1_2 = torch.stack((out2_1, out2_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out2_3_4 = torch.stack((out2_3, out2_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out2_1234 = torch.stack((out2_1_2, out2_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out1 = self.bn1_1(out1_1234)
        out1 = self.nonlinear(out1)
        out1 = self.conv3(out1)
        out1 = self.bn2(out1)

        out2 = self.bn1_2(out2_1234)

        out = out1 + out2
        out = self.nonlinear(out)

        return out
class edge_head(nn.Module):
    def __init__(self, dim):
        super(edge_head, self).__init__()
        self.conv_head_1                 = nn.Conv2d(in_channels=dim, out_channels=dim//4, kernel_size=3, stride=1, padding=1)
        self.bn_head                     = nn.BatchNorm2d(dim//4)
        self.conv_head_2                 = nn.Conv2d(in_channels=dim//4, out_channels=1, kernel_size=1)
    def forward(self, x):
        x = self.conv_head_1(x)
        x = F.leaky_relu(self.bn_head(x))
        # x = F.relu(self.bn_head(x))
        x = self.conv_head_2(x)
        x = torch.sigmoid(x)
        return x
      
class Fcrn_encode(nn.Module):
    def __init__(self, dim):
        super(Fcrn_encode, self).__init__()
        self.dim = dim
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.residual_block_1_down_1     = ResidualBlockDown('Detector.Res1', 1*dim, 2*dim, activate='leaky_relu')
		#128x128
        self.residual_block_1_1_none_1   = ResidualBlock('Detector.Res1.1', 2*dim, 2*dim, activate='leaky_relu')

        self.residual_block_2_down_1     = ResidualBlockDown('Detector.Res2', 2*dim, 4*dim, activate='leaky_relu')
		#64x64
        self.residual_block_2_1_none_1   = ResidualBlock('Detector.Res2.1', 4*dim, 4*dim, activate='leaky_relu')

        self.edge_head                   = edge_head(4*dim)

        self.residual_block_3_down_1     = ResidualBlockDown('Detector.Res3', 4*dim, 6*dim, activate='leaky_relu')
		#32x32
        self.residual_block_3_1_none_1   = ResidualBlock('Detector.Res3.1', 6*dim, 6*dim, activate='leaky_relu')

        self.residual_block_4_down_1     = ResidualBlockDown('Detector.Res4', 6*dim, 8*dim, activate='leaky_relu')
		#16x16
        self.residual_block_5_none_1     = ResidualBlock('Detector.Res5', 8*dim, 8*dim, activate='leaky_relu')

        self.residual_block_5_1_none_1   = ResidualBlock('Detector.Res5.1', 8*dim, 8*dim, activate='leaky_relu')
        

    def forward(self, x):
        x1 = self.conv_1(x)
        x = self.residual_block_1_down_1(x1)
        x2 = self.residual_block_1_1_none_1(x)
        x = self.residual_block_2_down_1(x2)
        x3 = self.residual_block_2_1_none_1(x)
        x_edge = self.edge_head(x3)
        x = self.residual_block_3_down_1(x3)
        x4 = self.residual_block_3_1_none_1(x)
        x = self.residual_block_4_down_1(x4)
        x5 = self.residual_block_5_none_1(x)
        feature = self.residual_block_5_1_none_1(x5)
        # x = torch.tanh(feature)   
        x = F.leaky_relu(feature)    
        return x, x2, x3, x4, x5, x_edge
'''
class Fcrn_decode(nn.Module):
    def __init__(self, sa_scale, dim):
        super(Fcrn_decode, self).__init__()
        self.dim = dim
        
        self.conv_2 = nn.Conv2d(in_channels=8*dim, out_channels=8*dim, kernel_size=3, stride=1, padding=1)

        self.residual_block_6_none_1     = ResidualBlock('Detector.Res6', 8*dim, 8*dim, activate='leaky_relu')
        self.residual_block_7_none_1     = ResidualBlock('Detector.Res7', 8*dim, 8*dim, activate='leaky_relu')
        # self.sa_0                        = Self_Attn(6*dim, sa_scale)
        #32x32
        # self.flex_conv1                  = nn.Conv2d(in_channels=6*dim, out_channels=6*dim, kernel_size=1, stride=1, padding=0)
        # self.UpProject_1                 = UpProject(2*8*dim, 4*dim)
        self.Upsample_1                  = nn.Upsample(scale_factor=2)
        self.residual_block_8_none_1     = ResidualBlock('Detector.Res8', 8*2*dim, 4*dim, activate='leaky_relu')

        # self.sa_1                        = Self_Attn(4*dim, sa_scale)
        #64x64
        # self.flex_conv2                  = nn.Conv2d(in_channels=4*dim, out_channels=4*dim, kernel_size=1, stride=1, padding=0)
        # self.UpProject_2                 = UpProject(2*4*dim, 4*dim)
        self.Upsample_2                  = nn.Upsample(scale_factor=2)
        self.residual_block_9_none_1     = ResidualBlock('Detector.Res9', 4*2*dim, 4*dim, activate='leaky_relu')
        # self.sa_2                        = Self_Attn(4*dim, sa_scale)
        
        self.residual_block_10_none_1    = ResidualBlock('Detector.Res10', 4*dim, 4*dim, activate='leaky_relu')
#------------------------------------------Edge head--------------------------------------------
        # self.edge_head                   = edge_head(2*dim)
        #128x128
        # self.flex_conv3                  = nn.Conv2d(in_channels=4*dim, out_channels=4*dim, kernel_size=1, stride=1, padding=0)
        # self.UpProject_3                 = UpProject(2*4*dim, 2*dim)
        self.Upsample_3                  = nn.Upsample(scale_factor=2)
        self.residual_block_11_none_1    = ResidualBlock('Detector.Res11', 4*2*dim, 2*dim, activate='leaky_relu')
#         self.residual_block_10_up_1      = ResidualBlockClass('Detector.Res10', 4*dim, 2*dim, resample='up', activate='leaky_relu')
        #256x256
        # self.flex_conv4                  = nn.Conv2d(in_channels=2*dim, out_channels=2*dim, kernel_size=1, stride=1, padding=0)
        # self.UpProject_4                 = UpProject(2*2*dim, 1*dim)
        self.Upsample_4                  = nn.Upsample(scale_factor=2)
        self.residual_block_12_none_1    = ResidualBlock('Detector.Res12', 2*2*dim, 1*dim, activate='leaky_relu')
        self.conv_3 = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=3, stride=1, padding=1)
#         self.sa_4                        = Self_Attn(1*dim)
#         self.residual_block_11_up_1      = ResidualBlockClass('Detector.Res11', 2*dim, 1*dim, resample='up', activate='leaky_relu')
    def forward(self, x, x2, x3, x4, x5):
        # x5_= self.flex_conv1(x5)
        # x4_= self.flex_conv2(x4)
        # x3_= self.flex_conv3(x3)
        # x2_= self.flex_conv4(x2)
        
        x = self.conv_2(x)
        x = self.residual_block_6_none_1(x)
        x = self.residual_block_7_none_1(x)
        # x = self.sa_0(x)
        print('!!!!!!!!!!!', x.shape, x5.shape)
        x = self.residual_block_8_none_1(torch.cat((x, x5), dim=1))
        x = self.Upsample_1(x)
        # x = self.UpProject_1(torch.cat((x, x5), dim=1))
        # x = self.sa_1(x)
        
        x = self.residual_block_9_none_1(torch.cat((x, x4), dim=1))
        x = self.Upsample_2(x)
        # x = self.UpProject_2(torch.cat((x, x4), dim=1))
        x = self.residual_block_10_none_1(x)

        # x_edge = self.edge_head(x)
        
        x = self.residual_block_11_none_1(torch.cat((x, x3), dim=1))
        x = self.Upsample_3(x)
        # x = self.UpProject_3(torch.cat((x, x3), dim=1))
        # x_edge = self.edge_head(x)
        
        x = self.residual_block_12_none_1(torch.cat((x, x2), dim=1))
        x = self.Upsample_4(x)
        # x = self.UpProject_4(torch.cat((x, x2), dim=1))
        
        x = self.conv_3(x)
        x = torch.sigmoid(x)
        return x
'''


class Generator(nn.Module):
    def __init__(self, dim):
        super(Generator, self).__init__()
        self.dim = dim
        self.conv_1 = nn.Conv2d(in_channels=4, out_channels=1*dim, kernel_size=3, stride=1, padding=1)

#         self.residual_block_1  = ResidualBlockDown('G.Res1', 1*dim, 2*dim)
#         #128x128
#         self.residual_block_2  = ResidualBlockDown('G.Res2', 2*dim, 4*dim)
#         #64x64
# #       self.residual_block_2_1  = ResidualBlockClass('G.Res2_1', 4*dim, 4*dim, resample='down')
#         #64x64
#         #self.residual_block_2_2  = ResidualBlockClass('G.Res2_2', 4*dim, 4*dim, resample=None)
#         #64x64
#         self.residual_block_3  = ResidualBlockDown('G.Res3', 4*dim, 4*dim)
#         #32x32
#         self.residual_block_4  = ResidualBlockDown('G.Res4', 4*dim, 8*dim)
#         #16x16 
#         self.residual_block_5  = ResidualBlock('G.Res5', 8*dim, 8*dim)
#         #16x16
#         self.residual_block_6  = ResidualBlock('G.Res6', 8*dim, 8*dim) 

        
        self.residual_block_1     = ResidualBlockDown('Detector.Res1', 1*dim, 2*dim, activate='leaky_relu')
		#128x128
        self.residual_block_2     = ResidualBlockDown('Detector.Res2', 2*dim, 4*dim, activate='leaky_relu')
		#64x64
        self.residual_block_3   = ResidualBlock('Detector.Res2.1', 4*dim, 4*dim, activate='leaky_relu')

        self.residual_block_4     = ResidualBlockDown('Detector.Res3', 4*dim, 4*dim, activate='leaky_relu')
		#32x32
        self.residual_block_5     = ResidualBlockDown('Detector.Res4', 4*dim, 8*dim, activate='leaky_relu')
		#16x16
        self.residual_block_6     = ResidualBlock('Detector.Res5', 8*dim, 8*dim, activate='leaky_relu')

        self.residual_block_7   = ResidualBlock('Detector.Res5.1', 8*dim, 8*dim, activate='leaky_relu')
        self.sa                = Self_Attn_shadow(8*dim)
        self.nonlinear         = nn.LeakyReLU()

    def forward(self, x):
     
        x = self.conv_1(x)
        x = self.residual_block_1(x)#x1:2*dimx128x128
        x = self.residual_block_2(x)#x2:4*dimx64x64
#       x = self.residual_block_2_1(x)
        #x = self.residual_block_2_2(x)
        x = self.residual_block_3(x)#x3:4*dimx32x32
        x = self.residual_block_4(x)#x4:6*dimx16x16
        x = self.residual_block_5(x)
        x = self.residual_block_6(x)
        x = self.residual_block_7(x)
        x = self.nonlinear(x) 
        x = self.sa(x)
        # x = torch.tanh(x)
        x = self.nonlinear(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()   
        self.conv_1 = nn.Conv2d(in_channels=6*dim, out_channels=6*dim, kernel_size=3, stride=1, padding=1)
        #16x16
        self.conv_2 = nn.Conv2d(in_channels=6*dim, out_channels=6*dim, kernel_size=3, stride=1, padding=1)
        
        self.conv_3 = nn.Conv2d(in_channels=6*dim, out_channels=4*dim, kernel_size=3, stride=1, padding=1)
        
        self.bn_1   = nn.BatchNorm2d(6*dim)
        
        self.conv_4 = nn.Conv2d(in_channels=4*dim, out_channels=4*dim, kernel_size=3, stride=2, padding=1)
        #8x8
        self.conv_5 = nn.Conv2d(in_channels=4*dim, out_channels=4*dim, kernel_size=3, stride=1, padding=1)
        #8x8
        self.conv_6 = nn.Conv2d(in_channels=4*dim, out_channels=2*dim, kernel_size=3, stride=2, padding=1)
        #4x4
        self.bn_2   = nn.BatchNorm2d(2*dim)
        
        self.conv_7 = nn.Conv2d(in_channels=2*dim, out_channels=2*dim, kernel_size=3, stride=1, padding=1)
        #4x4
        self.conv_8 = nn.Conv2d(in_channels=2*dim, out_channels=1*dim, kernel_size=3, stride=1, padding=1)
        #4x4
        #self.conv_9 = nn.Conv2d(in_channels=1*dim, out_channels=1, kernel_size=4, stride=1, padding=(0, 1), dilation=(1, 3))
        #1x1
        self.nonlinear = nn.LeakyReLU() 
    def forward(self, x):
        x = self.nonlinear(self.conv_1(x))
        x = self.nonlinear(self.conv_2(x))
        x = self.nonlinear(self.conv_3(x))
#         x = F.leaky_relu(self.bn_1(x), negative_slope=0.02)
        x = self.nonlinear(self.conv_4(x))
        x = self.nonlinear(self.conv_5(x))
        x = self.nonlinear(self.conv_6(x))
#         x = F.leaky_relu(self.bn_2(x), negative_slope=0.2)
        x = self.nonlinear(self.conv_7(x))
        x = self.nonlinear(self.conv_8(x))
        #x = self.conv_9(x)
        x = torch.mean(x, dim=[1, 2, 3])
        x = torch.sigmoid(x)

        return x.view(-1, 1).squeeze()

class ConvBlock(nn.Module):
    def __init__(self, in_channel, f, filters, s):
        super(ConvBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel,F1,1,stride=s, padding=0, bias=True),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,f,stride=1, padding=True, bias=True),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,1,stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F3),
        )
        self.shortcut_1 = nn.Conv2d(in_channel, F3, 1, stride=s, padding=0, bias=True)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(True)
        
    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X    
    
class IndentityBlock(nn.Module):
    def __init__(self, in_channel, f, filters):
        super(IndentityBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel,F1,1,stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,f,stride=1, padding=True, bias=True),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,1,stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)
        
    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X
'''    
class Fcrn_encode(nn.Module):
    def __init__(self, dim):
        super(Fcrn_encode, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3,dim,7,stride=2, padding=3, bias=True),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            # nn.MaxPool2d(3,2,padding=1),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(dim, f=3, filters=[1*dim, 1*dim, 4*dim], s=1),
            IndentityBlock(4*dim, 3, [1*dim, 1*dim, 4*dim]),
            
        )
        self.stage3 = nn.Sequential(
            ConvBlock(4*dim, f=3, filters=[2*dim, 2*dim, 8*dim], s=2),
            IndentityBlock(8*dim, 3, [2*dim, 2*dim, 8*dim]),
            
        )

        self.edge_head = edge_head(8*dim)

        self.stage4 = nn.Sequential(
            ConvBlock(8*dim, f=3, filters=[4*dim, 4*dim, 16*dim], s=2),
            IndentityBlock(16*dim, 3, [4*dim, 4*dim, 16*dim]),
            IndentityBlock(16*dim, 3, [4*dim, 4*dim, 16*dim]),
            
        )
        self.stage5 = nn.Sequential(
            ConvBlock(16*dim, f=3, filters=[8*dim, 8*dim, 16*dim], s=2),
            IndentityBlock(16*dim, 3, [8*dim, 8*dim, 16*dim]),
            
        )

        self.nonlinear = nn.ReLU(True) 
    def forward(self, x):
        x1 = self.stage1(x)#x1:128x128
        x2 = self.stage2(x1)#x2:64x64x4
        x3 = self.stage3(x2)#x3:32x32x8
        x_edge = self.edge_head(x3)
        x4 = self.stage4(x3)#x4:16x16x16
        x5 = self.stage5(x4)#x5:8x8x16
        # x = self.stage6(x)
        # x = self.conv(x)
        x = x5
        return x, x2, x3, x4, x5, x_edge
'''
class Fcrn_decode(nn.Module):
    def __init__(self, sa_scale, dim):
        super(Fcrn_decode, self).__init__()
        
        
        
        self.stage1 = nn.Sequential(
        	ResidualBlock('Detector.Res6', 8*dim, 8*dim, activate='leaky_relu'),
            ResidualBlock('Detector.Res7', 8*dim, 8*dim, activate='leaky_relu'),
            
        	# nn.Upsample(scale_factor=2),
            # IndentityBlock(16*dim, 3, [8*dim, 8*dim, 16*dim]),

        )

        self.stage2 = nn.Sequential(
        	nn.Conv2d(8*2*dim, 6*dim ,3 ,stride=1, padding=1, bias=True),
            nn.BatchNorm2d(6*dim),
            nn.ReLU(True),
        	ResidualBlock('Detector.Res8', 6*dim, 6*dim, activate='leaky_relu'),
        	nn.Upsample(scale_factor=2),
            ResidualBlock('Detector.Res9', 6*dim, 6*dim, activate='leaky_relu'),
            
            
            # IndentityBlock(16*dim, 3, [4*dim, 4*dim, 16*dim]),
            # IndentityBlock(16*dim, 3, [4*dim, 4*dim, 16*dim]),
            # IndentityBlock(16*dim, 3, [4*dim, 4*dim, 16*dim]),
            # nn.Conv2d(16*dim, 8*dim ,1 ,stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(8*dim),
            # nn.ReLU(True),
        )
        self.stage3 = nn.Sequential(
        	nn.Conv2d(6*2*dim, 4*dim ,3 ,stride=1, padding=1, bias=True),
            nn.BatchNorm2d(4*dim),
            nn.ReLU(True),
        	ResidualBlock('Detector.Res10', 4*dim, 4*dim, activate='leaky_relu'),
            nn.Upsample(scale_factor=2),
            ResidualBlock('Detector.Res11', 4*dim, 4*dim, activate='leaky_relu'),
            # nn.Conv2d(8*2*dim, 8*dim ,1 ,stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(8*dim),
            # nn.ReLU(True),
            # IndentityBlock(8*dim, 3, [2*dim, 2*dim, 8*dim]),
            # IndentityBlock(8*dim, 3, [2*dim, 2*dim, 8*dim]),
            # nn.Conv2d(8*dim, 4*dim ,1 ,stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(4*dim),
            # nn.ReLU(True),
            
        )
        # self.edge_head = edge_head(8*dim)

        self.stage4 = nn.Sequential(
        	nn.Conv2d(4*2*dim, 2*dim ,3 ,stride=1, padding=1, bias=True),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(True),
        	ResidualBlock('Detector.Res12', 2*dim, 2*dim, activate='leaky_relu'),
            nn.Upsample(scale_factor=2),
            ResidualBlock('Detector.Res13', 2*dim, 2*dim, activate='leaky_relu'),
        	# nn.Upsample(scale_factor=2),
         #    nn.Conv2d(4*2*dim, 4*dim ,1 ,stride=1, padding=0, bias=True),
         #    nn.BatchNorm2d(4*dim),
         #    nn.ReLU(True),
         #    IndentityBlock(4*dim, 3, [1*dim, 1*dim, 4*dim]),
        )
        self.stage5 = nn.Sequential(
        	# nn.Upsample(scale_factor=2),
        	nn.Conv2d(2*2*dim, 1*dim ,3 ,stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1*dim),
            nn.ReLU(True),
        	ResidualBlock('Detector.Res14', 1*dim, 1*dim, activate='leaky_relu'),
        	nn.Upsample(scale_factor=2),
        	ResidualBlock('Detector.Res15', 1*dim, 1*dim, activate='leaky_relu'),
        	nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=3, stride=1, padding=1),

            # nn.Conv2d(4*dim, 1*dim ,1 ,stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(1*dim),
            # nn.ReLU(True),
            # IndentityBlock(1*dim, 3, [1*dim, 1*dim, 1*dim]),
            # nn.Conv2d(dim, 1, 3, stride=1, padding=1, bias=True),
        )
        

        
       
        
#------------------------------------------Edge head--------------------------------------------
        
        #128x128
 
#     
    def forward(self, x, x2, x3, x4, x5):
        
        # print('!!!!!!!!!!!!!!!!!!!!!!', x.shape)
        x = self.stage1(x)
        x = self.stage2(torch.cat((x, x5), dim=1))

        x = self.stage3(torch.cat((x, x4), dim=1))
        # x_edge = self.edge_head(x)
        x = self.stage4(torch.cat((x, x3), dim=1))
        x = self.stage5(torch.cat((x, x2), dim=1))
        x = torch.sigmoid(x)
        return x
'''        
class Generator(nn.Module):
    def __init__(self, dim):
        super(Generator, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(4,dim,7,stride=2, padding=3, bias=True),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            # nn.MaxPool2d(3,2,padding=1),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(dim, f=3, filters=[1*dim, 1*dim, 4*dim], s=1),
            IndentityBlock(4*dim, 3, [1*dim, 1*dim, 4*dim]),
            
        )
        self.stage3 = nn.Sequential(
            ConvBlock(4*dim, f=3, filters=[2*dim, 2*dim, 8*dim], s=2),
            IndentityBlock(8*dim, 3, [2*dim, 2*dim, 8*dim]),
            
        )

        self.edge_head = edge_head(8*dim)

        self.stage4 = nn.Sequential(
            ConvBlock(8*dim, f=3, filters=[4*dim, 4*dim, 16*dim], s=2),
            IndentityBlock(16*dim, 3, [4*dim, 4*dim, 16*dim]),
            IndentityBlock(16*dim, 3, [4*dim, 4*dim, 16*dim]),
            
        )
        self.stage5 = nn.Sequential(
            ConvBlock(16*dim, f=3, filters=[8*dim, 8*dim, 16*dim], s=2),
            IndentityBlock(16*dim, 3, [8*dim, 8*dim, 16*dim]),
            
        )
        self.sa                = Self_Attn_shadow(16*dim)
        self.nonlinear = nn.ReLU(True)
    def forward(self, x):
        x1 = self.stage1(x)#x1:128x128
        x2 = self.stage2(x1)#x2:64x6432
        x3 = self.stage3(x2)#x3:32x32
        x4 = self.stage4(x3)#x4:16x16
        x5 = self.stage5(x4)#x5:8x8
        x  = self.sa(x5)
        x = self.nonlinear(x)
        return x
'''