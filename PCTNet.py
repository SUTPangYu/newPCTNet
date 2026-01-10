import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from multi_scale_module import ASPP,GPM,PPM,FoldConv_aspp,PAFEM
from torch.nn import Softmax

class RGBD_sal(nn.Module):

    def __init__(self):
        super(RGBD_sal, self).__init__()
        ################################vgg16#######################################
        feats = list(models.vgg16_bn(pretrained=True).features.children())
        self.conv0 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(*feats[1:6])
        self.conv2 = nn.Sequential(*feats[6:13])
        self.conv3 = nn.Sequential(*feats[13:23])
        self.conv4 = nn.Sequential(*feats[23:33])
        self.conv5 = nn.Sequential(*feats[33:43])

        ################################vgg19#######################################
        # feats = list(models.vgg19_bn(pretrained=True).features.children())
        # self.conv0 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        # self.conv1 = nn.Sequential(*feats[1:6])
        # self.conv2 = nn.Sequential(*feats[6:13])
        # self.conv3 = nn.Sequential(*feats[13:26])
        # self.conv4 = nn.Sequential(*feats[26:39])
        # self.conv5 = nn.Sequential(*feats[39:52])

        self.softmax = Softmax(dim=-1)

        # PAFEM
        self.dem1 = PAFEM(512, 512)
        self.dem2a= PAFEM(512,512)
        self.dem3a = PAFEM(256, 256)
        self.dem4a = PAFEM(128, 128)
        self.dem5a = PAFEM(64, 64)
        # vanilla convolution
        # self.dem1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())

        self.dem2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.dem3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.dem4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.dem5 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())

        self.fuse_1 =  nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU(),nn.Conv2d(256, 1, kernel_size=3, padding=1))
        self.fuse_2 =  nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU(),nn.Conv2d(128, 1, kernel_size=3, padding=1))
        self.fuse_3 =  nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU(),nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.fuse_4 =  nn.Sequential(nn.Conv2d(64, 32, kernel_size=1), nn.BatchNorm2d(32), nn.PReLU(),nn.Conv2d(32, 1, kernel_size=3, padding=1))
        self.fuse_5 =  nn.Sequential(nn.Conv2d(32, 16, kernel_size=1), nn.BatchNorm2d(16), nn.PReLU(),nn.Conv2d(16, 1, kernel_size=3, padding=1))

        self.output1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.output2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.output3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.output4 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1),nn.BatchNorm2d(32), nn.PReLU())
        self.output5 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1), nn.PReLU())

        self.br_conv1_1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.br_conv1_2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.br_conv2_1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.br_conv2_2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.br_conv3_1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.br_conv3_2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())

        self.scale1 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.PReLU())
        self.scale2 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.PReLU())
        self.scale3 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.PReLU())
        self.scale4 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.PReLU())
        self.scale5 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.PReLU())
        self.scale6 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.PReLU())
        self.scale7 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.PReLU())
        self.scale8 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.PReLU())
        self.scale9 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), nn.PReLU())
        self.scale = nn.Sequential(nn.Conv2d(4, 1, kernel_size=3, padding=1), nn.PReLU())

        self.fc_layer1 = nn.Linear(2304, 2304)
        self.fc_layer2 = nn.Linear(2304, 2304)
        self.fc_layer3 = nn.Linear(2304, 2304)
        self.fc_layer4 = nn.Linear(2304, 2304)
        self.fc_layer5 = nn.Linear(2304, 2304)
        self.fc_layer6 = nn.Linear(2304, 2304)
        self.fc_layer7 = nn.Linear(2304, 2304)
        self.fc_layer8 = nn.Linear(2304, 2304)
        self.fc_layer9 = nn.Linear(2304, 2304)

        self.fuseout1 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1), nn.PReLU())
        self.fuseout2 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1), nn.PReLU())
        self.fuseout3 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1), nn.PReLU())
        self.fuseout4 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1), nn.PReLU())
        self.fuseout5 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1), nn.PReLU())
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x,depth):
        input = x
        B,_,height_depth,width_depth = input.size()
        #c0 = self.conv0(depth)
        c0 = self.conv0(torch.cat((x,depth),1))
        ######### Thermal-aware#######
        depth_init=depth




        ########################### 第1层 #####################
        c1 = self.conv1(c0)
        ########################### 第2层 #####################
        c2 = self.conv2(c1)
        ########################### 第3层 #####################
        c3 = self.conv3(c2)
        # 注释
        m_batchsize_tisa, C_tisa, height_tisa, width_tisa = c3.size()
        depth = F.upsample(depth, size=c3.size()[2:], mode='bilinear')
        c3_depth=c3*depth
        depth_var1 = self.br_conv1_1(c3_depth)
        depth_var2 = self.br_conv1_2(c3_depth)
        depth_key = depth_var1.view(m_batchsize_tisa, -1, height_tisa * width_tisa).permute(0, 2, 1)
        depth_query = depth_var2.view(m_batchsize_tisa, -1, height_tisa * width_tisa)
        # depth_weight3 = 1 / (torch.bmm(depth_key, depth_query) + 0.2)
        depth_weight3 = torch.bmm(depth_key, depth_query)
        mean = depth_weight3.mean(dim=1, keepdim=True)  # 沿着维度1（第二维）计算平均值，并保持维度
        std = depth_weight3.std(dim=1, keepdim=True)
        depth_weight3 = (depth_weight3 - mean) / std
        tt3 = self.dem3a(c3, depth_weight3)
        tt3 = F.upsample(tt3, size=c3.size()[2:], mode='bilinear')
        ########################### 第4层 #####################
        c4 = self.conv4(tt3)
        m_batchsize_tisa, C_tisa, height_tisa, width_tisa = c4.size()
        depth = F.upsample(depth, size=c4.size()[2:], mode='bilinear')
        c4_depth = c4 * depth
        depth_var1 = self.br_conv2_1(c4_depth)
        depth_var2 = self.br_conv2_2(c4_depth)
        depth_key = depth_var1.view(m_batchsize_tisa, -1, height_tisa * width_tisa).permute(0, 2, 1)
        depth_query = depth_var2.view(m_batchsize_tisa, -1, height_tisa * width_tisa)
        #depth_weight4 = 1 / (torch.bmm(depth_key, depth_query) + 0.2)
        depth_weight4 = torch.bmm(depth_key, depth_query)
        mean = depth_weight4.mean(dim=1, keepdim=True)  # 沿着维度1（第二维）计算平均值，并保持维度
        std = depth_weight4.std(dim=1, keepdim=True)
        depth_weight4 = ( depth_weight4- mean) / std
        tt4 = self.dem2a(c4, depth_weight4)
        tt4 = F.upsample(tt4, size=c4.size()[2:], mode='bilinear')
        ########################### 第5层 #####################
        c5 = self.conv5(tt4)
        m_batchsize_tisa, C_tisa, height_tisa, width_tisa = c5.size()
        depth = F.upsample(depth, size=c5.size()[2:], mode='bilinear')
        c5_depth = c5 * depth
        depth_var1 = self.br_conv3_1(c5_depth)
        depth_var2 = self.br_conv3_2(c5_depth)
        depth_key = depth_var1.view(m_batchsize_tisa, -1, height_tisa * width_tisa).permute(0, 2, 1)
        depth_query = depth_var2.view(m_batchsize_tisa, -1, height_tisa * width_tisa)
        # depth_weight5 = 1 / (torch.bmm(depth_key, depth_query) + 0.2)
        depth_weight5=torch.bmm(depth_key, depth_query)
        mean = depth_weight5.mean(dim=1, keepdim=True)  # 沿着维度1（第二维）计算平均值，并保持维度
        std = depth_weight5.std(dim=1, keepdim=True)
        depth_weight5 = (depth_weight5 - mean) / std
        dem1 = self.dem1(c5, depth_weight5)

        #################################Transition layer###########################################

        dem2 = self.dem2(tt4)
        dem3 = self.dem3(tt3)
        dem4 = self.dem4(c2)
        dem5 = self.dem5(c1)
        # 注释结束
        ##额外填加
        # c4 = self.conv4(c3)
        # c5 = self.conv5(c4)
        # m_batchsize_tisa, C_tisa, height_tisa, width_tisa = c5.size()
        # depth = F.upsample(depth, size=c5.size()[2:], mode='bilinear')
        # c5_depth = c5 * depth
        # depth_var1 = self.br_conv3_1(c5_depth)
        # depth_var2 = self.br_conv3_2(c5_depth)
        # depth_key = depth_var1.view(m_batchsize_tisa, -1, height_tisa * width_tisa).permute(0, 2, 1)
        # depth_query = depth_var2.view(m_batchsize_tisa, -1, height_tisa * width_tisa)
        # # depth_weight5 = 1 / (torch.bmm(depth_key, depth_query) + 0.2)
        # depth_weight5=torch.bmm(depth_key, depth_query)
        # mean = depth_weight5.mean(dim=1, keepdim=True)  # 沿着维度1（第二维）计算平均值，并保持维度
        # std = depth_weight5.std(dim=1, keepdim=True)
        # depth_weight5 = (depth_weight5 - mean) / std
        # dem1 = self.dem1(c5, depth_weight5)
        # # dem1 = self.dem1(c5)
        # dem2 = self.dem2(c4)
        # dem3 = self.dem3(c3)
        # dem4 = self.dem4(c2)
        # dem5 = self.dem5(c1)
        # m_batchsize_tisa, C_tisa, height_tisa, width_tisa = c5.size()
        ##额外填加结束
        ################################Thermal representation#######################################
        s1=self.scale1(depth_init)
        s2=self.scale2(s1+depth_init)
        s3=self.scale3(s2+depth_init)
        s1=F.upsample(s1, size=[48,48], mode='bilinear')
        s2 = F.upsample(s2, size=[48, 48], mode='bilinear')
        s3 = F.upsample(s3, size=[48, 48], mode='bilinear')
        s1_new=s1.view(m_batchsize_tisa,1,48*48)
        s2_new = s2.view(m_batchsize_tisa, 1, 48 * 48)
        s3_new = s3.view(m_batchsize_tisa, 1, 48 * 48)
        s1_newest = self.fc_layer1(self.fc_layer1(self.fc_layer1(s1_new)))
        s2_newest = self.fc_layer1(self.fc_layer1(self.fc_layer1(s2_new)))
        s3_newest = self.fc_layer1(self.fc_layer1(self.fc_layer1(s3_new)))
        s1 = s1_newest.view(m_batchsize_tisa,1,48,48)
        s2 = s2_newest.view(m_batchsize_tisa, 1, 48, 48)
        s3 = s3_newest.view(m_batchsize_tisa, 1, 48, 48)
        s1=self.scale4(s1)
        s2=self.scale5(s2)
        s3=self.scale6(s3)
        s1 = F.upsample(s1, size=[height_depth, width_depth], mode='bilinear')
        s2 = F.upsample(s2, size=[height_depth, width_depth], mode='bilinear')
        s3 = F.upsample(s3, size=[height_depth, width_depth], mode='bilinear')
        DIA=torch.cat((s1, s2, s3, depth_init), 1)
        DIA=self.scale(DIA)
        ################################Decoder layer#######################################
        dem1_attention = F.sigmoid(self.fuse_1(dem1 + F.upsample(depth, size=dem1.size()[2:], mode='bilinear')+ F.upsample(DIA, size=dem1.size()[2:], mode='bilinear')))
        output1 = self.output1(dem1 * dem1_attention )
        # output1 = self.output1(dem1 )

        dem2_attention = F.sigmoid(self.fuse_2(dem2 + F.upsample(output1, size=dem2.size()[2:], mode='bilinear') + F.upsample(DIA, size=dem2.size()[2:], mode='bilinear')))
        dem2_inter=dem2_attention*(1-torch.exp(-6*F.upsample(dem1_attention, size=dem2_attention.size()[2:], mode='bilinear')))
        dem2_attention=self.fuseout2(torch.cat((dem2_attention,dem2_inter),1))
        dem2_attention=F.sigmoid(dem2_attention+dem2_inter)
        output2 = self.output2(F.upsample(output1, size=dem2.size()[2:], mode='bilinear') + dem2 *dem2_attention)
        # output2 = self.output2(F.upsample(output1, size=dem2.size()[2:], mode='bilinear') + dem2 )

        dem3_attention = F.sigmoid(self.fuse_3(dem3 + F.upsample(output2, size=dem3.size()[2:], mode='bilinear') + F.upsample(DIA, size=dem3.size()[2:],mode='bilinear')))
        dem3_inter = dem3_attention * (1 - torch.exp(-6 * F.upsample(dem2_attention, size=dem3_attention.size()[2:], mode='bilinear')))
        dem3_attention = self.fuseout3(torch.cat((dem3_attention, dem3_inter), 1))
        dem3_attention = F.sigmoid(dem3_attention + dem3_inter)
        output3 = self.output3(F.upsample(output2, size=dem3.size()[2:], mode='bilinear') + dem3 * dem3_attention)
        # output3 = self.output3(F.upsample(output2, size=dem3.size()[2:], mode='bilinear') + dem3 )

        dem4_attention = F.sigmoid(self.fuse_4(dem4 + F.upsample(output3, size=dem4.size()[2:], mode='bilinear') + F.upsample(DIA, size=dem4.size()[2:],mode='bilinear')))
        dem4_inter = dem4_attention * (1 - torch.exp(-6 * F.upsample(dem3_attention, size=dem4_attention.size()[2:], mode='bilinear')))
        dem4_attention = self.fuseout4(torch.cat((dem4_attention, dem4_inter), 1))
        dem4_attention = F.sigmoid(dem4_attention + dem4_inter)
        output4 = self.output4(F.upsample(output3, size=dem4.size()[2:], mode='bilinear') + dem4 * dem4_attention)
        # output4 = self.output4(F.upsample(output3, size=dem4.size()[2:], mode='bilinear') + dem4 )

        dem5_attention = F.sigmoid(self.fuse_5(dem5 + F.upsample(output4, size=dem5.size()[2:], mode='bilinear') + F.upsample(DIA, size=dem5.size()[2:],mode='bilinear')))
        dem5_inter = dem5_attention * (1 - torch.exp(-6 * F.upsample(dem4_attention, size=dem5_attention.size()[2:], mode='bilinear')))
        dem5_attention = self.fuseout5(torch.cat((dem5_attention, dem5_inter ), 1))
        dem5_attention = F.sigmoid(dem5_attention + dem5_inter )
        output5 = self.output5(F.upsample(output4, size=dem5.size()[2:], mode='bilinear') + dem5 * dem5_attention)
        # output5 = self.output5(F.upsample(output4, size=dem5.size()[2:], mode='bilinear') + dem5 )

        ################################Dual Branch Fuse#######################################
        output = F.upsample(output5, size=input.size()[2:], mode='bilinear')
        ################################Dual Branch Fuse#######################################
        output=output5

        if self.training:
            return output, dem1_attention, dem2_attention, dem3_attention, dem4_attention, dem5_attention
        return F.sigmoid(output), dem1_attention, dem2_attention, dem3_attention, dem4_attention, dem5_attention


if __name__ == "__main__":
    model = RGBD_sal()
    input = torch.autograd.Variable(torch.randn(4, 3, 384, 384))
    depth = torch.autograd.Variable(torch.randn(4, 1, 384, 384))
    output = model(input,depth)
