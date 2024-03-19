# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from function import calc_mean_std, mean_variance_norm
import cv2
import numpy as np

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # ssim_map = (2 * sigma12 + C2) / (2 * np.sqrt(sigma1_sq * sigma2_sq) + C2)
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
  the same outputs as MATLAB's
  img1, img2: [0, 255]
  '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


# 调用


flag = True


# Aesthetic discriminator
class AesDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(AesDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        # Construct three discriminator models
        self.models = nn.ModuleList()
        self.score_models = nn.ModuleList()
        for i in range(3):
            self.models.append(
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512)
                )
            )
            self.score_models.append(
                nn.Sequential(
                    nn.Conv2d(512, 1, 3, padding=1)
                )
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    # AesDiscriminator(
    #     (models): ModuleList(
    # (0): Sequential(
    # (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    # (1): LeakyReLU(negative_slope=0.2)
    # (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    # (3): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    # (4): LeakyReLU(negative_slope=0.2)
    # (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    # (7): LeakyReLU(negative_slope=0.2)
    # (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    # (9): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    # (10): LeakyReLU(negative_slope=0.2)
    # )
    # (1): Sequential(
    #     (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    # (1): LeakyReLU(negative_slope=0.2)
    # (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    # (3): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    # (4): LeakyReLU(negative_slope=0.2)
    # (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    # (7): LeakyReLU(negative_slope=0.2)
    # (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    # (9): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    # (10): LeakyReLU(negative_slope=0.2)
    # )
    # (2): Sequential(
    #     (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    # (1): LeakyReLU(negative_slope=0.2)
    # (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    # (3): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    # (4): LeakyReLU(negative_slope=0.2)
    # (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    # (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    # (7): LeakyReLU(negative_slope=0.2)
    # (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    # (9): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    # (10): LeakyReLU(negative_slope=0.2)

    # (score_models): ModuleList(
    #     (0): Sequential(
    #     (0): Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # )
    # (1): Sequential(
    #     (0): Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # )
    # (2): Sequential(
    #     (0): Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # )
    # )
    # (downsample): AvgPool2d(kernel_size=3, stride=2, padding=[1, 1])
    # )

    # Compute the MSE between model output and scalar gt
    def compute_loss(self, x, gt):
        _, outputs = self.forward(x)  # 把x给forward，会返回两个值feat和outputs

        loss = sum([torch.mean((out - gt) ** 2) for out in outputs])
        return loss

    # 注释，因为刚开始x传入feats没有进项下采样，第二次下采样传给feats,第三次下采样传给feats;符合论文讲的
    # 训练一轮 鉴别器被调用了三次forward,是在计算以下损失函数时，调用。
    # loss_gan_d = self.discriminator.compute_loss(style, 1) + self.discriminator.compute_loss(g_t.detach(), 0)
    # loss_gan_g = self.discriminator.compute_loss(g_t, 1)
    def forward(self, x):  # 80000次后把style给鉴别器
        outputs = []
        feats = []
        for i in range(len(self.models)):  # for i in range(3) 0,1,2
            feats.append(self.models[i](x))  # x放入了models[0] ; x放入了models[1] ;
            outputs.append(self.score_models[i](self.models[i](x)))  # x放入models[0]后再放入score_models[0],score_models[1]
            x = self.downsample(x)  # x放入downsample-downsample

        # print(feats[0].size()) torch.Size([4, 512, 16, 16])
        # print(feats[1].size()) torch.Size([4, 512, 8, 8])
        # print(feats[2].size())  torch.Size([4, 512, 4, 4])
        # print(outputs[2].size()[3])  # 1 1 35 46 / 1 1 17 23/ 1 1 8 11
        #                      feats[0]    feats[1]    feats[2]
        # print(feats[0].size()[0]) 4         4          4
        # print(feats[0].size()[1]) 512      512         512
        # print(feats[0].size()[2])  16        8          4
        # print(feats[0].size()[3])  16        8          4

        self.upsample = nn.Upsample(size=(feats[0].size()[2], feats[0].size()[3]), mode='nearest')  # （16,16）

        # print(feats[2]) # 打印feat[3]就报错，说明只有feats[0]/feats[1]/feats[2] 即三个相同的审美鉴别器块
        feat = feats[0]
        for i in range(1, len(feats)):  # for i in range(1,3) 即i=1,2
            # feat是models 共3块feats[0]/feats[1]/feats[2]
            feat += self.upsample(feats[i])  # feat=feats[0]+upsample(feats[1])+upsmaple(feats[2])

        # feat=feats[0]+upsample(feats[1])+upsmaple(feats[2])
        # outputs=modles_score[models(x)][0]+modles_score[models(x)][1]+modles_score[models(x)][2]
        return feat, outputs
        # feat是提取到的特征Fa

class SA_fusion(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(SA_fusion,self).__init__()
        self.conv_f1 = nn.Conv2d(in_channel,out_channel,(1,1))
        self.conv_f2 = nn.Conv2d(in_channel,out_channel,(1,1))
        self.conv_f3 = nn.Conv2d(in_channel,out_channel,(1,1))
        self.conv_frs = nn.Conv2d(in_channel,out_channel,(1,1))
        self.softmax = nn.Softmax(dim = -1)


    def forward(self,x_fcc,x_fss):  # style, aesthetic_feats
        Bc,Cc,Hc,Wc = x_fcc.shape
        # print("x_fcc",x_fcc.shape)  # torch.Size([2, 512, 32, 32])
        Bs,Cs,Hs,Ws = x_fss.shape
        # print("x_fss", x_fss.shape)  # torch.Size([2, 512, 32, 32])
        x_fcc1 = self.conv_f1(x_fcc)  # [B,C,H,W]
        # print("x_fcc1",x_fcc1.shape)  # torch.Size([2, 512, 32, 32])
        x_fss2 = self.conv_f2(x_fss)  # [B,C,H,W]
        # print("x_fss2", x_fss2.shape)  # torch.Size([2, 512, 32, 32])
        x_fss3 = self.conv_f3(x_fss).view(Bc,Cs,Hs*Ws)
        # position_wise Multiplication
        # print("x_fss3",x_fss3.shape)  # ([2, 512, 1024])  32*32=1024
        # softmax操作通常用于将一组数转换为概率分布。给定一个输入向量，softmax函数会将向量中的每个元素转换为非负值，并且所有元素的和为1
        # bmm矩阵乘法
        # print("bmm",torch.bmm(x_fcc1.view(Bc,Cc,Hc*Wc).permute(0, 2, 1),x_fss2.view(Bs,Cs,Hs*Ws)).shape)  torch.Size([2, 1024, 1024])
        Acs = self.softmax(torch.bmm(x_fcc1.view(Bc,Cc,Hc*Wc).permute(0, 2, 1),x_fss2.view(Bs,Cs,Hs*Ws))) # softmax 按行计算的 [Nc,Ns]
        # print("Acs",Acs.shape)  # ([2, 1024, 1024])
        x_frs= torch.bmm(x_fss3,Acs.permute(0, 2, 1)) # Acs的每一行代表相关性 [Cs,Ns] x [Ns,Nc] = [Cs,Nc]
        # print("x_frs1",x_frs.shape)  # torch.Size([2, 512, 1024])
        x_frs = self.conv_frs(x_frs.view(x_fcc.shape))
        # print("x_frs2",x_frs.shape)  # torch.Size([2, 512, 32, 32])
        x_fcs = torch.add(x_frs,x_fcc)
        # print("x_fcs",x_fcs.shape)  # torch.Size([2, 512, 32, 32])
        return x_fcs

# Aesthetic-aware style-attention (AesSA) module
class AesSA(nn.Module):
    def __init__(self, in_planes):
        super(AesSA, self).__init__()

        self.SA_fusion = SA_fusion(in_channel=in_planes, out_channel=in_planes)
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.d = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.e = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.o1 = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = 256*256
        self.out_conv1 = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.out_conv2 = nn.Conv2d(in_planes, in_planes, (1, 1))

    def mean_variance_norm(feat):
        size = feat.size()
        mean, std = calc_mean_std(feat)
        normalized_feat = (feat - mean.expand(size)) / std.expand(size)
        return normalized_feat

    def forward(self, content, style, aesthetic_feats):

        if aesthetic_feats != None:
            style_key = self.SA_fusion(style, aesthetic_feats)
        else:
            style_key = mean_variance_norm(style)
        content_key = mean_variance_norm(content)

        F = self.f(content_key)
        # print("F",F.shape)  # torch.Size([2, 512, 32, 32])
        G = self.g(style_key)
        # print("G",G.shape)  # torch.Size([2, 512, 32, 32])
        H = self.h(style)
        # print("H",H.shape)  # torch.Size([2, 512, 32, 32])
        b, _, h_g, w_g = G.size()
        # 使用 contiguous 函数确保张量在内存中是连续存储的
        G = G.view(b, -1, w_g * h_g).contiguous()
        # print("G",G.shape)  # torch.Size([2, 512, 1024])
        # .transpose(1, 2)：这一步是对 view 后的张量进行转置操作，将第1维和第2维进行交换
        style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        # print("style_flat",style_flat.shape)  # torch.Size([2, 1024, 512])
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        # print("F",F.shape)  # torch.Size([2, 1024, 512])
        S = torch.bmm(F, G)  # content_key,style_key批矩阵乘法操作
        # print("S1",S.shape)  # torch.Size([2, 1024, 1024])
        # S: b, n_c, n_s
        S = self.sm(S)  # 将一组数转换为概率分布,每个元素转换为非负值，并且所有元素的和为1
        # print("S",S.shape)  # torch.Size([2, 1024, 1024])
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # print("mean1",mean.shape)  # torch.Size([2, 1024, 512])
        # style_flat ** 2 表示对 style_flat 中的每个元素进行平方操作。
        # torch.relu 表示修正线性单元函数，它的作用是将输入张量中的每个元素的负值置为零
        # torch.sqrt 则表示对输入张量中的每个元素进行开方操作，得到对应的平方根值。
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # print("std1",std.shape)  # torch.Size([2, 1024, 512])
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        # print("mean",mean.shape)  # torch.Size([2, 512, 32, 32])
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        # print("std",std.shape)  # torch.Size([2, 512, 32, 32])
        # print("jieguo",(std * mean_variance_norm(content) + mean).shape)  # torch.Size([2, 512, 32, 32])

        return std * mean_variance_norm(content) + mean


class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.AesSA_4_1 = AesSA(in_planes=in_planes)
        self.AesSA_5_1 = AesSA(in_planes=in_planes)

        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1, aesthetic_feats=None):
        # print(content4_1.size())  torch.Size([4, 512, 32, 32])
        # print(style4_1.size())    torch.Size([4, 512, 32, 32])
        # print(style5_1.size())    torch.Size([4, 512, 16, 16])
        self.upsample_content4_1 = nn.Upsample(size=(content4_1.size()[2], content4_1.size()[3]),
                                               mode='nearest')  # size=(32,32)
        self.upsample_style4_1 = nn.Upsample(size=(style4_1.size()[2], style4_1.size()[3]),
                                             mode='nearest')  # size=(32,32)
        self.upsample_style5_1 = nn.Upsample(size=(style5_1.size()[2], style5_1.size()[3]),
                                             mode='nearest')  # size=(16,16)

        # return self.merge_conv(self.merge_conv_pad(
        #     self.AesSA_4_1(content4_1, style4_1) + self.upsample_content4_1(
        #         self.AesSA_5_1(content5_1, style5_1))))

        if aesthetic_feats != None:
            return self.merge_conv(self.merge_conv_pad(self.AesSA_4_1(content4_1, style4_1, self.upsample_style4_1(aesthetic_feats)) + self.upsample_content4_1(self.AesSA_5_1(content5_1, style5_1, self.upsample_style5_1(aesthetic_feats)))))
        else:
            return self.merge_conv(self.merge_conv_pad(
                self.AesSA_4_1(content4_1, style4_1, aesthetic_feats) + self.upsample_content4_1(
                    self.AesSA_5_1(content5_1, style5_1, aesthetic_feats))))
class Net(nn.Module):
    def __init__(self, encoder, decoder, discriminator):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        self.transform = Transform(in_planes=512)
        self.decoder = decoder
        self.discriminator = discriminator
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 features from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # content loss
    def calc_content_loss(self, input, target, norm=False):
        if (norm == False):
            return self.mse_loss(input, target)
        else:
            return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    # style loss
    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    #  Net前向传播 输入（content, style, aesthetic=False）
    def forward(self, content, style, aesthetic):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        if aesthetic:
            aesthetic_s_feats, _ = self.discriminator(style)
            stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4], aesthetic_s_feats)
                # 把编码器提取到的特征Content4_1, Style4_1, Content5_1, Style5_1传入transform 得到stylized

            g_t = self.decoder(stylized)  # g_t是最后decoder输出的风格化图像
            # print("g_t",g_t.shape)  # torch.Size([4, 3, 256, 256])

            g_t_feats = self.encode_with_intermediate(g_t)  # encode_with_intermediate提取 relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 特征从出入图像中

            # content loss
            loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm=True) + self.calc_content_loss(
                g_t_feats[4], content_feats[4], norm=True)
            # style loss
            loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
            for i in range(1, 5):
                loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

            # adversarial loss
            loss_gan_d = self.discriminator.compute_loss(style, 1) + self.discriminator.compute_loss(g_t.detach(), 0)
            loss_gan_g = self.discriminator.compute_loss(g_t, 1)


            aesthetic_g_t_feats, _ = self.discriminator(g_t)
            # loss_AR2
            aesthetic_s_feats, _ = self.discriminator(style)
            loss_aesthetic = self.calc_style_loss(aesthetic_g_t_feats, aesthetic_s_feats)

        # other losses in stage I
            # identity loss
        Icc = self.decoder(self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4]))
        Iss = self.decoder(self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4]))

        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0],
                                                                                                    style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i],
                                                                                                         style_feats[i])
        # loss_aesthetic = 0

        l_identity = 50 * l_identity1 + l_identity2
        return g_t, loss_c, loss_s, loss_gan_d, loss_gan_g, l_identity, loss_aesthetic


