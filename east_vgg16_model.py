#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# #### EAST VGG16

# In[ ]:


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


# #### conv->batch norm->relu->pool

# In[ ]:


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# #### inplace=True
# inplace=True means that it will modify the input directly, without allocating any additional output. It can sometimes slightly decrease the memory usage, but may not always be a valid operation (because the original input is destroyed). However, if you don’t see an error, it means that your use case is valid.
# 
# 作者：VanJordan
# 链接：https://www.jianshu.com/p/8385aa74e2de
# 来源：简书
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

# In[ ]:


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), 
            nn.ReLU(True), 
            nn.Dropout(), 
            nn.Linear(4096, 4096), 
            nn.ReLU(True), 
            nn.Dropout(), 
            nn.Linear(4096, 1000), 
        )
        
        # 根据不同的module类型，进行不同的参数初始化
        for m in self.modules():  # 遍历VGG的所有层
            if isinstance(m, nn.Conv2d):  # 判断m的类型
                # fan_out表示初始化参数后输出的方差为1
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # bias初始化为0
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 服从正态分布，均值0， 方差0.01
                nn.init.constanct_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # batch_size, 512 * 7 * 7
        x = self.calssifier(x)  # batch_size, 1000
        return x


# In[ ]:


class extractor(nn.Module):
    def __init__(self, pretrained):
        super(extractor, self).__init__()
        vgg16_bn = VGG(make_layers(cfg, batch_norm=True))
        if pretrained:
            vgg16_bn.load_state_dict(torch.load('./pths/vgg16_bn-6c64b313.pth'))
        self.features = vgg16_bn.features
    
    def forward(self, x):
        out = []
        for m in self.features:
            x = m(x)
            if isinstance(m, nn.MaxPool2d):
                out.append(x)  # 保存maxpool之后的结果
        return out[1:]  # 1/4, 1/8, 1/16, 1/32


# In[ ]:


class merge(nn.Module):
    def __init__(self):
        super(merge, self).__init__()
        # 定义 conv stage_1
        self.conv1 = nn.Conv2d(1024, 128, 1)  # concat:512+512, conv 1x1,128
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3) # conv 3x3,128
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        
        # 定义 conv stage_2
        self.conv3 = nn.Conv2d(384, 64, 1)  # concat:256+128, conv 1x1, 64
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3)  # conv 3x3, 64
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        
        # 定义 conv stage_3
        self.conv5 = nn.Conv2d(192, 32, 1)  # concat:128+64, conv 1x1, 32
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32, 32, 3)  # conv 3x3, 32
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()
        
        # 定义 conv stage_4
        self.conv7 = nn.Conv2d(32, 32, 3)  # conv 3x3, 32
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[2]), 1)
        y = self.relu1(self.bn1(self.conv1(y)))
        y = self.relu2(self.bn2(self.conv2(y)))
        
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[1]), 1)
        y = self.relu3(self.bn3(self,conv3(y)))
        y = self.relu4(self.bn4(self.conv4(y)))
        
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[0]), 1)
        y = self.relu5(self.bn5(self.conv5(y)))
        y = self.relu6(self.bn6(self.conv6(y)))
        
        y = self.relu7(self.bn7(self.conv7(y)))
        return y


# In[ ]:


class output(nn.Module):
    def __init__(self, scope=512):
        super(output, self).__init__()
        
        # 输出score map，conv 1x1，1
        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()  # 0~1
        
        # 输出d1，d2，d3, d4 map, conv 1x1, 4
        self.conv2 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        
        # 输出angle_map, conv1x1, 1
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        
        self.scope = scope # 512
        for m in self.modules():
            if instance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        score = self.sigmoid1(self.conv1(x))  # 0~1
        loc = self.sigmoid2(self.conv2(x)) * self.scope  # 0~512
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi  # -pi/2 ~ pi/2
        geo = torch.cat((loc, angle), 1)
        return score, geo


# In[ ]:


class EAST(nn.Module):
    def __init__(self, pretrained=True):
        super(EAST, self).__init__()
        self.extractor = extractor(pretrained)
        self.merge = merge()
        self.output = output()
    
    def forward(self, x):
        return self.output(self.merge(self.extractor(x)))


# In[ ]:


if __name__ == '__main__':
    m = EAST()
    x = torch.randn(1, 3, 256, 256)
    score, geo = m(x)
    print(score.shape)
    print(geo.shape)

