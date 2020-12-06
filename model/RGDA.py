import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from functions import ReverseLayerF
from IPython import embed
import torch
import model.backbone as backbone
from model.local_att import DRAL
import torch.nn.functional as F
import numpy as np
from functools import reduce

class RGDA(nn.Module):

    def __init__(self, num_classes=65, base_net='ResNet50', groups=32, M=3):
        super(RGDA, self).__init__()
        self.sharedNet = backbone.network_dict[base_net]()
        self.groups = groups
        self.M=M
        
        # local attention
        self.conv1 = nn.Conv2d(2048, 4096, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4096)
        self.conv2=nn.ModuleList()
        for i in range(M):
            self.conv2.append(nn.Sequential(
                 nn.Conv2d(4096,4096,3,stride=1,padding=1+i,dilation=1+i,groups=self.groups,bias=False),
                 nn.BatchNorm2d(4096),
                 nn.ReLU(inplace=True)
            ))
        self.global_pool=nn.AdaptiveAvgPool2d(1)

        # local block
        self.localattens = nn.Sequential()
        self.localatten = {}
        for i in range(self.M):
            self.localatten[i] = DRAL(4096)
            self.localattens.add_module('localatten_'+str(i), self.localatten[i])

        self.fc1=nn.Sequential(nn.Conv2d(4096,1024,1,bias=False),
                               nn.BatchNorm2d(1024),
                               nn.ReLU(inplace=True))
        self.fc2=nn.Conv2d(1024,4096*M,1,1,bias=False)
        self.bn2 = nn.BatchNorm2d(4096)
        self.conv3 = nn.Conv2d(4096, 2048, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # bottleneck & classifier
        self.bottleneck = nn.Linear(2048, 256)
        self.source_fc = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.classes = num_classes

        # domain discriminator
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('fc1', nn.Linear(256, 1024))
        self.domain_classifier.add_module('relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dpt1', nn.Dropout())
        self.domain_classifier.add_module('fc2', nn.Linear(1024, 1024))
        self.domain_classifier.add_module('relu2', nn.ReLU(True))
        self.domain_classifier.add_module('dpt2', nn.Dropout())
        self.domain_classifier.add_module('fc3', nn.Linear(1024, 2))



    def forward(self, source, target, DEV, alpha=0.0):
        source_share = self.sharedNet(source)
        target_share = self.sharedNet(target)

        s_batch_size=source_share.size(0)
        t_batch_size=target_share.size(0)

        # attention
        s_residual = source_share
        source_share = self.conv1(source_share)
        source_share = self.bn1(source_share)
        source_share = self.relu(source_share)
        source_share_output=[]
        for i,conv in enumerate(self.conv2):
            source_share_output.append(conv(source_share))

        t_residual = target_share
        target_share = self.conv1(target_share)
        target_share = self.bn1(target_share)
        target_share = self.relu(target_share)
        target_share_output=[]
        for i,conv in enumerate(self.conv2):
            target_share_output.append(conv(target_share))

        # local block
        source_block_output = []
        target_block_output = []
        for i in range(self.M):
            source_block_output.append(self.localattens[i](source_share_output[i], target_share_output[i]))
            target_block_output.append(self.localattens[i](target_share_output[i], target_share_output[i]))
        

        source_share_U=reduce(lambda x,y:x+y,source_block_output)
        target_share_U=reduce(lambda x,y:x+y,target_block_output)

        s_s = self.global_pool(source_share_U)
        s_z = self.fc1(s_s)
        s_a_b = self.fc2(s_z)
        s_a_b = s_a_b.reshape(s_batch_size, self.M, 4096, -1)
        s_a_b = self.softmax(s_a_b)
        s_a_b = list(s_a_b.chunk(self.M, dim=1))
        s_a_b = list(map(lambda x:x.reshape(s_batch_size, 4096 ,1, 1), s_a_b))
        s_V = list(map(lambda x,y:x*y, source_block_output, s_a_b))
        source_share = reduce(lambda x,y:x+y, s_V)
        source_share = self.bn2(source_share)
        source_share = self.conv3(source_share)
        source_share = self.bn3(source_share)
        source_share += s_residual
        source_share = self.relu(source_share)
        source_share = self.avgpool(source_share)
        source_share = source_share.view(source_share.size(0), -1)
        source_share = self.bottleneck(source_share)
        source = self.source_fc(source_share)

        
        t_s = self.global_pool(target_share_U)
        t_z = self.fc1(t_s)
        t_a_b = self.fc2(t_z)
        t_a_b = t_a_b.reshape(t_batch_size, self.M, 4096, -1)
        t_a_b = self.softmax(t_a_b)
        t_a_b = list(t_a_b.chunk(self.M, dim=1))
        t_a_b = list(map(lambda x:x.reshape(t_batch_size, 4096 ,1, 1), t_a_b))
        t_V = list(map(lambda x,y:x*y, target_block_output, t_a_b))
        target_share = reduce(lambda x,y:x+y, t_V)
        target_share = self.bn2(target_share)
        target_share = self.conv3(target_share)
        target_share = self.bn3(target_share)
        target_share += t_residual
        target_share = self.relu(target_share)
        target_share = self.avgpool(target_share)
        target_share = target_share.view(target_share.size(0), -1)
        target_share = self.bottleneck(target_share)
        target = self.source_fc(target_share)


        if self.training == True:
            s_batch, _ = source_share.shape[:2]
            t_batch, _ = target_share.shape[:2]
            # RevGrad
            s_reverse_feature = ReverseLayerF.apply(source_share, alpha)
            t_reverse_feature = ReverseLayerF.apply(target_share, alpha)
            
            s_domain_output = self.domain_classifier(s_reverse_feature)
            s_domain_attention = F.softmax(s_domain_output, dim=1)
            s_g_mi = (1 + torch.sum(- s_domain_attention * torch.log(s_domain_attention), dim=1, keepdim=True)) / np.log(s_batch)
            s_g_mi = normalize_weight(s_g_mi)

            t_domain_output = self.domain_classifier(t_reverse_feature)
            t_domain_attention = F.softmax(t_domain_output, dim=1)
            t_g_mi = (1 + torch.sum(- t_domain_attention * torch.log(t_domain_attention), dim=1, keepdim=True)) / np.log(t_batch)
            t_g_mi = normalize_weight(t_g_mi)

        else:
            s_domain_output = 0
            t_domain_output = 0
            s_g_mi = 0
            t_g_mi = 0
        
        return source, target, s_domain_output, t_domain_output, s_g_mi, t_g_mi, source_share



def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    x = x / torch.mean(x)
    return x.detach()


if __name__ == '__main__':
    import torch

    DEVICE = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

    x = torch.rand(32, 3, 224, 224).to(DEVICE)
    y = torch.rand(32, 3, 224, 224).to(DEVICE)

    net = RGDA().to(DEVICE)
    out, _ = net(x, y, DEVICE)
    print(out.size())