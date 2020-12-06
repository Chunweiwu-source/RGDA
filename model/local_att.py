import torch
from torch import nn
from torch.nn import functional as F


class DRAL(nn.Module):
    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super(DRAL, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )

        self.g = nn.Sequential(self.g, max_pool_layer)
        self.phi = nn.Sequential(self.phi, max_pool_layer)
    
    def forward(self, x, y):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_y = self.g(y).view(batch_size, self.inter_channels, -1)
        g_y = g_y.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_y = self.phi(y).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_y)
        f_div_C = F.softmax(f, dim=-1)

        z = torch.matmul(f_div_C, g_y)
        z = z.permute(0, 2, 1).contiguous()
        z = z.view(batch_size, self.inter_channels, *x.size()[2:])
        W_z = self.W(z)
        out = W_z + x

        return out
