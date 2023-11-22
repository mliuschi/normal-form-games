##############################
## Partly adapted from unofficial implementation of Deep Learning for Predicting Human Strategic Behavior (NeurIPS 2016)
## https://github.com/zudi-lin/human_behavior_prediction/
##############################

import torch
import torch.nn as nn
import torch.nn.functional as F

class GameModelPooling(nn.Module):
    def __init__(self, in_planes=2, out_planes=1, num_layers=4, kernels=8, mode='max_pool', 
                 bias=True, residual=False, activation='relu'):
        super().__init__()

        assert mode in ['max_pool', 'avg_pool']
        self.mode = mode
        self.residual = residual
        self.num_layers = num_layers
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        else:
            raise NotImplementedError("Activation not implemented!")

        if num_layers == 1:
            kernels = out_planes

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                conv_layer = nn.Conv2d(in_planes, kernels, kernel_size=1, bias=bias)
            elif i == num_layers - 1:
                conv_layer = nn.Conv2d(3*kernels, out_planes, kernel_size=1, bias=bias) # kernels * 3 for the pooled concatenated features
            else:
                conv_layer = nn.Conv2d(3*kernels, kernels, kernel_size=1, bias=bias)
            self.convs.append(conv_layer)

        # normalization layers
        self.bn = torch.nn.ModuleList([nn.BatchNorm2d(kernels) for i in range(num_layers - 1)])

        # weights initialization
        self._init_weight()

    def _forward_residual(self, x):
        residual = 0.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                x = self.activation(self.bn[i](self.convs[i](x)) + residual)
                residual = x
                x = self.pool_and_cat(x)
            else:
                x = self.convs[i](x)
        return x

    def _forward_plain(self, x):
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                x = self.activation(self.bn[i](self.convs[i](x)))
                x = self.pool_and_cat(x)
            else:
                x = self.convs[i](x)
        return x

    def forward(self, x):
        if self.residual:
            x = self._forward_residual(x)
        else:
            x = self._forward_plain(x)

        B, C, H, W = x.size()

        # run softmax and get marginal distribution for row player
        x = F.max_pool2d(x, kernel_size=(1,W)) # max pool across row player
        x = x.view(B,C,-1) # shape (B, C, H)
        x = F.softmax(x, dim=-1) 
        #x = x.view(B, C, H, W)
        return x

    def pool_and_cat(self, x):
        B, C, H, W = x.size()

        if self.mode == 'max_pool':
            x_row = F.max_pool2d(x, kernel_size=(1,W))
            x_col = F.max_pool2d(x, kernel_size=(H,1))
        else:
            x_row = F.avg_pool2d(x, kernel_size=(1,W))
            x_col = F.avg_pool2d(x, kernel_size=(H,1))
        x_row = x_row.expand(x.size())
        x_col = x_row.expand(x.size())
        y = torch.cat((x, x_row, x_col), dim=1)
        return y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
