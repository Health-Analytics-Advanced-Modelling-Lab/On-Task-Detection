#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import facial_data_process
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils import weight_norm

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
      if hasattr(layer, 'reset_parameters'):
          print(f'Reset trainable parameters of layer = {layer}')
          layer.reset_parameters()
    
def reset_weights_vgg(alex_net):    
    for layer in alex_net.net.classifier:
        if type(layer) == nn.Linear:
            layer.reset_parameters()
            print(f'Reset trainable parameters of layer = {layer}')
            
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

        
class alexnet(nn.Module):

    def __init__(self, num_classes=1) -> None:
        super(alexnet, self).__init__()
        self.net = models.vgg11(pretrained=True)
#        self.fc_features = nn.Sequential(*list(self.net.children())[-1])
        for p in self.net.features.parameters():
            p.requires_grad=False
        for p in self.net.avgpool.parameters():
            p.requires_grad=False
        # for p in self.net.classifier[:-5].parameters():
        #     p.requires_grad=False        
        self.net.classifier[-1] = nn.Linear(4096, num_classes)
        
       
    def forward(self,input):
        x0 = self.net(input)
        x = torch.sigmoid(x0)
        return x
        


