#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:37:12 2019

@author: aditya
"""

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    def forward(self, x):
        return x

model = Net()
inp = torch.randn(1,3,512,512)
out = model(inp)
print(out.shape)
