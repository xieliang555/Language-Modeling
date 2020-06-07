import torch
import torch.nn as nn
from torch.autograd import Variable


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, drop_ratio):
        # x: [T, N, E]
        # turn off dropout during evaluation or non-drop mode
        if not self.training or not drop_ratio:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1-drop_ratio)
        mask = Variable(m, requires_grad=False)/(1-drop_ratio)
        mask = mask.expand_as(x)
        return mask * x
    
    