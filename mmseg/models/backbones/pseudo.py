import torch.nn as nn

from ..builder import BACKBONES


@BACKBONES.register_module()
class pseudo(nn.Module):

    def __init__(self,**kwargs):
        super(pseudo, self).__init__(**kwargs)
        
    def forward(self, x):  # should return a tuple
        return x

    def init_weights(self, pretrained=None):
        pass