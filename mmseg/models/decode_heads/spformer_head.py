from turtle import forward
import torch.nn as nn
from mmcv.cnn import ConvModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from mmengine.model import BaseModule

import torch
import torch.nn.functional as F
class Unfold(BaseModule):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b*c, 1, h, w), self.weights, stride=1, padding=self.kernel_size//2)        
        return x.reshape(b, c*9, h*w)

class Fold(BaseModule):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size//2)        
        return x

@MODELS.register_module()
class SpformerHead(BaseDecodeHead):
    def __init__(self,stoken_size,in_channels,channels,num_classes,load_from=None,**kwargs):
        super().__init__(in_channels=in_channels,channels=channels,num_classes=num_classes)
        self.unfold = Unfold(3)
        self.fold = Fold(3)
        self.stoken_size=stoken_size
        self.classifier_head=nn.Linear(in_channels,num_classes)
        self.load_from=load_from
        self.unfold = Unfold(3)
        self.fold = Fold(3)

    
        
    def forward(self,stoken_features, pixel_features):
        
        x = pixel_features
        B, C, H0, W0 = x.shape
        h, w = to_2tuple(self.stoken_size[-1])
        
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            
        _, _, H, W = x.shape
        
        hh, ww = H // h, W // w
        
        #get class_predictions and reshape
        class_predictions = self.classifier_head(stoken_features.permute(0, 2, 3, 1)) # (B, hh, ww, n_classes)
        class_predictions = class_predictions.permute(0,3,1,2)
        class_predictions = self.unfold(class_predictions)#(B, n_classes*9, hh, ww)
        class_predictions = class_predictions.view(B*9,-1,hh,ww)#(B*9, n_classes, hh, ww)
        class_predictions = F.interpolate(class_predictions, scale_factor=8, mode='bilinear', align_corners=False)#(B*9, n_classes, hh*8, ww*8)
        class_predictions = class_predictions.view(B,9,-1,hh*8,ww*8).permute(0,2,3,4,1)#(B, n_classes, hh*8, ww*8, 9)
        
        #get affinity_matrix
        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh*ww, h*w, C)
        stoken_features = self.unfold(stoken_features) # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh*ww, C, 9)
        
        affinity_matrix = pixel_features @ stoken_features * C**-0.5 # (B, hh*ww, h*w, 9)
        
        # Upsample the affinity matrix by factor of 8 using bilinear interpolation
        
        affinity_matrix = affinity_matrix.permute(0, 3, 2, 1).reshape(B*9,h*w,hh,ww)#(B,9, h*w,hh*ww)
        affinity_matrix = F.interpolate(affinity_matrix, scale_factor=8, mode='bilinear', align_corners=False)#(B,9, h*w,hh*8,ww*8)
        affinity_matrix = affinity_matrix.permute(0, 2, 3, 1).softmax(-1)# (B*9, h*w, hh*8, ww*8)
        affinity_matrix = affinity_matrix.view(B,9,h*w,hh*8,ww*8).permute(0,2,3,4,1) # (B, h*w, hh*8, ww*8,9)
        
        #affinity_matrix is (B, h*w, hh*8, ww*8,9),h*w is the pixel number a superpixel convered,hh and ww is the number of superpixel
        #class_predictions is (B,n_classes, hh, ww,9) 

        # Reshape affinity_matrix to match class_predictions
        affinity_matrix = affinity_matrix.view(B, h*w, hh*8, ww*8,9) # (B, h*w, hh*8, ww*8,9)
        
        # class_predictions: (B, n_classes, hh*8, ww*8, 9) -> (B, 1, n_classes, hh*8, ww*8, 9)      
        class_predictions = class_predictions.unsqueeze(1)
        
        # affinity_matrix: (B, h*w, hh*8, ww*8, 9) -> (B, h*w, 1, hh*8, ww*8, 9)
        affinity_matrix = affinity_matrix.unsqueeze(2)

        # Apply affinity_matrix to class_predictions and sum over the last dimension
        Y = (affinity_matrix * class_predictions).sum(dim=-1)

        # Reshape Y to new spatial resolution
        Y = Y.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, hh*8*h, ww*8*w)

        return Y
