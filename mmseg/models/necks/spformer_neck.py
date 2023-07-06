# --------------------------------------------------------
# Super Token Vision Transformer (STViT)
# Copyright (c) 2023 CASIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Huaibo Huang
# --------------------------------------------------------

from heapq import _heapify_max
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import scipy.io as sio
import torch.nn.functional as F
import math
from functools import partial
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import time
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms


from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner import CheckpointLoader
from mmengine.utils import to_2tuple

from mmseg.registry import MODELS

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(BaseModule):
    def forward(self, x):
        return SwishImplementation.apply(x)

class LayerNorm2d(BaseModule):
    def __init__(self, dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

class ResDWC(BaseModule):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
                
        self.shortcut = nn.Parameter(torch.eye(kernel_size).reshape(1, 1, kernel_size, kernel_size))
        self.shortcut.requires_grad = False
        
    def forward(self, x):
        return F.conv2d(x, self.conv.weight+self.shortcut, self.conv.bias, stride=1, padding=self.kernel_size//2, groups=self.dim) # equal to x + conv(x)
class Attention(BaseModule):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
                
        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5
                
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
                
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads *3, N).chunk(3, dim=2) # (B, num_heads, head_dim, N)
        
        attn = (k.transpose(-1, -2) @ q) * self.scale
        
        attn = attn.softmax(dim=-2) # (B, h, N, N)
        attn = self.attn_drop(attn)
        
        x = (v @ attn).reshape(B, C, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Unfold(BaseModule):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
        if torch.cuda.is_available():
            self.weights = nn.Parameter(self.weights.to('cuda'), requires_grad=False)

        
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
        if torch.cuda.is_available():
            self.weights = nn.Parameter(self.weights.to('cuda'), requires_grad=False)
           
        
    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size//2)        
        return x

class StokenAttention(BaseModule): 
    def __init__(self, dim, stoken_size, n_iter=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.n_iter = n_iter
        self.stoken_size = stoken_size
                
        self.scale = dim ** - 0.5
        
        self.unfold = Unfold(3)
        self.fold = Fold(3)
        
        self.query_s=nn.Linear(dim,dim)
        self.key_s=nn.Linear(dim,dim)
        self.value_s=nn.Linear(dim,dim)
        self.query_p=nn.Linear(dim,dim)
        self.key_p=nn.Linear(dim,dim)
        self.value_p=nn.Linear(dim,dim)
        self.num_heads=num_heads
        self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        
        # Initializing position embedding
        self.pos_embedding_stoken = nn.Parameter(torch.randn(1, dim, 512// stoken_size[0] // 4 ,512 // stoken_size[1] // 4 ))
        self.pos_embedding_pixel = nn.Parameter(torch.randn(1, dim, 512// 4 ,512 // 4 ))

    def stoken_forward(self, stoken_features,pixel_features):
        '''
           x: (B, C, H, W)
        '''
        x=pixel_features
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size
        
        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            
        _, _, H, W = x.shape
        
        hh, ww = H//h, W//w
        
        pos_embedding_stoken = F.interpolate(self.pos_embedding_stoken, size=(hh, ww), mode='bilinear', align_corners=False)
        pos_embedding_pixel =  F.interpolate(self.pos_embedding_stoken, size=(H, W), mode='bilinear', align_corners=False)
        
        stoken_features = stoken_features + pos_embedding_stoken
        pixel_features = pixel_features + pos_embedding_pixel
        for idx in range(self.n_iter):
            pixel_features = pixel_features.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh*ww, h*w, C)

            stoken_features_q=self.query_s(stoken_features.view(B,C,-1).permute(0, 2, 1)).view(B,hh*ww,1,C)
            stoken_features_k=self.key_s(stoken_features.view(B,C,-1).permute(0, 2, 1)).view(B,C,hh,ww)
            stoken_features_v=self.value_s(stoken_features.view(B,C,-1).permute(0, 2, 1)).view(B,C,hh,ww)
            
            pixel_features_q=self.query_p(pixel_features.view(B,C,-1).permute(0, 2, 1)).view(B, hh*ww, h*w, C)
            pixel_features_k=self.key_p(pixel_features.view(B,C,-1).permute(0, 2, 1)).view(B, C,H,W)
            pixel_features_v=self.value_p(pixel_features.view(B,C,-1).permute(0, 2, 1)).view(B, C,H,W)
            
            
            stoken_features_k = self.unfold(stoken_features_k) # (B, C*9, hh*ww)
            stoken_features_k = stoken_features_k.transpose(1, 2).reshape(B, hh*ww, C, 9)
                        
            stoken_features_v = self.unfold(stoken_features_v) # (B, C*9, hh*ww)
            stoken_features_v = stoken_features_v.transpose(1, 2).reshape(B, hh*ww, C, 9)

                
            affinity_matrix = pixel_features_q @ stoken_features_k * self.scale # (B, hh*ww, h*w, 9)
            affinity_matrix = affinity_matrix.softmax(-1) # (B, hh*ww, h*w, 9)
            #update pixel
            
            pixel_features_new = stoken_features_v @ affinity_matrix.transpose(-1, -2)
             
            pixel_features_new =pixel_features_new.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W) 
                
            pixel_features = pixel_features.reshape(B, hh, ww, h, w, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
            pixel_features = pixel_features+pixel_features_new
            

            #update superpixel
            
            pixel_features_k=self.unfold(pixel_features_k)# (B, C*9, h*w)
            pixel_features_k = pixel_features_k.transpose(1, 2).reshape(B, hh*ww, C, 9*h*w)

            pixel_features_v=self.unfold(pixel_features_v)# (B, C*9, h*w)
            pixel_features_v = pixel_features_v.transpose(1, 2).reshape(B, hh*ww, C, 9*h*w)
           
            att= (stoken_features_q @ pixel_features_k* self.scale).softmax(-1)
            
            stoken_features_new=pixel_features_v @ att.transpose(-1, -2)
            
            stoken_features_new=stoken_features_new.permute(0, 2, 1, 3).reshape(B, C, hh, ww)
            stoken_features+=stoken_features_new

        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]
        
        stoken_features = self.stoken_refine(stoken_features)
        return stoken_features,pixel_features
    
    
    def direct_forward(self, stoken_features,pixel_features):
        stoken_features = self.stoken_refine(stoken_features)        
        return stoken_features,pixel_features
        
    def forward(self, stoken_features,pixel_features):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(stoken_features,pixel_features)
        else:
            return self.direct_forward(stoken_features,pixel_features)

class StokenAttentionLayer(BaseModule):
    def __init__(self, dim, n_iter, stoken_size, 
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5):
        super().__init__()
                        
        self.layerscale = layerscale
        
        self.pos_embed = ResDWC(dim, 3)
                                        
        self.norm1 = LayerNorm2d(dim)
        self.attn = StokenAttention(dim, stoken_size=stoken_size, 
                                    n_iter=n_iter,                                     
                                    num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                    attn_drop=attn_drop, proj_drop=drop)   
                    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer, drop=drop)
                
        if layerscale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
        
    def forward(self,stoken_features,pixel_features):
            
        # x=self.pos_embed(x)
        
        stoken_features=self.norm1(stoken_features)
        pixel_features=self.norm1(pixel_features)
        stoken_features,pixel_features=self.attn(stoken_features,pixel_features)
              
        return stoken_features,pixel_features

class BasicLayer(BaseModule):        
    def __init__(self, num_layers, dim, n_iter, stoken_size, 
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5,
                 downsample=False,
                 use_checkpoint=False, checkpoint_num=None):
        super().__init__()        
                
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
                
        self.blocks = nn.ModuleList([StokenAttentionLayer(
                                           dim=dim[0],  n_iter=n_iter, stoken_size=stoken_size,                                           
                                           num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                           drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                           act_layer=act_layer, 
                                           layerscale=layerscale, init_values=init_values) for i in range(num_layers)])
                                           
                                                                           
                
        if downsample:            
            self.downsample = PatchMerging(dim[0], dim[1])
        else:
            self.downsample = None
        # self.downsample = None
         
    def forward(self, stoken_features,pixel_features):
        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint and idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x)
            else:
                stoken_features,pixel_features = blk(stoken_features,pixel_features)
        # if self.downsample is not None:
        #     x = self.downsample(x)
        return stoken_features,pixel_features
       


class PatchEmbed(BaseModule):        
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),
            
            nn.Conv2d(out_channels // 2, out_channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),
                        
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),            
            
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchMerging(BaseModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x

@MODELS.register_module()
class SpformerNeck(BaseModule):   
    def __init__(self, in_chans=3, num_classes=1000,
                 embed_dim=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 n_iter=[3, 2, 1, 0], stoken_size=[8, 4, 2, 1],                
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 projection=None, freeze_bn=False,
                 use_checkpoint=False, checkpoint_num=[0,0,0,0], 
                 layerscale=[False, False, False, False], init_values=1e-6, seg=False,downsample=True,load_from=None,**kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim        
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        self.stoken_size=stoken_size
        self.freeze_bn = freeze_bn

        self.patch_embed = PatchEmbed(in_chans, embed_dim[0])
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.downsample=downsample
        self.load_from=load_from
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
                
        # build layers
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            downsample_value = i_layer < self.num_layers - 1 if self.downsample else False
            layer = BasicLayer(num_layers=depths[i_layer],
                               dim=[embed_dim[i_layer], embed_dim[i_layer+1] if i_layer<self.num_layers-1 else None],                              
                               n_iter=n_iter[i_layer],
                               stoken_size=to_2tuple(stoken_size[i_layer]),                                                       
                               num_heads=num_heads[i_layer], 
                               mlp_ratio=self.mlp_ratio, 
                               qkv_bias=qkv_bias, qk_scale=qk_scale, 
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               downsample=downsample_value,
                               use_checkpoint=use_checkpoint,
                               checkpoint_num=checkpoint_num[i_layer],                               
                               layerscale=layerscale[i_layer],
                               init_values=init_values)
            self.layers.append(layer)
        
        self.proj = nn.Conv2d(self.num_features, projection, 1) if projection else None
        self.norm = nn.BatchNorm2d(projection)
        self.swish = MemoryEfficientSwish()        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.seg=seg
        self.classifier_head=nn.Linear(projection or self.num_features,num_classes)
        self.apply(self._init_weights)

     
   
                            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    


    def forward(self, x):
        
        x = self.patch_embed(x)        
        x = self.pos_drop(x)
        pixel_features=x
        B, C, H0, W0 = x.shape
        h = self.stoken_size[0]
        w= self.stoken_size[0]

        hh, ww = H0//h, W0//w
        
        #extract super pixel
        stoken_features = F.adaptive_avg_pool2d(x,(hh, ww))
        
        #iteratively update super pixel and pixel
        for i,layer in enumerate(self.layers):            
            stoken_features,pixel_features = layer(stoken_features,pixel_features)

        return stoken_features,pixel_features



