"""
Author: Chenhongyi Yang
LePE attention References: https://github.com/microsoft/CSWin-Transformer
"""
from ast import Try
from calendar import c
from typing import Sequence
from numpy import block
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from einops import rearrange
import numpy as np
from mmcv.cnn import build_norm_layer, build_conv_layer, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, AdaptivePadding, build_dropout
from mmengine.model.weight_init import trunc_normal_
from timm.models import layers as timm_layers
from mmengine.model import BaseModule, ModuleList
from mmcv.cnn.bricks import DropPath
import math
from timm.models.vision_transformer import Block, _cfg
from functools import partial
import sys

from timm.layers import Mlp
import os.path as osp
from PIL import Image
import os
import h5py
from timm.models import layers as timm_layers
from torch.utils.tensorboard import SummaryWriter
import timm.models.layers as tml

class SE(nn.Module):
    """
    Squeeze and excitation block
    """

    def __init__(self,
                 inp,
                 oup,
                 expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """

        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FeatExtract(nn.Module):
    """
    Feature extraction block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self, dim, keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.keep_dim = keep_dim

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x

class FullAttnModule(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 q_project=True,
                 association_embedding = False,
                 keep_multihead = True):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.keep_multihead = keep_multihead
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias) if q_project else None
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        if self.keep_multihead:
            self.proj = nn.Sequential(nn.Linear(head_dim, out_dim // self.num_heads), nn.Dropout(proj_drop))
        else:
            self.proj = nn.Sequential(nn.Linear(dim, out_dim), nn.Dropout(proj_drop))
        self.proj_drop = nn.Dropout(proj_drop)
        self.association_embedding = association_embedding
        
    def forward(self, query, key, value, att_bias=None, attn_dict_list=None):
        bq, nq, cq = query.shape
        bk, nk, ck = key.shape
        bv, nv, cv = value.shape

        # [bq, nh, nq, cq//nh]
        if self.q_proj:
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=bq, n=nq, c=cq // self.num_heads)
        else:
            q = rearrange(query, 'b n (h c)-> b h n c', h=self.num_heads, b=bq, n=nq, c=cq // self.num_heads)
        # [bk, nh, nk, ck//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=bk, n=nk, c=ck // self.num_heads)
        # [bv, nh, nv, cv//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=bv, n=nv, c=cv // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if att_bias is not None:
            attn = attn + att_bias.unsqueeze(dim=1)
        if self.association_embedding:
            if len(attn_dict_list) >0:
                attn = attn + attn_dict_list[-1].transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (bq, self.num_heads, nq, nk)
        if isinstance(attn_dict_list,list):
            attn_dict_list.append(attn.transpose(-2, -1))
        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.keep_multihead:
            out = rearrange(attn @ v, 'b h n c -> b (n h) c', h=self.num_heads, b=bq, n=nq, c=cv // self.num_heads)
        else:
            out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=bq, n=nq, c=cv // self.num_heads)        
        out = self.proj(out)
        if self.keep_multihead:
            out = rearrange(out, 'b (n h) c ->  b n (h c)', h=self.num_heads, b=bq, n=nq, c=cv // self.num_heads)        
        out = self.proj_drop(out)
        return out,attn_dict_list
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Reweight(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.reweight = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (0.5 + torch.sigmoid(self.reweight))
class ReweightSigmoid(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.weight)


class GroupAttnBlock(nn.Module):
    def __init__(self,
                 group_embed_dims,
                 num_heads,
                 ffn_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 drop_path=0.,
                 proj_drop=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 key_is_query=False,
                 value_is_key=False,
                 with_cp=False,
                 ls_init_value = None,
                 identity:bool = True,
                 group_reweight_method:str = None,
                 use_ffn:bool = False,
                 concat: bool = False,
                ):
        super().__init__()

        self.with_cp = with_cp

        self.norm_query = norm_layer(group_embed_dims)

        if not key_is_query:
            self.norm_key = norm_layer(group_embed_dims)
        else:
            self.norm_key = None
        self.key_is_query = key_is_query
        
        if group_reweight_method is None:
            self.reweight = nn.Identity()
        elif group_reweight_method == 'reweight':
            self.reweight = Reweight()
        elif group_reweight_method == 'reweight_sigmoid':           
            self.reweight = ReweightSigmoid()
        self.writer = SummaryWriter()
        self.forward_counter = 0
        self.log_interval = 50        

        if not value_is_key:
            self.norm_value = norm_layer(group_embed_dims)
        else:
            self.norm_value = None
        self.value_is_key = value_is_key
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = FullAttnModule(
            group_embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            q_project=True,)
        self.identity = identity
        self.use_ffn = use_ffn
        if self.identity:
            self.ls = LayerScale(group_embed_dims, init_values=ls_init_value) if ls_init_value else nn.Identity()
            if self.use_ffn:
                self.norm2 = norm_layer(group_embed_dims)
                self.mlp = Mlp(
                    in_features=group_embed_dims,
                    hidden_features=int(group_embed_dims * ffn_ratio),
                    act_layer=act_layer,
                    drop=proj_drop,
                )
        self.concat = concat
        if self.concat:
            self.proj = nn.Linear(group_embed_dims * 2, group_embed_dims, bias=True)



    def forward(self, query, key, value, att_bias=None, attn_dict_list=None):
        def _inner_forward(query, key, value, att_bias,attn_dict_list = None):
            q = self.norm_query(query)
            k = q if self.key_is_query else self.norm_key(key)
            v = k if self.value_is_key else self.norm_value(value)
            new_x, attn_dict_list = self.attn(q, k, v, att_bias=att_bias, attn_dict_list = attn_dict_list)
            if self.concat:
                x = torch.cat((query,new_x), dim= -1)
                x = self.proj(x)
            if self.identity:
                if self.use_ffn is False:
                    x = self.reweight(query) + self.drop_path(self.ls(new_x))
                else:
                    x = self.reweight(query) + self.drop_path(self.ls(self.mlp((self.norm2(new_x)))))                   
            else:
                x = new_x
            self.forward_counter += 1
            if self.forward_counter % self.log_interval == 0:
                if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:                
                    if hasattr(self.reweight, 'reweight'):
                        param =  (0.5 + torch.sigmoid(self.reweight.reweight)).detach().cpu().numpy().astype(np.float32)
                        self.writer.add_scalar(f'reweight', param, self.forward_counter)                                    
                    elif hasattr(self.reweight, 'weight') : 
                        param =  (torch.sigmoid(self.reweight.weight)).detach().cpu().numpy().astype(np.float32)
                        self.writer.add_scalar(f'reweight', param, self.forward_counter)                    
                    
            return x, attn_dict_list


        if self.with_cp:
            return cp.checkpoint(_inner_forward, query, key, value, att_bias,attn_dict_list)
        else:
            return _inner_forward(query, key, value, att_bias,attn_dict_list)


class GPBlock(nn.Module):
    def __init__(self,
                 group_embed_dims,
                 group_block_depth,
                 num_group_heads,
                 num_ungroup_heads,
                 num_block_heads,
                 num_group_token,
                 ffn_ratio=4.,
                 qkv_bias=True,
                 group_qk_scale=None,
                 proj_drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 with_cp=False,
                 group_att_cfg=dict(),
                 ungroup_att_cfg=dict(),
                 group_projector =None,
                 zero_init_group_token = False,
                 association_embedding = False,
                 group_token_init_method:str = "learnable",
                 init_kernel_size:int = -1,   
                 init_stride : int = -1, 
                 group_ls_init_value = None,
                 ungroup_ls_init_value = 1e-5, 
                 block_init_values=None,
                 group_identity: bool =True,
                 ungroup_identity: bool =True,                                                  
                 gt_iter: int = 1,
                 vis_gt_eff: bool = False,
                 output_dir: str = None,
                 layer_num : int = None,
                 group_pe_method = None,
                 superpixel_shape = None, 
                 use_global_token: bool = False,  
                 log_interval: int = 50,
                 group_reweight_method: str = None,
                 group_projector_method = None,
                 concat: bool = False,
                 **kwargs):

        super().__init__()

        self.group_embed_dims = group_embed_dims
        self.num_group_token = num_group_token
        self.with_cp = with_cp
        self.group_token_init_method = group_token_init_method
        self.group_pe_method = group_pe_method
        self.init_stride = init_stride
        self.use_global_token = use_global_token
        if self.group_pe_method == "learnable":
            self.sp_pos_embed = self._create_pos_embed(
                group_embed_dims, superpixel_shape
            )
            self.gt_pos_embed = self._create_pos_embed(
                group_embed_dims, num_group_token
            )
        elif self.group_pe_method == "depthwise":
            self.sp_pos_embed = tml.create_conv2d(
                group_embed_dims, group_embed_dims, 3, bias=True, depthwise=True
            )
            self.gt_pos_embed = tml.create_conv2d(
                group_embed_dims,
                group_embed_dims,
                3,
                bias=True,
                depthwise=True,
            )
        else:
            raise ValueError()                        

        if  self.group_token_init_method =='learnable':
            self.group_token = nn.Parameter(torch.zeros(1, num_group_token, group_embed_dims))
        elif self.group_token_init_method == 'conv_avgpool':

            self.group_token_init = \
            nn.Sequential(  
                timm_layers.create_conv2d(group_embed_dims, group_embed_dims, 1, padding="same"),
                timm_layers.LayerNorm2d(group_embed_dims),
                nn.GELU(),
                nn.AvgPool2d(kernel_size=init_stride,stride=init_stride),                    
            )
        elif self.group_token_init_method == 'avgpool_conv':

            self.group_token_init = \
            nn.Sequential( 
                nn.AvgPool2d(kernel_size=init_stride,stride=init_stride),                            
                timm_layers.create_conv2d(group_embed_dims, group_embed_dims, 1, padding="same"),
                timm_layers.LayerNorm2d(group_embed_dims),
                nn.GELU(),            
            )            

        elif self.group_token_init_method == 'avgpool':
            if isinstance(init_kernel_size, tuple) and isinstance(init_stride, tuple):
                layers = []
                for k_size, stride in zip(init_kernel_size, init_stride):
                    layers.extend([
                        nn.AvgPool2d(kernel_size=init_stride,stride=init_stride),
                        timm_layers.LayerNorm2d(group_embed_dims),
                        nn.GELU(),

                    ])
                self.group_token_init = nn.Sequential(*layers)
            elif isinstance(init_kernel_size, int) and isinstance(init_stride, int):
                self.group_token_init = \
                nn.Sequential(         
                nn.AvgPool2d(kernel_size=init_stride,stride=init_stride),
                timm_layers.LayerNorm2d(group_embed_dims),
                nn.GELU(),
            )
            else:
                raise(NotImplementedError)
            
        elif self.group_token_init_method == 'depthwise':
                self.group_token_init = \
                nn.Sequential(                         
                timm_layers.create_conv2d(group_embed_dims, group_embed_dims, kernel_size = init_kernel_size, stride = init_stride, padding = "same",groups = group_embed_dims),
                timm_layers.LayerNorm2d(group_embed_dims),
                nn.GELU(),
                )
        elif self.group_token_init_method == 'conv':
            if isinstance(init_kernel_size, tuple) and isinstance(init_stride, tuple):
                layers = []
                for k_size, stride in zip(init_kernel_size, init_stride):
                    layers.extend([
                        timm_layers.create_conv2d(group_embed_dims, group_embed_dims, kernel_size=k_size, stride=stride, padding="same"),
                        timm_layers.LayerNorm2d(group_embed_dims),
                        nn.GELU()
                    ])
                self.group_token_init = nn.Sequential(*layers)
            elif isinstance(init_kernel_size, int) and isinstance(init_stride, int):
                self.group_token_init = \
                nn.Sequential(                         
                timm_layers.create_conv2d(group_embed_dims, group_embed_dims, kernel_size = init_kernel_size, stride = init_stride, padding = "same"),
                timm_layers.LayerNorm2d(group_embed_dims),
                nn.GELU(),
                )
            else:
                raise(NotImplementedError)
        elif self.group_token_init_method == 'conv_relu':
            if isinstance(init_kernel_size, int) and isinstance(init_stride, int):
                self.group_token_init = \
                nn.Sequential(                         
                timm_layers.create_conv2d(group_embed_dims, group_embed_dims, kernel_size = init_kernel_size, stride = init_stride, padding = "same"),
                timm_layers.LayerNorm2d(group_embed_dims),
                nn.ReLU(),
                )
            else:
                raise(ValueError)
        
        elif self.group_token_init_method == 'GCViT':
            self.group_token_init = \
            nn.Sequential(
                FeatExtract(group_embed_dims, keep_dim=False),
                FeatExtract(group_embed_dims, keep_dim=False),
            )
        elif self.group_token_init_method == 'from_feature':
            pass
        elif  self.group_token_init_method == 'from_global':
            pass
        else:
            raise(NotImplementedError)
        self.group_projector = group_projector
        self.group_projector_method = group_projector_method
        
        if not zero_init_group_token and self.group_token_init_method =='learnable':
                trunc_normal_(self.group_token, std=.02)
        # make group layer
        self.group_layers = nn.ModuleList()
        self.un_group_layers = nn.ModuleList()
        self.gt_attn = nn.ModuleList()
        self.pos_embeds =nn.ParameterList()
        _group_att_cfg = dict(
            group_embed_dims=group_embed_dims,
            num_heads=num_group_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=group_qk_scale,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=0.,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp,
            ls_init_value = group_ls_init_value,
            identity = group_identity,
            group_reweight_method = group_reweight_method)
        _group_att_cfg.update(group_att_cfg)
        _ungroup_att_cfg = dict(
            group_embed_dims=group_embed_dims,
            num_heads=num_ungroup_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=None,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp,
            ls_init_value = ungroup_ls_init_value,
            identity = ungroup_identity,
            group_reweight_method = group_reweight_method,
            concat = concat)
        _ungroup_att_cfg.update(ungroup_att_cfg)
        _block_cfg = dict(
            dim=group_embed_dims,
            mlp_ratio=4.0,
            num_heads = num_block_heads,
            qkv_bias = qkv_bias,
            drop_path=drop_path,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            init_values=block_init_values,
            )
        _block_cfg.update(_block_cfg)
        self.use_global_token = use_global_token 
        if self.use_global_token:
            self.global_projector = nn.Linear(group_embed_dims, group_embed_dims)
            self.writer = SummaryWriter()
            self.forward_counter = 0
            self.log_interval = log_interval            
        for i in range(gt_iter):
        
            pos_embed = nn.Parameter(torch.randn(1, num_group_token, group_embed_dims) * 0.02)      
            group_layer = GroupAttnBlock(**_group_att_cfg)        
            un_group_layer = GroupAttnBlock(**_ungroup_att_cfg)
            blocks = nn.Sequential(
                *[
                    Block(
                        **_block_cfg
                    )
                    for i in range(group_block_depth)
                ]
            )
            self.pos_embeds.append(pos_embed)
            self.group_layers.append(group_layer)
            self.un_group_layers.append(un_group_layer)
            self.gt_attn.append(blocks)
            
        self.vis_gt_eff = vis_gt_eff
        self.output_dir = output_dir
        self.layer_num = layer_num

        
        self.init_weights()
    def init_weights(self):
        if self.pos_embeds is not None:
            for pos_embed in self.pos_embeds:
                timm_layers.trunc_normal_(pos_embed, std=0.02)
    def _create_pos_embed(self, dim, shape,) -> torch.Tensor:
        if isinstance(shape, list) :
            pos_embed = nn.Parameter(torch.zeros(1, shape[0]*shape[1],dim))
                                  
        elif isinstance(shape, int):
            pos_embed_shape = shape
            pos_embed = nn.Parameter(torch.zeros(1, shape,dim ))
        else:
            raise(TypeError)
        return pos_embed
    
    def forward(self, x, hw_shape, attn_dict_list = None,prev_token = None,global_token = None):
        """
        Args:
            x: image tokens, shape [B, L, C]
            hw_shape: tuple or list (H, W)
        Returns:
            proj_tokens: shape [B, L, C]
        """
        if self.use_global_token:
            self.forward_counter += 1
            if self.forward_counter % self.log_interval == 0 and dist.get_rank() == 0 and self.training:
                param =  (0.5 + torch.sigmoid(self.reweight.reweight)).detach().cpu().numpy().astype(np.float32)
                self.writer.add_scalar(f'global_token_reweight', param, self.forward_counter)                    
                        
        B, L, C = x.size()      
        sw = sh = int(math.sqrt(L))
        if self.vis_gt_eff:
            sp_before = x.clone()

        if self.group_token_init_method in["avgpool",'conv_avgpool','conv','GCViT','conv_relu','avgpool_conv','depthwise']:
            x = rearrange(x,
                        'b (h w) c -> b c h w',
                        h=sh, w=sw)
            group_token = self.group_token_init(x)
            if self.group_pe_method == 'depthwise':
                group_token = group_token + self.gt_pos_embed(group_token)
                x = x + self.sp_pos_embed(x)

            group_token = rearrange(
                group_token,
                'b c h w -> b (h w) c'
            )
            x = rearrange(x,
                'b c h w -> b (h w) c ',
                h=sh, w=sw)
            if self.group_pe_method == 'learnable':
                group_token = group_token + self.gt_pos_embed
                x = x + self.sp_pos_embed   
        elif self.group_token_init_method == "learnable":
            gt = self.group_token.expand(x.size(0), -1, -1)
            if self.group_pe_method == 'learnable':
                gt = gt + self.gt_pos_embed
                x = x + self.sp_pos_embed
        elif self.group_token_init_method == "from_feature":
            gt = prev_token
        elif self.group_token_init_method == "from_global":
            gt = global_token
        # if prev_token is None:
        #     gt = group_token
        # elif self.group_projector_method in ["linear","conv",'conv_relu']:
        #     gt = group_token + self.group_projector(prev_token)
        # elif self.group_projector_method == "cross" or self.group_projector_method == None:
        #     gt = group_token 
        
        # else:
        #     raise(NotImplementedError)
        
        if global_token != None and self.use_global_token:
            if self.group_token_init_method != "from_global":
                gt = gt + self.global_projector(global_token)
              

                 
        for i, (group_layer, pos_embed, un_group_layer, blocks) in enumerate(
            zip(self.group_layers, self.pos_embeds, self.un_group_layers, self.gt_attn if self.gt_attn else [None]*len(self.group_layers))
        ):
            if self.group_projector_method == "cross" and prev_token is not None:
                gt, _= self.group_projector(query=gt, key=prev_token, value=prev_token)            
            gt, _ = group_layer(query=gt, key=x, value=x, attn_dict_list = None)
            if len(blocks) > 0 :           
                gt = gt + pos_embed

                gt = blocks(gt)
            
         
            proj_tokens, attn_dict_list = un_group_layer(query=x, key=gt, value=gt, attn_dict_list = attn_dict_list)
        if self.vis_gt_eff:
            self.visualize_gt_eff(sp_before = sp_before, sp_after = proj_tokens, gt = gt)
        return proj_tokens, attn_dict_list, gt
            
    def reshape_vis(self, tensor):
        if tensor.dim() == 3:
            B, L, C = tensor.size()
            sw = sh = int(math.sqrt(L))
            tensor = tensor.detach().cpu().numpy()
            tensor = rearrange(tensor,
                            'b (h w) c -> b c h w ',
                            h=sh, w=sw)
        return tensor
    def write_vis(self, tensor,output_file,key):
        with h5py.File(output_file, "a") as f:
            keys = list(f.keys())
            original_key = key
            count = int(0)
            while key in keys:
                count += 1
                key = original_key + str(count)
            f.create_dataset(key, data=tensor)        
    def visualize_gt_eff(self, sp_before, sp_after, gt):
            sp_before =self.reshape_vis(sp_before)
            sp_after = self.reshape_vis(sp_after)
            sp_diff = sp_after - sp_before
            gt = self.reshape_vis(gt)         
               
            # Determine the highest count in the directory.
            vis_gt_eff_path = osp.join(self.output_dir, 'vis_gt_eff')
            if not os.path.exists(vis_gt_eff_path):
                os.makedirs(vis_gt_eff_path)
            existing_files = [f for f in os.listdir(vis_gt_eff_path) if f.startswith('gt') and f.endswith('.h5')]
            counts = [int(f.split('gt')[1].split('.h5')[0]) for f in existing_files]
            max_count = max(counts, default=0)  # get the maximum count or set it to 0 if the folder is empty

            output_file = osp.join(self.output_dir, 'vis_gt_eff', f'gt{max_count}.h5')  

            keys = []
            with h5py.File(output_file, "a") as f:
                keys = list(f.keys())
            if len(keys) >= self.layer_num:
                max_count += 1
                output_file = osp.join(self.output_dir, 'vis_gt_eff', f'gt{max_count}.h5') 
                f.close()
                
            self.write_vis(sp_before,output_file,'sp_before')                
            self.write_vis(sp_after,output_file,'sp_after')                
            self.write_vis(gt,output_file,'gt')                
            self.write_vis(sp_diff,output_file,'sp_diff')                


