"""
Author: Chenhongyi Yang
LePE attention References: https://github.com/microsoft/CSWin-Transformer
"""
from ast import Try
from typing import Sequence
from numpy import block

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from einops import rearrange

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
# sys.path.append("/root/autodl-tmp/GroupViT/models")
import os.path as osp
from PIL import Image
import os
import h5py

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
    
class MLPMixerLayer(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dims,
                 patch_expansion,
                 channel_expansion,
                 drop_path,
                 drop_out,
                 **kwargs):

        super(MLPMixerLayer, self).__init__()

        patch_mix_dims = int(patch_expansion * embed_dims)
        channel_mix_dims = int(channel_expansion * embed_dims)

        self.patch_mixer = nn.Sequential(
            nn.Linear(num_patches, patch_mix_dims),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(patch_mix_dims, num_patches),
            nn.Dropout(drop_out)
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dims, channel_mix_dims),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(channel_mix_dims, embed_dims),
            nn.Dropout(drop_out)
        )

        self.drop_path1 = build_dropout(dict(type='DropPath', drop_prob=drop_path))
        self.drop_path2 = build_dropout(dict(type='DropPath', drop_prob=drop_path))

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        x = x + self.drop_path1(self.patch_mixer(self.norm1(x).transpose(1,2)).transpose(1,2))
        x = x + self.drop_path2(self.channel_mixer(self.norm2(x)))
        return x

class MLPMixer(BaseModule):
    def __init__(self,
                 num_patches,
                 embed_dims,
                 patch_expansion=0.5,
                 channel_expansion=4.0,
                 depth=1,
                 drop_path=0.,
                 drop_out=0.,
                 init_cfg=None,
                 **kwargs):
        super(MLPMixer, self).__init__(init_cfg)
        layers = [
            MLPMixerLayer(num_patches, embed_dims, patch_expansion, channel_expansion, drop_path, drop_out)
            for _ in range(depth)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class AttentionWithRelPos(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 attn_map_dim=None, num_cls_tokens=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.num_cls_tokens = num_cls_tokens
        if attn_map_dim is not None:
            one_dim = attn_map_dim[0]
            rel_pos_dim = (2 * one_dim - 1)
            self.rel_pos = nn.Parameter(torch.zeros(num_heads, rel_pos_dim ** 2))
            tmp = torch.arange(rel_pos_dim ** 2).reshape((rel_pos_dim, rel_pos_dim))
            out = []
            offset_x = offset_y = one_dim // 2
            for y in range(one_dim):
                for x in range(one_dim):
                    for dy in range(one_dim):
                        for dx in range(one_dim):
                            out.append(tmp[dy - y + offset_y, dx - x + offset_x])
            self.rel_pos_index = torch.tensor(out, dtype=torch.long)
            trunc_normal_(self.rel_pos, std=.02)
        else:
            self.rel_pos = None

    def forward(self, x, patch_attn=False, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rel_pos is not None and patch_attn:
            # use for the indicating patch + cls:
            rel_pos = self.rel_pos[:, self.rel_pos_index.to(attn.device)].reshape(self.num_heads, N - self.num_cls_tokens, N - self.num_cls_tokens)
            attn[:, :, self.num_cls_tokens:, self.num_cls_tokens:] = attn[:, :, self.num_cls_tokens:, self.num_cls_tokens:] + rel_pos

        if mask is not None:
            ## mask is only (BH_sW_s)(ksks)(ksks), need to expand it
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn = attn.masked_fill(mask == 0, torch.finfo(attn.dtype).min)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LightAttModule(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 q_project=True,
                 k_project=True,
                 v_project=True,
                 proj_after_att=True):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias) if q_project else None
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias) if k_project else None
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias) if v_project else None

        self.attn_drop = nn.Dropout(attn_drop)

        if proj_after_att:
            self.proj = nn.Sequential(nn.Linear(dim, out_dim), nn.Dropout(proj_drop))
        else:
            self.proj = None

    def forward(self, query, key, value, att_bias=None,attn_dict_list=None):
        bq, nq, cq = query.shape
        bk, nk, ck = key.shape
        bv, nv, cv = value.shape

        # [bq, nh, nq, cq//nh]
        if self.q_proj:
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=bq, n=nq, c=cq // self.num_heads)
        else:
            q = rearrange(query, 'b n (h c)-> b h n c', h=self.num_heads, b=bq, n=nq, c=cq // self.num_heads)
        # [bk, nh, nk, ck//nh]
        if self.k_proj:
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=bk, n=nk, c=ck // self.num_heads)
        else:
            k = rearrange(key, 'b n (h c)-> b h n c', h=self.num_heads, b=bk, n=nk, c=ck // self.num_heads)
        # [bv, nh, nv, cv//nh]
        if self.v_proj:
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=bv, n=nv, c=cv // self.num_heads)
        else:
            v = rearrange(value, 'b n (h c)-> b h n c', h=self.num_heads, b=bv, n=nv, c=cv // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if att_bias is not None:
            attn = attn + att_bias.unsqueeze(dim=1)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if isinstance(attn_dict_list,list):
            attn_dict_list.append(attn)
        assert attn.shape == (bq, self.num_heads, nq, nk)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=bq, n=nq, c=cv // self.num_heads)
        if self.proj:
            out = self.proj(out)
        return out,attn_dict_list


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
                 association_embedding = False):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias) if q_project else None
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
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
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=bq, n=nq, c=cv // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out,attn_dict_list

class FullAttnCatBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 ffn_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 proj_drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 key_is_query=False,
                 value_is_key=False,
                 q_project=True,
                 with_cp=False,
                 association_embedding = False,
                 ls_init_value = 1e-5,
                 
                 **kwargs):
        super().__init__()
        self.with_cp = with_cp

        self.norm_query = build_norm_layer(norm_cfg, embed_dims)[1]

        if not key_is_query:
            self.norm_key = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm_key = None
        self.key_is_query = key_is_query

        if not value_is_key:
            self.norm_value = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm_value = None
        self.value_is_key = value_is_key

        self.attn = FullAttnModule(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            q_project=q_project,
            association_embedding = association_embedding)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        _ffn_cfgs = {
            'embed_dims': embed_dims,
            'feedforward_channels': int(embed_dims * ffn_ratio),
            'num_fcs': 2,
            'ffn_drop': proj_drop,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'act_cfg': act_cfg,
            'layer_scale_init_value':ls_init_value
        }

        self.ffn = FFN(**_ffn_cfgs)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.proj = nn.Linear(embed_dims * 2, embed_dims, bias=True)

    def forward(self, query, key, value, att_bias=None,attn_dict_list = None):
        def _inner_forward(query, key, value, att_bias,attn_dict_list):
            
            q = self.norm_query(query)
            k = q if self.key_is_query else self.norm_key(key)
            v = k if self.value_is_key else self.norm_value(value)
           
            x, attn_dict_list = self.attn(q, k, v, att_bias=att_bias,attn_dict_list = attn_dict_list)
            x = torch.cat((query, self.drop_path(x)),dim=-1)
            x = self.proj(x)
            x = self.ffn(self.norm2(x), identity=x) 
                
            return x,attn_dict_list
        if self.with_cp:
            return cp.checkpoint(_inner_forward, query, key, value, att_bias)
        else:
            return _inner_forward(query, key, value, att_bias,attn_dict_list)

class LightGroupAttnBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 ffn_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 key_is_query=False,
                 value_is_key=False,
                 with_cp=False,
                 ls_init_value = None):
        super().__init__()

        self.with_cp = with_cp

        self.norm_query = build_norm_layer(norm_cfg, embed_dims)[1]

        if not key_is_query:
            self.norm_key = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm_key = None
        self.key_is_query = key_is_query

        if not value_is_key:
            self.norm_value = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm_value = None
        self.value_is_key = value_is_key

        self.attn = LightAttModule(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            q_project=True,
            k_project=True,
            v_project=True,
            proj_after_att=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, key, value, att_bias=None, attn_dict_list=None):
        def _inner_forward(query, key, value, att_bias,attn_dict_list = None):
            q = self.norm_query(query)
            k = q if self.key_is_query else self.norm_key(key)
            v = k if self.value_is_key else self.norm_value(value)
            x, attn_dict_list = self.attn(q, k, v, att_bias=att_bias, attn_dict_list = attn_dict_list)
            x = self.drop_path(x)
            return x, attn_dict_list


        if self.with_cp:
            return cp.checkpoint(_inner_forward, query, key, value, att_bias,attn_dict_list)
        else:
            return _inner_forward(query, key, value, att_bias,attn_dict_list)
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class FullGroupAttnBlock(nn.Module):
    def __init__(self,
                 embed_dims,
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
                 ls_init_value = None):
        super().__init__()

        self.with_cp = with_cp

        self.norm_query = norm_layer(embed_dims)

        if not key_is_query:
            self.norm_key = norm_layer(embed_dims)
        else:
            self.norm_key = None
        self.key_is_query = key_is_query

        if not value_is_key:
            self.norm_value = norm_layer(embed_dims)
        else:
            self.norm_value = None
        self.norm2 = norm_layer(embed_dims)
        self.value_is_key = value_is_key
        self.ls1 = LayerScale(embed_dims, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.mlp = Mlp(
            in_features=embed_dims,
            hidden_features=int(embed_dims * ffn_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(embed_dims, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = LightAttModule(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            q_project=True,
            k_project=True,
            v_project=True,
            proj_after_att=True)


    def forward(self, query, key, value, att_bias=None, attn_dict_list=None):
        def _inner_forward(query, key, value, att_bias,attn_dict_list = None):
            q = self.norm_query(query)
            k = q if self.key_is_query else self.norm_key(key)
            v = k if self.value_is_key else self.norm_value(value)
            x, attn_dict_list = self.attn(q, k, v, att_bias=att_bias, attn_dict_list = attn_dict_list)
            x = x + self.drop_path1(self.ls1(x))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x, attn_dict_list


        if self.with_cp:
            return cp.checkpoint(_inner_forward, query, key, value, att_bias,attn_dict_list)
        else:
            return _inner_forward(query, key, value, att_bias,attn_dict_list)


class GPBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 depth,
                 num_group_heads,
                 num_ungroup_heads,
                 num_group_token,
                 ffn_ratio=4.,
                 qkv_bias=True,
                 group_qk_scale=None,
                 proj_drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 with_cp=False,
                 group_att_cfg=dict(),
                 fwd_att_cfg=dict(),
                 ungroup_att_cfg=dict(),
                 group_projector =None,
                 group_projector_methonds ="linear",
                 zero_init_group_token = False,
                 association_embedding = False,
                 group_token_init_method:str = "learnable",
                 init_kernel_size:int = -1,   
                 init_stride : int = -1, 
                 use_assign: bool = False,
                 ls_init_value = 1e-5,
                 gt_iter: int = 1,
                 vis_gt_eff: bool = False,
                 output_dir: str = None,
                 layer_num : int = None,
                 same_group_method: bool = False,                                  
                 **kwargs):

        super().__init__()

        self.embed_dims = embed_dims
        self.num_group_token = num_group_token
        self.with_cp = with_cp
        self.group_token_init_method = group_token_init_method
        
        if  self.group_token_init_method =='learnable':
            self.group_token = nn.Parameter(torch.zeros(1, num_group_token, embed_dims))
        elif self.group_token_init_method == 'conv_avgpool':
            from timm.models import layers as timm_layers

            self.group_token_init = \
            nn.Sequential(  
                timm_layers.create_conv2d(embed_dims, embed_dims, 1, padding="same"),
                timm_layers.LayerNorm2d(embed_dims),
                nn.GELU(),
                nn.AvgPool2d(kernel_size=init_stride,stride=init_stride),                    
            )
        elif self.group_token_init_method == 'avgpool_conv':
            from timm.models import layers as timm_layers

            self.group_token_init = \
            nn.Sequential( 
                nn.AvgPool2d(kernel_size=init_stride,stride=init_stride),                            
                timm_layers.create_conv2d(embed_dims, embed_dims, 1, padding="same"),
                timm_layers.LayerNorm2d(embed_dims),
                nn.GELU(),            
            )            
        elif self.group_token_init_method == 'avgpool':
            from timm.models import layers as timm_layers

            self.group_token_init = \
            nn.Sequential(         
                nn.AvgPool2d(kernel_size=init_stride,stride=init_stride),
                timm_layers.LayerNorm2d(embed_dims),
                nn.GELU(),

            )
        elif self.group_token_init_method == 'conv':
            from timm.models import layers as timm_layers
            if isinstance(init_kernel_size, tuple) and isinstance(init_stride, tuple):
                layers = []
                for k_size, stride in zip(init_kernel_size, init_stride):
                    layers.extend([
                        timm_layers.create_conv2d(embed_dims, embed_dims, kernel_size=k_size, stride=stride, padding="same"),
                        timm_layers.LayerNorm2d(embed_dims),
                        nn.GELU()
                    ])
                self.group_token_init = nn.Sequential(*layers)
            elif isinstance(init_kernel_size, int) and isinstance(init_stride, int):
                self.group_token_init = \
                nn.Sequential(                         
                timm_layers.create_conv2d(embed_dims, embed_dims, kernel_size = init_kernel_size, stride = init_stride, padding = "same"),
                timm_layers.LayerNorm2d(embed_dims),
                nn.GELU(),
                )
            else:
                raise(NotImplementedError)
        elif self.group_token_init_method == 'conv_relu':
            from timm.models import layers as timm_layers
            if isinstance(init_kernel_size, int) and isinstance(init_stride, int):
                self.group_token_init = \
                nn.Sequential(                         
                timm_layers.create_conv2d(embed_dims, embed_dims, kernel_size = init_kernel_size, stride = init_stride, padding = "same"),
                timm_layers.LayerNorm2d(embed_dims),
                nn.ReLU(),
                )
            else:
                raise(ValueError)
            
        elif self.group_token_init_method == 'GCViT':
            self.group_token_init = \
            nn.Sequential(
                FeatExtract(embed_dims, keep_dim=False),
                FeatExtract(embed_dims, keep_dim=False),
            )
        else:
            raise(NotImplementedError)
        self.group_projector = group_projector
        self.group_projector_methonds = group_projector_methonds
        if not zero_init_group_token:
                trunc_normal_(self.group_token, std=.02)
        
        self.group_layers = nn.ModuleList()
        self.un_group_layers = nn.ModuleList()
        self.gt_attn = nn.ModuleList()
        self.pos_embeds =nn.ParameterList()
        _group_att_cfg = dict(
            embed_dims=embed_dims,
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
            ls_init_value = None)
        _group_att_cfg.update(group_att_cfg)
        _ungroup_att_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_ungroup_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=None,
            drop=proj_drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp,
            ls_init_value = ls_init_value,)
        _ungroup_att_cfg.update(ungroup_att_cfg)
        _mixer_cfg = dict(
            num_patches=num_group_token,
            embed_dims=embed_dims,
            patch_expansion=0.5,
            channel_expansion=4.0,
            depth=depth,
            drop_path=drop_path)
        _mixer_cfg.update(fwd_att_cfg)
        
        for i in range(gt_iter):
        
            pos_embed = nn.Parameter(torch.randn(1, num_group_token, embed_dims) * 0.02)      
            if same_group_method:
                group_layer = FullAttnCatBlock(**_group_att_cfg)        
            else:        
                group_layer = FullGroupAttnBlock(**_group_att_cfg)
            un_group_layer = FullAttnCatBlock(**_ungroup_att_cfg)
            blocks = nn.Sequential(
                *[
                    Block(
                        dim=embed_dims,
                        num_heads=2,
                        mlp_ratio=4,
                        qkv_bias=True,
                        proj_drop=proj_drop,
                        attn_drop=attn_drop,
                        drop_path=(
                            drop_path[i]
                            if isinstance(drop_path, Sequence)
                            else drop_path
                        ),
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        act_layer=nn.GELU,
                        init_values=1e-6,
                    )
                    for i in range(_mixer_cfg['depth'])
                ]
            )
            self.pos_embeds.append(pos_embed)
            self.group_layers.append(group_layer)
            self.un_group_layers.append(un_group_layer)
            self.gt_attn.append(blocks)
        # self.use_assign = use_assign
        # if self.use_assign:
        #     self.pre_assign_attn = CrossAttnBlock(
        #         dim=embed_dims, num_heads=num_ungroup_heads, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm, post_norm=True)
        #     self.assign = AssignAttention(
        #     dim=embed_dims,
        #     num_heads=1,
        #     qkv_bias=True,
        #     hard=True,
        #     gumbel=True,
        #     gumbel_tau=1.,
        #     sum_assign=False,
        #     assign_eps=1.)
        # else:
        self.vis_gt_eff = vis_gt_eff
        self.output_dir = output_dir
        self.layer_num = layer_num
        self.init_weights()
    def init_weights(self):
        if self.pos_embeds is not None:
            for pos_embed in self.pos_embeds:
                timm_layers.trunc_normal_(pos_embed, std=0.02)
       
    def forward(self, x, hw_shape, attn_dict_list = None,prev_token = None):
        """
        Args:
            x: image tokens, shape [B, L, C]
            hw_shape: tuple or list (H, W)
        Returns:
            proj_tokens: shape [B, L, C]
        """
        B, L, C = x.size()
    
        
        sw = sh = int(math.sqrt(L))
        if self.vis_gt_eff:
            sp_before = x.clone()
            
        if self.group_token_init_method in["avgpool",'conv_avgpool','conv','GCViT','conv_relu','avgpool_conv']:
            x = rearrange(x,
                          'b (h w) c -> b c h w',
                          h=sh, w=sw)
            group_token = self.group_token_init(x)
            group_token = rearrange(
                group_token,
                'b c h w -> b (h w) c'
            )
            x = rearrange(x,
                'b c h w -> b (h w) c ',
                h=sh, w=sw)
 
        elif self.group_token_init_method == "learnable":
            group_token = self.group_token.expand(x.size(0), -1, -1)
        if prev_token is None:
            gt = group_token
        elif self.group_projector_methonds in ["linear","conv",'conv_relu']:
            gt = group_token + self.group_projector(prev_token)
        elif self.group_projector_methonds == "cross" or self.group_projector_methonds == None:
            gt = group_token 
        
        else:
            raise(NotImplementedError)
        

        for i, (group_layer, pos_embed, un_group_layer, blocks) in enumerate(
            zip(self.group_layers, self.pos_embeds, self.un_group_layers, self.gt_attn if self.gt_attn else [None]*len(self.group_layers))
        ):
 

            if self.group_projector_methonds == "cross" and prev_token is not None:
                gt, _= self.group_projector(query=gt, key=prev_token, value=prev_token)            
            gt, _ = group_layer(query=gt, key=x, value=x, attn_dict_list = None)
            if len(blocks) > 0 :
                gt = gt + pos_embed
                gt = blocks(gt)
            proj_tokens, attn_dict_list = un_group_layer(query=x, key=gt, value=gt, attn_dict_list = attn_dict_list)
        if self.vis_gt_eff:
            self.visualize_gt_eff(sp_before = sp_before, sp_after = proj_tokens, gt = gt)
        return proj_tokens, attn_dict_list, gt
            
        # ungroup_tokens = ungroup_tokens.permute(0,2,1).contiguous().reshape(B, C, hw_shape[0], hw_shape[1])
        # proj_tokens = self.dwconv(ungroup_tokens).view(B, C, -1).permute(0,2,1).contiguous().view(B, L, C)
        # import h5py
        # with h5py.File("/data2/yunfei/vis_.h5", 'a') as f:
        #         keys = list(f.keys())            
        #         key = "ungroup_tokens"
        #         original_key = key
        #         count = 0
        #         while f"{original_key}{count}" in keys:
        #             count += 1
        #         key = f"{original_key}{count}"
        #         f.create_dataset(key, data=ungroup_tokens.detach().cpu().numpy())
        #         key = "proj_tokens"
        #         original_key = key
        #         count = 0
        #         while f"{original_key}{count}" in keys:
        #             count += 1
        #         key = f"{original_key}{count}"
        #         f.create_dataset(key, data=rearrange(
        #             proj_tokens,
        #             'b (h w) c -> b c h w',
        #             h = hw_shape[0],
        #             w = hw_shape[1]).detach().cpu().numpy())
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
