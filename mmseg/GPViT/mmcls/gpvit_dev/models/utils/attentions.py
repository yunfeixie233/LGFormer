"""
Author: Chenhongyi Yang
LePE attention References: https://github.com/microsoft/CSWin-Transformer
"""
from typing import Sequence

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
# sys.path.append("/root/autodl-tmp/GroupViT/models")
# from group_vit import *
def img2windows(img, H_sp, W_sp):
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class LePEAttention(nn.Module):
    def __init__(self, dim, mode, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        assert mode in (0, 1)
        self.mode = mode
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        if self.mode == 0:
            H_sp, W_sp = H, self.split_size
        else:
            H_sp, W_sp = self.split_size, W
        x = img2windows(x, H_sp, W_sp)
        x = x.reshape(-1, H_sp * W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, hw_shape, func):
        B, N, C = x.shape
        H, W = hw_shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        if self.mode == 0:
            H_sp, W_sp = H, self.split_size
        else:
            H_sp, W_sp = self.split_size, W
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'
        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        x = x.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv, hw_shape):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]
        ### Img2Window
        H, W = hw_shape
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q, hw_shape)
        k = self.im2cswin(k, hw_shape)
        v, lepe = self.get_lepe(v, hw_shape, self.get_v)

        if self.mode == 0:
            H_sp, W_sp = H, self.split_size
        else:
            H_sp, W_sp = self.split_size, W

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, H_sp * W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, H_sp, W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x

class LePEAttnSimpleDWBlock(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,  # For convenience, we use window size to denote split size
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path=0.,
                 attn_cfgs=dict(),
                 ffn_cfgs=dict(),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super().__init__(init_cfg)
        self.with_cp = with_cp
        self.dim = embed_dims
        self.num_heads = num_heads
        self.split_size = window_size
        self.ffn_ratio = ffn_ratio
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=True)

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.branch_num = 2
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(0.)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attns = nn.ModuleList([
            LePEAttention(
                embed_dims // 2, mode=i,
                split_size=self.split_size, num_heads=num_heads // 2, dim_out=embed_dims // 2,
                qk_scale=None, attn_drop=0., proj_drop=drop_rate)
            for i in range(self.branch_num)])

        _ffn_cfgs = {
            'embed_dims': embed_dims,
            'feedforward_channels': int(embed_dims * ffn_ratio),
            'num_fcs': 2,
            'ffn_drop': drop_rate,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'act_cfg': dict(type='GELU'),
            'layer_scale_init_value':1e-6,
            **ffn_cfgs
        }
        self.ffn = FFN(**_ffn_cfgs)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.dw = nn.Conv2d(embed_dims, embed_dims, kernel_size=(3, 3), padding=(1, 1), bias=False, groups=embed_dims)

    def forward(self, x, hw_shape):
        """
        x: B, H*W, C
        """
        def _inner_forward(x, hw_shape):
            H, W = hw_shape
            B, L, C = x.shape
            assert L == H * W, "flatten img_tokens has wrong size"
            img = self.norm1(x)
            qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3).contiguous()

            x1 = self.attns[0](qkv[:, :, :, :C // 2], hw_shape)
            x2 = self.attns[1](qkv[:, :, :, C // 2:], hw_shape)
            attened_x = torch.cat([x1, x2], dim=2)
            attened_x = self.proj(attened_x)
            x = x + self.drop_path(attened_x)

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            B, L, C = x.shape
            x = x.permute(0, 2, 1).contiguous().reshape(B, C, hw_shape[0], hw_shape[1])
            x = self.dw(x)
            x = x.reshape(B, C, L).permute(0, 2, 1).contiguous()
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, hw_shape)
        else:
            x = _inner_forward(x, hw_shape)
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
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 key_is_query=False,
                 value_is_key=False,
                 q_project=True,
                 with_cp=False,
                 association_embedding = False,
                #  layer_scale_init_value = 1e-5,
                 layer_scale_init_value = 0.,
                 
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
            proj_drop=drop,
            q_project=q_project,
            association_embedding = association_embedding)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        _ffn_cfgs = {
            'embed_dims': embed_dims,
            'feedforward_channels': int(embed_dims * ffn_ratio),
            'num_fcs': 2,
            'ffn_drop': drop,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'act_cfg': act_cfg,
            'layer_scale_init_value':layer_scale_init_value
        }

        self.ffn = FFN(**_ffn_cfgs)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.proj = nn.Linear(embed_dims * 2, embed_dims, bias=True)

    def forward(self, query, key, value, att_bias=None,attn_dict_list = None):
        def _inner_forward(query, key, value, att_bias,attn_dict_list):
            q = self.norm_query(query)
            k = q if self.key_is_query else self.norm_key(key)
            v = k if self.value_is_key else self.norm_value(value)
            new_x, attn_dict_list = self.attn(q, k, v, att_bias=att_bias,attn_dict_list = attn_dict_list)
            # x = torch.cat((query, self.drop_path(x)),dim=-1)
            # x = self.proj(x)
            x = self.ffn(self.norm2(new_x), identity=query)
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
                 with_cp=False):
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
            q_project=False,
            k_project=True,
            v_project=False,
            proj_after_att=False)

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
                 drop=0.,
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
                 all_ls: bool = False,
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
                 nn.AvgPool2d(kernel_size=5,stride=5)
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
        else:
            raise(NotImplementedError)
        self.group_projector = group_projector
        self.group_projector_methonds = group_projector_methonds
        if not zero_init_group_token:
                trunc_normal_(self.group_token, std=.02)
        
        self.pos_embed = nn.Parameter(torch.randn(1, num_group_token, embed_dims) * 0.02)      
        _group_att_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_group_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=group_qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=0.,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp)
        _group_att_cfg.update(group_att_cfg)
        if all_ls:
            self.group_layer = FullAttnCatBlock(**_group_att_cfg)
        else:    
            self.group_layer = LightGroupAttnBlock(**_group_att_cfg)

        _mixer_cfg = dict(
            num_patches=num_group_token,
            embed_dims=embed_dims,
            patch_expansion=0.5,
            channel_expansion=4.0,
            depth=depth,
            drop_path=drop_path)
        _mixer_cfg.update(fwd_att_cfg)
        # self.mixer = MLPMixer(**_mixer_cfg)

        _ungroup_att_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_ungroup_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=None,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp,)
        _ungroup_att_cfg.update(ungroup_att_cfg)
        self.use_assign = use_assign
        if self.use_assign:
            self.pre_assign_attn = CrossAttnBlock(
                dim=embed_dims, num_heads=num_ungroup_heads, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm, post_norm=True)
            self.assign = AssignAttention(
            dim=embed_dims,
            num_heads=1,
            qkv_bias=True,
            hard=True,
            gumbel=True,
            gumbel_tau=1.,
            sum_assign=False,
            assign_eps=1.)
        else:
            self.un_group_layer = FullAttnCatBlock(**_ungroup_att_cfg)

        self.dwconv = torch.nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=(3,3), padding=(1,1), bias=False, groups=embed_dims),
            nn.BatchNorm2d(num_features=embed_dims),
            # nn.ReLU(True))
            nn.GELU())
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dims,
                    num_heads=2,
                    mlp_ratio=4,
                    qkv_bias=True,
                    # drop=drop_rate,
                    proj_drop=drop,
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
        self.init_weights()
    def init_weights(self):
        if self.pos_embed is not None:
            timm_layers.trunc_normal_(self.pos_embed, std=0.02)
        
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
        vis_gt_eff = False
        if vis_gt_eff:
            import h5py
            sp_before = x.detach()
            sp_before = rearrange(sp_before,
                                'b (h w) c -> b c h w ',
                                h = sh, w = sw)
            
            with h5py.File("/root/autodl-tmp/vis_before.h5","a") as f:
                keys = list(f.keys())
                key = "sp_before"
                original_key = key
                count = int(0)
                while key in keys:
                    count = int(count) + 1
                    key = original_key + str(count)
                if int(count) < 5:
                    f.create_dataset(key,data=sp_before.detach().cpu().numpy()) 
            
        if self.group_token_init_method in["avgpool",'conv_avgpool','conv']:
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
        elif self.group_projector_methonds == "linear" or "conv":
            gt = group_token + self.group_projector(prev_token)
        elif self.group_projector_methonds == "cross" or self.group_projector_methonds == None:
            gt = group_token 
        
        else:
            raise(NotImplementedError)
        gt, _ = self.group_layer(query=gt, key=x, value=x, attn_dict_list = None)
        gt = gt + self.pos_embed
        gt = self.blocks(gt)
        if self.group_projector_methonds == "cross" and prev_token is not None:
            gt= self.group_projector(query=gt, key=prev_token, value=prev_token)
        if self.use_assign:
            gt = self.pre_assign_attn(gt, x)
            new_x, attn_dict_list = self.assign(query = x,key = gt, value = gt ,attn_dict_list = attn_dict_list)
            x = new_x + x
            return x,  attn_dict_list, gt
        else:
            proj_tokens, attn_dict_list = self.un_group_layer(query=x, key=gt, value=gt, attn_dict_list = attn_dict_list)
        
        if vis_gt_eff:
            import h5py
            sp_after = proj_tokens.detach()
            sp_after = rearrange(sp_after,
                                'b (h w) c -> b c h w ',
                                h = sh, w = sw)
            
            with h5py.File("/root/autodl-tmp/vis_after.h5","a") as f:
                keys = list(f.keys())
                key = "sp_after"
                original_key = key
                count = int(0)
                while key in keys:
                    count = int(count) + 1
                    key = original_key + str(count)
                if int(count) < 5:
                    f.create_dataset(key,data=sp_after.detach().cpu().numpy()) 
            
            diff = sp_after - sp_before

            
            with h5py.File("/root/autodl-tmp/diff.h5","a") as f:
                keys = list(f.keys())
                key = "sp_after"
                original_key = key
                count = int(0)
                while key in keys:
                    count = int(count) + 1
                    key = original_key + str(count)
                if int(count) < 5:
                    f.create_dataset(key,data=diff.detach().cpu().numpy()) 
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

