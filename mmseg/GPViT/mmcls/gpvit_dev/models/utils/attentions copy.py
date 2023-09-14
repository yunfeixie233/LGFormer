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

from mmengine.model import BaseModule, ModuleList
from mmcv.cnn.bricks import DropPath

def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

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

class LightAttModule(nn.Module):
    def __init__(self,
                 query_dims,
                 key_dims,
                 value_dims,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 q_project=True,
                 k_project=True,
                 v_project=True,
                 proj_after_att=True,
                 hard=False):
        super().__init__()
        if out_dim is None:
            out_dim = query_dims
        self.num_heads = num_heads
        head_dim = key_dims // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(value_dims, query_dims, bias=qkv_bias) if q_project else None
        self.k_proj = nn.Linear(key_dims, key_dims, bias=qkv_bias) if k_project else None
        self.v_proj = nn.Linear(query_dims, value_dims, bias=qkv_bias) if v_project else None

        self.attn_drop = nn.Dropout(attn_drop)

        if proj_after_att:
            self.proj = nn.Sequential(nn.Linear(query_dims, out_dim), nn.Dropout(proj_drop))
        else:
            self.proj = None
        self.hard = hard
    def forward(self, query, key, value, att_bias=None,attn_dict_list=None):
        bq, nq, cq = query.shape
        bk, nk, ck = key.shape
        bv, nv, cv = value.shape

        # [bq, nh, nq, cq//nh]
        if self.q_proj:
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=bq, n=nq, c=ck // self.num_heads)
        else:
            q = rearrange(query, 'b n (h c)-> b h n c', h=self.num_heads, b=bq, n=nq, c=cq // self.num_heads)
        # [bk, nh, nk, ck//nh]
        if self.k_proj:
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=bk, n=nk, c=ck // self.num_heads)
        else:
            k = rearrange(key, 'b n (h c)-> b h n c', h=self.num_heads, b=bk, n=nk, c=ck // self.num_heads)
        # [bv, nh, nv, cv//nh]
        if self.v_proj:
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=bv, n=nv, c=cq // self.num_heads)
        else:
            v = rearrange(value, 'b n (h c)-> b h n c', h=self.num_heads, b=bv, n=nv, c=cv // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if att_bias is not None:
            attn = attn + att_bias.unsqueeze(dim=1)
        
        # attn = attn.softmax(dim=-1)
        attn = gumbel_softmax(attn,hard = self.hard, dim = -1)
        attn = self.attn_drop(attn)
        if isinstance(attn_dict_list,list):
            attn_dict_list.append(attn)
        assert attn.shape == (bq, self.num_heads, nq, nk)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=bq, n=nq, c=cq // self.num_heads)
        if self.proj:
            out = self.proj(out)
        return out,attn_dict_list



class FullAttnModule(nn.Module):
    def __init__(self,
                 query_dims,
                 key_dims,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 q_project=True):
        super().__init__()
        if out_dim is None:
            out_dim = query_dims
        self.num_heads = num_heads
        head_dim = query_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(query_dims, query_dims, bias=qkv_bias) if q_project else None
        self.k_proj = nn.Linear(key_dims, key_dims, bias=qkv_bias)
        self.v_proj = nn.Linear(key_dims, key_dims, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(query_dims, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, att_bias=None):
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

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (bq, self.num_heads, nq, nk)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=bq, n=nq, c=cv // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class FullAttnCatBlock(nn.Module):
    def __init__(self,
                 query_dims,
                 key_dims,
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
                 **kwargs):
        super().__init__()
        self.with_cp = with_cp

        self.norm_query = build_norm_layer(norm_cfg, query_dims)[1]

        if not key_is_query:
            self.norm_key = build_norm_layer(norm_cfg, key_dims)[1]
        else:
            self.norm_key = None
        self.key_is_query = key_is_query

        if not value_is_key:
            self.norm_value = build_norm_layer(norm_cfg, query_dims)[1]
        else:
            self.norm_value = None
        self.value_is_key = value_is_key

        self.attn = FullAttnModule(
            query_dims,
            key_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            q_project=q_project)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        _ffn_cfgs = {
            'embed_dims': query_dims,
            'feedforward_channels': int(query_dims * ffn_ratio),
            'num_fcs': 2,
            'ffn_drop': drop,
            'dropout_layer': dict(type='DropPath', drop_prob=drop_path),
            'act_cfg': act_cfg,
        }
        self.ffn = FFN(**_ffn_cfgs)
        self.norm2 = build_norm_layer(norm_cfg, query_dims)[1]

        self.proj = nn.Linear(key_dims + query_dims, query_dims, bias=True)

    def forward(self, query, key, value, att_bias=None):
        def _inner_forward(query, key, value, att_bias):
            q = self.norm_query(query)
            k = q if self.key_is_query else self.norm_key(key)
            v = k if self.value_is_key else self.norm_value(value)

            x = torch.cat((query, self.drop_path(self.attn(q, k, v, att_bias=att_bias))), dim=-1)
            x = self.proj(x)
            x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp:
            return cp.checkpoint(_inner_forward, query, key, value, att_bias)
        else:
            return _inner_forward(query, key, value, att_bias)

class LightGroupAttnBlock(nn.Module):
    def __init__(self,
                 pixel_dims,
                 gt_dims,
                 pixel_shape,
                 gt_shape,
                 num_pixel_heads,
                 num_gt_heads,
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
                 hard=False):
        super().__init__()

        self.with_cp = with_cp

        self.norm_query = build_norm_layer(norm_cfg, query_dims)[1]

        if not key_is_query:
            self.norm_key = build_norm_layer(norm_cfg, key_dims)[1]
        else:
            self.norm_key = None
        self.key_is_query = key_is_query

        if not value_is_key:
            self.norm_value = build_norm_layer(norm_cfg, key_dims)[1]
        else:
            self.norm_value = None
        self.value_is_key = value_is_key
        position_embedding_method = "learnable"
        position_embedding_stride = 4
        sp_position_embedding_stride = 4
        pixel_qkv_method = "linear"
        self.attn = DualPathCrossAttentionLayer(
            in_channels_pixel = pixel_dims,
            in_channels_gt = gt_dims,
            pixel_qkv_channels =  (pixel_dims, pixel_dims, gt_dims),
            sp_qkv_channels = (pixel_dims, pixel_dims,pixel_dims),
            num_heads_pixel=num_pixel_heads,
            num_heads_sp=num_gt_heads,
            pixel_shape=pixel_shape,
            superpixel_shape=gt_shape,
            position_embedding_method=position_embedding_method,
            position_embedding_stride=position_embedding_stride,
            sp_position_embedding_stride=sp_position_embedding_stride,
            pixel_qkv_method=pixel_qkv_method,
            pixelify= (self.pixel_features_update_method == "residual"),
            hard_assign = False,
        )

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
                 pixel_dims,
                 sp_dims,
                 gt_dims,
                
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
                 zero_init_group_token = False,
                 hard = False,
                 **kwargs):

        super().__init__()

        self.pixel_dims = pixel_dims
        self.sp_dims = sp_dims
        self.gt_dims = gt_dims
        self.num_group_token = num_group_token
        self.with_cp = with_cp

        self.group_token = nn.Parameter(torch.zeros(1, num_group_token, sp_dims))
        self.group_projector = group_projector
        if not zero_init_group_token:
                trunc_normal_(self.group_token, std=.02)

        _group_att_cfg = dict(
            query_dims=gt_dims,
            key_dims=pixel_dims,
            value_dims=gt_dims,
            num_heads=num_group_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=group_qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=0.,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp,
            hard = hard)
        _group_att_cfg.update(group_att_cfg)
        self.group_layer = LightGroupAttnBlock(**_group_att_cfg)

        _mixer_cfg = dict(
            num_patches=num_group_token,
            embed_dims=gt_dims,
            patch_expansion=0.5,
            channel_expansion=4.0,
            depth=depth,
            drop_path=drop_path)
        _mixer_cfg.update(fwd_att_cfg)
        self.mixer = MLPMixer(**_mixer_cfg)

        _ungroup_att_cfg = dict(
            query_dims=sp_dims,
            key_dims=gt_dims,
            num_heads=num_ungroup_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=None,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp)
        _ungroup_att_cfg.update(ungroup_att_cfg)
        self.un_group_layer = FullAttnCatBlock(**_ungroup_att_cfg)

        self.dwconv = torch.nn.Sequential(
            nn.Conv2d(sp_dims, sp_dims, kernel_size=(3,3), padding=(1,1), bias=False, groups=sp_dims),
            nn.BatchNorm2d(num_features=sp_dims),
            nn.ReLU(True))

    def forward(self, pixel, superpixel, hw_shape, attn_dict_list = None, prev_token = None):
        """
        Args:
            x: image tokens, shape [B, L, C]
            hw_shape: tuple or list (H, W)
        Returns:
            proj_tokens: shape [B, L, C]
        """
        B, L, C = pixel.size()
        group_token = self.group_token.expand(pixel.size(0), -1, -1)
        if prev_token is None:
            gt = group_token
        else:
            gt = group_token + self.group_projector(prev_token)

        gt, attn_dict_list = self.group_layer(query=gt, key=pixel, value=pixel, attn_dict_list = attn_dict_list)

        gt = self.mixer(gt)
        superpixel_c = superpixel.shape[-1]
        superpixel_l = superpixel.shape[-2]
        ungroup_tokens = self.un_group_layer(query=superpixel, key=gt, value=gt)
        ungroup_tokens = ungroup_tokens.permute(0,2,1).contiguous().reshape(B, superpixel_c, hw_shape[0], hw_shape[1])
        proj_tokens = self.dwconv(ungroup_tokens).view(B, superpixel_c, -1).permute(0,2,1).contiguous().view(B, superpixel_l, superpixel_c)

        return proj_tokens, attn_dict_list,gt

# class DualPathCrossAttentionLayer(BaseModule):
#     """Applies a transformer layer, as proposed in MaX-DeepLab models."""

#     def __init__(
#         self,
#         in_channels_pixel: int,
#         in_channels_gt: int,
#         pixel_qkv_channels: ThreeInt,
#         gt_qkv_channels: ThreeInt,
#         num_heads_pixel: int = 1,
#         num_heads_gt: int = 1,
#         norm_layer=partial(tml.LayerNorm2d, eps=1e-6),
#         pixel_shape=None,
#         gt_shape=None,
#         position_embedding_method="none",
#         position_embedding_stride=4,
#         gt_position_embedding_stride=None,
#         pixel_qkv_method: str = "linear",
#         ls_init_value: Optional[float] = None,
#         pixelify: bool = True,
#         # always_return_similarity_pixel: bool = False,
#         hard_assign: bool = True,
#     ):
#         """Initializes a DualPathTransformerSimpleLayer."""
#         super().__init__()

#         gt_q_channels, gt_k_channels, gt_v_channels = gt_qkv_channels
#         pixel_q_channels, pixel_k_channels, pixel_v_channels = pixel_qkv_channels

#         assert gt_q_channels == pixel_k_channels
#         assert pixel_q_channels == gt_k_channels
#         assert in_channels_gt == pixel_v_channels
#         assert in_channels_pixel == gt_v_channels

#         if pixel_q_channels % num_heads_pixel:
#             raise ValueError("Total_key_depth should be divisible by num_heads.")

#         if gt_q_channels % num_heads_gt:
#             raise ValueError("Total_value_depth should be divisible by num_heads.")

#         self.gt_qkv_channels = gt_qkv_channels
#         self.pixel_qkv_channels = pixel_qkv_channels

#         # Compute query key value with one convolution and a batch norm layer. The
#         # initialization std is standard transformer initialization (without batch
#         # norm), as used in SASA and ViT. In our case, we use batch norm by default,
#         # so it does not require careful tuning. If one wants to remove all batch
#         # norms in axial attention, this standard initialization should still be
#         # good, but a more careful initialization is encouraged.
#         # FIXME(meijieru): initialization
#         # initialization_std = bottleneck_channels**-0.5

#         # TODO(meijieru): if we don't need pixelify, we can further remove sp_value
#         self._gt_qkv = Conv2D(
#             in_channels_gt,
#             sum(gt_qkv_channels),
#             use_bn=True,
#             bn_layer=norm_layer,
#             activation="none",
#         )

#         self._always_return_similarity_pixel = always_return_similarity_pixel
#         self._pixelify = pixelify
#         self._pixel_qkv_method = pixel_qkv_method
#         if self._pixel_qkv_method == "linear":
#             self._pixel_qkv = Conv2D(
#                 in_channels_pixel,
#                 sum(pixel_qkv_channels),
#                 use_bn=True,
#                 bn_layer=norm_layer,
#                 activation="none",
#             )
#         else:
#             raise ValueError(f"Unknown pixel_qkv_method: {self._pixel_qkv_method}")

#         self._num_heads_pixel = num_heads_pixel
#         self._num_heads_gt = num_heads_gt
#         self._position_embedding_method = position_embedding_method
#         self._position_embedding_stride = position_embedding_stride
#         self._gt_position_embedding_stride = (
#             gt_position_embedding_stride or position_embedding_stride
#         )

#         if self._position_embedding_method == "none":
#             pass
#         elif self._position_embedding_method == "learnable":
#             self.gt_pos_embed = self._create_pos_embed(
#                 in_channels_gt, gt_shape, self._gt_position_embedding_stride
#             )
#             self.pixel_pos_embed = self._create_pos_embed(
#                 in_channels_pixel, pixel_shape, self._position_embedding_stride
#             )
#             print(
#                 f"gt_pos_embed shape: {self.gt_pos_embed.shape}, "
#                 f"pixel_pos_embed shape: {self.pixel_pos_embed.shape}"
#             )
#         elif self._position_embedding_method == "depthwise":
#             self.gt_pos_conv = tml.create_conv2d(
#                 in_channels_gt, in_channels_gt, 3, bias=True, depthwise=True
#             )
#             self.pixel_pos_conv = tml.create_conv2d(
#                 in_channels_pixel,
#                 in_channels_pixel,
#                 3,
#                 bias=True,
#                 depthwise=True,
#             )
#         else:
#             raise ValueError()

#         if ls_init_value is not None:
#             self.gt_ls1 = LayerScale2d(in_channels_gt, ls_init_value)
#             if pixelify:
#                 self.pixel_ls1 = LayerScale2d(in_channels_pixel, ls_init_value)
#             else:
#                 self.pixel_ls1 = nn.Identity()
#         else:
#             self.gt_ls1 = nn.Identity()
#             self.pixel_ls1 = nn.Identity()

#         self.hard_assign = hard_assign
#     def _create_pos_embed(self, dim, shape, stride) -> torch.Tensor:
#         pos_embed_shape = [int(math.ceil(val / stride)) for val in shape]
#         for val in pos_embed_shape:
#             assert val > 1
#         pos_embed = nn.Parameter(torch.zeros(1, dim, *pos_embed_shape))
#         return pos_embed

#     def _add_pos_embed(self, val, pos_embed) -> torch.Tensor:
#         _, _, h, w = val.shape
#         return val + F.interpolate(
#             pos_embed, size=(h, w), mode="bilinear", align_corners=False
#         )

#     def _split_qkv(
#         self,
#         val: torch.Tensor,
#         qkv_channels: Sequence[int],
#         qkv_method: str,
#     ) -> Sequence[torch.Tensor]:
#         if qkv_method == "linear":
#             query, key, value = torch.split(
#                 val,
#                 list(qkv_channels),
#                 dim=1,
#             )
#         else:
#             raise ValueError(f"Unknown pixel_qkv_method: {self._pixel_qkv_method}")

#         return query, key, value

#     def _gt_update_step(
#         self,
#         gt_query: torch.Tensor,
#         pixel_key: torch.Tensor,
#         pixel_value: torch.Tensor,
        
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         gt_query, pixel_key, pixel_value = [
#             reshape_and_transpose_for_attention_operation(val, self._num_heads_gt)
#             for val in [gt_query, pixel_key, pixel_value]
#         ]

#         bgt, _, _, sh, sw = gt_query.shape
#         b, num_heads, c_key, h, w = pixel_key.shape
#         _, _, c_value, _, _ = pixel_value.shape
#         assert b == bgt
#         scale = c_key**-0.5
        
#         # [b * num_heads]
#         gt_query, pixel_key, pixel_value = [
#             val.flatten(end_dim=1) for val in [gt_query, pixel_key, pixel_value]
#         ]
        
#         attn = ( gt_query @ pixel_key.transpose(-2, -1))*scale
#         attn = attn.softmax(dim=-1)
        
#         gt_delta = superpixel_ops.update_superpixel_features(
#             pixel_value, sp_query, similarities,hard_assign=self.hard_assign
#         )
#         gt_delta = sp_delta.reshape(b, num_heads * c_value, sh, sw)
#         return gt_delta, similarities.reshape(b, num_heads, 9, h, w)

#     def _superpixel_assign_step(
#         self,
#         pixel_query: torch.Tensor,
#         sp_key: torch.Tensor,
#         sp_value: torch.Tensor,
#     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
#         if not self._always_return_similarity_pixel and not self._pixelify:
#             return None, None

#         pixel_query, sp_key, sp_value = [
#             reshape_and_transpose_for_attention_operation(val, self._num_heads_pixel)
#             for val in [pixel_query, sp_key, sp_value]
#         ]

#         bsp, _, _, _, _ = sp_key.shape
#         b, num_heads, c_key, h, w = pixel_query.shape
#         _, _, c_value, _, _ = sp_value.shape
#         assert b == bsp
#         scale = c_key**-0.5
#         pixel_query, sp_key, sp_value = [
#             val.flatten(end_dim=1) for val in [pixel_query, sp_key, sp_value]
#         ]
#         similarities = (
#             superpixel_ops.compute_similarities_dot_product(pixel_query, sp_key) * scale
#         )
#         pixel_delta = None
#         if self._pixelify:
#             pixel_delta = superpixel_ops.update_pixel_features(
#                 None, sp_value, similarities
#             )
#             pixel_delta = pixel_delta.reshape(b, num_heads * c_value, h, w)
#         return pixel_delta, similarities.reshape(b, num_heads, 9, h, w)

#     def forward(self, pixel_features, sp_features):
#         """Performs a forward pass."""
#         # [b, c, h, w], [b, csp, sh, sw]

#         if self._position_embedding_method == "learnable":
#             sp_features_embeded = self._add_pos_embed(sp_features, self.sp_pos_embed)
#             pixel_features_embeded = self._add_pos_embed(
#                 pixel_features, self.pixel_pos_embed
#             )
#         elif self._position_embedding_method == "depthwise":
#             sp_features_embeded = self.sp_pos_conv(sp_features) + sp_features
#             pixel_features_embeded = (
#                 self.pixel_pos_conv(pixel_features) + pixel_features
#             )
#         else:
#             sp_features_embeded = sp_features
#             pixel_features_embeded = pixel_features

#         sp_query, sp_key, sp_value = self._split_qkv(
#             self._sp_qkv(sp_features_embeded), self.sp_qkv_channels, "linear"
#         )

#         pixel_query, pixel_key, pixel_value = self._split_qkv(
#             self._pixel_qkv(pixel_features_embeded),
#             self.pixel_qkv_channels,
#             self._pixel_qkv_method,
#         )
         
#         (
#             sp_feature_delta,
#             similarities_multi_head,
#         ) = self._superpixel_update_step(sp_query, pixel_key, pixel_value)
#         if self._pixelify:
#             (
#                 pixel_features_delta,
#                 similarities_multi_head_pixel,
#             ) = self._superpixel_assign_step(pixel_query, sp_key, sp_value)
#         if self._pixelify:
#             pixel_return = pixel_features + self.pixel_ls1(pixel_features_delta)
#         else:
#             pixel_return = None

#         return (
#             pixel_return,
#             sp_features + self.sp_ls1(sp_feature_delta),
#             similarities_multi_head,
#             similarities_multi_head_pixel,
#         )

#     def init_weights(self):
#         initialization_std = 0.02
#         for conv in [
#             self._sp_qkv.conv,
#             self._pixel_qkv.conv,
#         ]:
#             # NOTE(meijieru): don't use conv init, as they are actually 3 linear
#             # Linear initialization like timm
#             tml.trunc_normal_(conv.weight, std=initialization_std)
#             if conv.bias is not None:
#                 nn.init.zeros_(conv.bias)