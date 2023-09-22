from tkinter import N
from typing import Any, Callable, Dict, MutableMapping, Optional, Sequence, Tuple, Union
import warnings
import math
import einops
from matplotlib.pyplot import xcorr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, lru_cache

import timm
from timm.models.vision_transformer import Block, _cfg
from timm.models.registry import register_model
from timm.models import layers as timm_layers

from ...superformer.superpixel.dual_path_transformer_ops import Conv2D
from .decode_head import BaseDecodeHead
from ..builder import HEADS


from ...superformer.superpixel import superpixel_transformer as st
from ...superformer.superpixel import superpixel_ops
import math
rearrange = einops.rearrange
LayerScale2d = st.LayerScale2d
SKIP_CONFIRM = False

def positionalencoding1d(d_model, length,device):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(5.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    pe = pe.to(device=device)
    return pe

    
def build_2d_sincos_position_embedding(patches_resolution, embed_dim, device, temperature=1. ):
    h = w = patches_resolution
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    
    pos_dim = embed_dim // 4
    
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    
    # Generate one set of positional embeddings using the grid for width
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    pos_emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)
    
    # Replicate these embeddings for height
    pos_emb_h = pos_emb_w.clone()
    
    # Concatenate to form the final 2D positional embedding
    pos_emb = torch.cat([pos_emb_w, pos_emb_h], dim=1)[None, :, :]
    
    # Uncomment the following line if you don't want to update the positional embeddings during training
    # pos_emb.requires_grad = False
    pos_emb = nn.Parameter(pos_emb).to(device)
    
    return pos_emb

def _update_params(defaults, **kwargs):
    defaults.update(kwargs)
    return defaults


def get_norm_layer_2d(norm_layer):
    layer_func = (
        norm_layer.func if isinstance(norm_layer, partial) else norm_layer.__class__
    )
    if layer_func is nn.LayerNorm:
        return timm_layers.LayerNorm2d
    else:
        raise NotImplementedError()




def init_superformer_weights(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        print(f"init_weights: {name}")
        module.init_weights()


def create_downsample(
    stride: int,
    in_channels: int,
    out_channels: int,
    norm_layer_2d: Callable,
    pre_norm: bool,
) -> nn.Module:
    if stride != 1 or in_channels != out_channels:
        assert stride in [1, 2], f"stride must be 1 or 2, got {stride}"
        ds_ks = 1 if stride == 1 else 2
        padding = 0

        downsample = nn.Sequential(
            # TODO(meijieru): norm first or not?
            norm_layer_2d(in_channels),
            timm_layers.create_conv2d(
                in_channels,
                out_channels,
                ds_ks,
                stride=stride,
                bias=True,
                padding=padding,
            ),
        )
    else:
        downsample = norm_layer_2d(in_channels) if pre_norm else nn.Identity()
    return downsample


def prepare_similarities(
    patch_embed: st.SuperPixelTokenization,
    similarities: torch.Tensor,
    scale_factor,
    merge_multihead_similarities: bool = True,
) -> torch.Tensor:
    if similarities.dim() == 5:
        if not merge_multihead_similarities:
            raise NotImplementedError(f"TODO(meijieru): {similarities.shape}")

        # For compatibility with old checkpoints.
        _, num_heads, _, _, _ = similarities.shape
        similarities = similarities.sum(1) / math.sqrt(num_heads)

    b, _, h, w = similarities.shape
    sh, sw = patch_embed.superpixel_shape
    ph, pw = h // sh, w // sw

    similarities = similarities.reshape(b, 9, sh, ph, sw, pw)
    if scale_factor != 1:
        similarities = superpixel_ops.resize_similarities(
            similarities,
            scale_factor=scale_factor,
            # mode = "bicubic"
        )
        similarities = superpixel_ops.maskout_boundary(similarities)
    return similarities


def visualize_superpixel(tokenization_info, patch_embed, scale_factor):
    assert tokenization_info is not None
    colormap = superpixel_ops.create_superpixel_colormap("random")

    res = {}
    for key, similarities in tokenization_info.items():
        if similarities.dim() == 5 and similarities.size(1) > 1:
            # visualize each head
            for i in range(similarities.size(1)):
                val = prepare_similarities(
                    patch_embed,
                    similarities[:, i],
                    scale_factor,
                    merge_multihead_similarities=False,
                )
                labels = superpixel_ops.compute_hard_association(val).cpu()
                vis = colormap[labels % colormap.size(0)]
                res[f"{key}_head{i}"] = vis
        similarities = prepare_similarities(
            patch_embed, similarities, scale_factor, merge_multihead_similarities=True
        )
        labels = superpixel_ops.compute_hard_association(similarities).cpu()
        vis = colormap[labels % colormap.size(0)]
        res[key] = vis
    return res


class PixelRefineSepConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size,
        norm_layer_2d,
        act_layer,
        use_shortcut,
        **kwargs,
    ) -> None:
        super().__init__()
        self.sep_conv = timm_layers.separable_conv.SeparableConv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            **kwargs,
        )
        self.norm = norm_layer_2d(in_channels)
        self.act = act_layer()
        self.ls = (
            LayerScale2d(in_channels, init_values=1e-5)
            if use_shortcut
            else nn.Identity()
        )
        self.use_shortcut = use_shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.sep_conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.ls(x)
        return shortcut + x if self.use_shortcut else x


class SuperformerStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_channels_sp: Optional[int],
        out_channels: int,
        stride: int,
        sp_stride: int,
        num_heads: int,
        depth: int,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate: Union[float, Sequence[float]] = 0.0,
        norm_layer=nn.LayerNorm,
        norm_layer_2d=timm_layers.LayerNorm2d,
        act_layer=nn.GELU,
        block_fn=Block,
        ls_init_value: Optional[float] = None,
        superpixel_layer: Optional[st.SuperPixelTokenization] = None,
        superpixel_shape: Tuple[int, int] = (-1, -1),
        seg_block_idx: Optional[int] = None,
        sp_embed_method: str = "identity",
        sp_project_method: str = "noact",
        pre_norm_pixel: bool = False,
        pixel_refine_method: str = "identity",
        return_updated_pixel_features: bool = False,
        unflatten_sp_features: bool = False,
        use_pos_embed: bool = False,
        pos_embed_method: Union[str,Sequence[str]] = "fixed",
        use_cls_token: bool = False,
        no_embed_class: bool = False,
        use_pixel_similarities: bool = False,
        use_middle_pixel_features: bool = False,
        similarities_embedding: bool = False,
    ) -> None:
        super().__init__()

        assert stride == 1

        print(f"superpixel_shape: {superpixel_shape}")

        self.sp_stride = sp_stride
        # [0, seg_block_idx) are the blocks for segmentation
        self.seg_block_idx = seg_block_idx if seg_block_idx is not None else depth

        self.return_updated_pixel_features = return_updated_pixel_features
        self.unflatten_sp_features = unflatten_sp_features
        self.superpixel_shape = superpixel_shape
        self.no_embed_class = no_embed_class

        self.patch_embed = superpixel_layer()
        self.use_pixel_similarities = use_pixel_similarities
        self.use_middle_pixel_features = use_middle_pixel_features

        if pre_norm_pixel:
            raise NotImplementedError()

        if use_pixel_similarities:
            warnings.warn(
                "`use_pixel_similarities` is for compatible with old models only"
            )
            # if not SKIP_CONFIRM:
            #     __import__("ipdb").set_trace()

        if (
            isinstance(self.patch_embed, nn.AvgPool2d)
            or self.patch_embed.superpixel_features_init_method != "from_feature"
            or sp_embed_method == "identity"
        ):
            self.sp_transform_last = nn.Identity()
        elif sp_embed_method == "conv3x3_norm":
            self.sp_transform_last = nn.Sequential(
                timm_layers.create_conv2d(
                    out_channels, out_channels, 3, padding="same"
                ),
                norm_layer_2d(out_channels),
            )
        elif sp_embed_method == "conv1x1_norm":
            self.sp_transform_last = nn.Sequential(
                timm_layers.create_conv2d(
                    out_channels, out_channels, 1, padding="same"
                ),
                norm_layer_2d(out_channels),
            )
        elif sp_embed_method == "conv3x3_norm_act":
            self.sp_transform_last = nn.Sequential(
                timm_layers.create_conv2d(
                    out_channels, out_channels, 3, padding="same"
                ),
                norm_layer_2d(out_channels),
                act_layer(),
            )
        elif sp_embed_method == "conv3x3":
            self.sp_transform_last = nn.Sequential(
                timm_layers.create_conv2d(
                    out_channels, out_channels, 3, padding="same"
                ),
            )
        else:
            raise ValueError(f"Unknown sp_embed_method: {sp_embed_method}")

        # NOTE(meijieru): break non asym
        self.sp_lift = nn.Identity()
        # if in_channels == out_channels or in_channels_sp is None:
        #     self.sp_lift = nn.Identity()
        # else:
        #     self.sp_lift = nn.Sequential(
        #         timm_layers.create_conv2d(in_channels, out_channels, 1, padding="same"),
        #         norm_layer_2d(out_channels),
        #         act_layer(),
        #     )

        if return_updated_pixel_features:
            assert sp_project_method in ["noact", "act"]
            self.sp_project = nn.Sequential(
                timm_layers.create_conv2d(out_channels, in_channels, 1, padding="same"),
                norm_layer_2d(in_channels),
                act_layer() if sp_project_method == "act" else nn.Identity(),
            )
            self.pixel_delta_ls = (
                LayerScale2d(in_channels, ls_init_value)
                if ls_init_value is not None
                else nn.Identity()
            )
        else:
            self.sp_project = nn.Identity()
            self.pixel_delta_ls = nn.Identity()

        if not return_updated_pixel_features or pixel_refine_method == "identity":
            self.pixel_refine = nn.Identity()
        elif pixel_refine_method.startswith("sep_conv_"):
            kernel_size = int(pixel_refine_method.split("_")[-1])
            print(f"Using separable conv with residual, kernel size {kernel_size}")
            self.pixel_refine = PixelRefineSepConv(
                in_channels,
                kernel_size,
                act_layer=act_layer,
                norm_layer_2d=norm_layer_2d,
                use_shortcut=True,
            )
        elif pixel_refine_method.startswith("no_residual_sep_conv_"):
            kernel_size = int(pixel_refine_method.split("_")[-1])
            print(f"Using separable conv without residual, kernel size {kernel_size}")
            self.pixel_refine = PixelRefineSepConv(
                in_channels,
                kernel_size,
                act_layer=act_layer,
                norm_layer_2d=norm_layer_2d,
                use_shortcut=False,
            )
        else:
            raise ValueError(f"Unknown pixel_refine_method: {pixel_refine_method}")

        if isinstance(drop_path_rate, Sequence):
            assert len(drop_path_rate) == depth

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, out_channels)) if use_cls_token else None
        )
        self.num_prefix_tokens = 1 if use_cls_token else 0
        num_patches = superpixel_shape[0] * superpixel_shape[1]
        embed_len = num_patches + 1 if use_cls_token else num_patches
        self.pos_embed_method = pos_embed_method
        if use_pos_embed:
            self.pos_embed = None
            self.pixel_pos_embed = None
            assert pos_embed_method in ["fixed", "pixel", "pixel_no_grad", "both", "both_no_grad"]
            if pos_embed_method in ["fixed", "both", "both_no_grad"]:
                self.pos_embed = nn.Parameter(torch.randn(1, embed_len, out_channels) * 0.02)
            if pos_embed_method in ["pixel", "pixel_no_grad", "both", "both_no_grad"]:
                assert isinstance(self.patch_embed, st.SuperPixelTokenization)
                num_heads_sp = self.patch_embed.num_heads
                self.pixel_pos_embed = nn.Parameter(
                    torch.randn(
                        1,
                        num_heads_sp,
                        out_channels // num_heads_sp,
                        *self.patch_embed.pixel_shape,
                    ) * 0.02
                )
        else:
            self.pos_embed = None
            self.pixel_pos_embed = None
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=out_channels,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    qkv_bias=True,
                    # drop=drop_rate,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_rate[i]
                        if isinstance(drop_path_rate, Sequence)
                        else drop_path_rate
                    ),
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    init_values=ls_init_value
                )
                for i in range(depth)
            ]
        )
        self.similarities_embedding = similarities_embedding
        if True:
            self.emb_init = nn.Sequential(
                timm_layers.create_conv2d(
                    in_channels=in_channels_sp,
                    out_channels=in_channels_sp,
                    kernel_size=4,
                    stride=4,
                    padding="same",
                    bias=False,
                ),
                norm_layer_2d(in_channels_sp),
                act_layer(),
            )

        self.init_weights()

    def init_weights(self):
        if self.pos_embed is not None:
            timm_layers.trunc_normal_(self.pos_embed, std=0.02)
        if self.pixel_pos_embed is not None:
            timm_layers.trunc_normal_(self.pixel_pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)

    def forward_patchify(
        self, x: torch.Tensor, sp_features_last: Optional[torch.Tensor],prev_info = None
    ):
        if isinstance(self.patch_embed, nn.AvgPool2d):
            # NOTE(meijieru): doesn't use sp_features_last
            sp_features = self.patch_embed(x)
            return None, sp_features, x


        if sp_features_last is not None:
            sp_features_last = self.sp_transform_last(sp_features_last)
        info, pixel_features, sp_features = self.patch_embed(x, sp_features_last)

        self._tokenization_info = {
            key: val for key, val in info.items() if "similarities" in key
        }
        return info, sp_features, pixel_features

    @property
    def tokenization_info(self) -> Optional[MutableMapping[str, torch.Tensor]]:
        return getattr(self, "_tokenization_info", None)

    def forward_pixelify(
        self, x: torch.Tensor, sp_features_unflatten: torch.Tensor, info: Any
    ) -> torch.Tensor:
        assert x is not None
        if isinstance(self.patch_embed, nn.AvgPool2d):
            pixel_delta = F.interpolate(
                sp_features_unflatten,
                scale_factor=self.patch_embed.kernel_size,
                mode="bilinear",
            )
        else:
            similarity = st.get_final_similarity(
                info,
                self.patch_embed.num_blocks,
                pixel=self.use_pixel_similarities,
                merge=False,
            )
            if similarity is None:
                raise ValueError("Similarity is None")

            b, num_heads, _, h, w = similarity.shape
            if num_heads == 1:
                pixel_delta = superpixel_ops.update_pixel_features(
                    None, sp_features_unflatten, similarity.squeeze(1)
                )
            else:
                _, c, _, _ = sp_features_unflatten.shape
                pixel_delta = superpixel_ops.update_pixel_features(
                    None,
                    st.reshape_and_transpose_for_attention_operation(
                        sp_features_unflatten, num_heads
                    ).flatten(end_dim=1),
                    similarity.flatten(end_dim=1),
                )
                pixel_delta = pixel_delta.reshape(b, c, h, w)
        res = x + self.pixel_delta_ls(pixel_delta)
        res = self.pixel_refine(res)
        # import h5py
        # with h5py.File("/root/autodl-tmp/vis.h5","a") as f:
        #     keys = list(f.keys())
        #     key = "pixel_delta"
        #     original_key = key
        #     count = 0
        #     while key in keys:
        #         print(f"Dataset with key {key} already exists. Updating key name.")
        #         count += 1
        #         key = original_key + str(count)
        #     f.create_dataset(key,data=pixel_delta.detach().cpu().numpy()) 
            
        #     key = "pixel_delta_ls"
        #     original_key = key
        #     count = 0
        #     while key in keys:
        #         print(f"Dataset with key {key} already exists. Updating key name.")
        #         count += 1
        #         key = original_key + str(count)
        #     f.create_dataset(key,data=self.pixel_delta_ls(pixel_delta).detach().cpu().numpy()) 
        #     key = "x"
        #     original_key = key
        #     count = 0
        #     while key in keys:
        #         print(f"Dataset with key {key} already exists. Updating key name.")
        #         count += 1
        #         key = original_key + str(count)
        #     f.create_dataset(key,data=x.detach().cpu().numpy()) 
        #     key = "res"
        #     original_key = key
        #     count = 0
        #     while key in keys:
        #         print(f"Dataset with key {key} already exists. Updating key name.")
        #         count += 1
        #         key = original_key + str(count)                          
        #     f.create_dataset(key,data=res.detach().cpu().numpy())               

        return res

    def forward_blocks_range(
        self, x: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        for i in range(start, end):
            x = self.blocks[i](x)
        return x

    def add_pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            if self.pos_embed is not None:
                x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pos_embed is not None:
                x = x + self.pos_embed
        return x

    def forward_unflatten_sp_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.cls_token is not None:
            raise ValueError("Cannot unflatten sp_features when using cls_token")

        b, n, c = x.shape
        assert n == self.superpixel_shape[0] * self.superpixel_shape[1]
        x = x.transpose(1, 2).reshape(b, c, *self.superpixel_shape)
        return x
    @lru_cache
    def get_flattened_pixel_pos_embed(self, batch_size: int):
        pixel_features = self.pixel_pos_embed + torch.zeros([batch_size, 1, 1, 1, 1]).to(self.pixel_pos_embed.device)
        pixel_features = pixel_features.flatten(end_dim=1)
        return pixel_features

    def add_pos_embed_pixel(self, x: torch.Tensor) -> torch.Tensor:
        # [b, num_heads, 9, h, w]
        similarity = st.get_final_similarity(
            self.tokenization_info,
            self.patch_embed.num_blocks,
            pixel=self.use_pixel_similarities,
            merge=False,
        )
        if similarity is None:
            raise ValueError("Similarity is None")
        if self.pos_embed_method in ["pixel_no_grad", "both_no_grad"]:
            similarity = similarity.detach()


        b, c, sh, sw = x.shape
        pixel_features = self.get_flattened_pixel_pos_embed(x.size(0))
        # NOTE(meijieru): we know this func doesn't use sp_features, so we don't reshape for multihead.
        sp_delta = superpixel_ops.update_superpixel_features(
            pixel_features,
            x,
            similarity.flatten(end_dim=1)
        )


        sp_delta = sp_delta.reshape(b, c, sh, sw)
        return x + sp_delta
        # import h5py
        # with h5py.File('pe.h5', 'a') as f:  # 使用 'a' 模式以便在文件存在时进行追加
        #     keys = list(f.keys())
        #     key = "pixel_features"
        #     original_key = key
        #     count = 0
        #     while key in keys:
        #         print(f"Dataset with key {key} already exists. Updating key name.")
        #         count += 1
        #         key = original_key + str(count)
                
        #     f.create_dataset(key, data=pixel_features[:,:10,...].detach().cpu().numpy())
        #     key = "sp_delta"
        #     original_key = key
        #     count = 0
        #     while key in keys:
        #         print(f"Dataset with key {key} already exists. Updating key name.")
        #         count += 1
        #         key = original_key + str(count)
                
        #     f.create_dataset(key, data=sp_delta[:,:10,...].detach().cpu().numpy())
            
        #     key = "sp"
        #     original_key = key
        #     count = 0
        #     while key in keys:
        #         print(f"Dataset with key {key} already exists. Updating key name.")
        #         count += 1
        #         key = original_key + str(count)

        #     f.create_dataset(key, data=x[:,:10,...].detach().cpu().numpy())
            
        #     key = "pe+sp"
        #     original_key = key
        #     count = 0
        #     while key in keys:
        #         print(f"Dataset with key {key} already exists. Updating key name.")
        #         count += 1
        #         key = original_key + str(count)

        #     f.create_dataset(key, data=(x+sp_delta)[:,:10,...].detach().cpu().numpy())
            

    def add_sim_embed(self,x,info):
        patch_embed = self.patch_embed
        for key, val in info.items():
            if "similarities_pixel" in key:
                similarities = prepare_similarities(
                    patch_embed, val, scale_factor=1, merge_multihead_similarities=True
                )
                labels = superpixel_ops.compute_hard_association(similarities)
                # pos_emb = build_2d_sincos_position_embedding(patches_resolution = patch_embed.superpixel_shape[0],embed_dim = x.shape[-1],device = x.device)
                pos_emb = positionalencoding1d(d_model=x.shape[-1], length = patch_embed.superpixel_shape[0]* patch_embed.superpixel_shape[1], device = x.device)
                pos_emb = pos_emb.unsqueeze(0)
                b, h, w = labels.shape

                # Flatten labels tensor and expand dimensions to (4, 640 * 640, 1)
                labels_flat = labels.view(b, -1).unsqueeze(-1)

                # Use advanced indexing to gather from pos_emb
                labels_embedding = pos_emb[:, labels_flat].squeeze(-2).squeeze(0)

                # Reshape back to original shape (4, 640, 640, 4)
                labels_embedding = rearrange(
                    labels_embedding,
                    'b (h w) c->b c h w ',
                    h = h,
                    w = w)
                # import torch.nn.functional as F
                # labels_embedding_conv = self.emb_init(labels_embedding)
                labels_embedding_avg = F.avg_pool2d(labels_embedding,kernel_size = 4,stride = 4)
                # labels_embedding_max = F.max_pool2d(labels_embedding,kernel_size = 4,stride = 4)
                # import h5py
                # with h5py.File('labels_embedding.h5', 'a') as f:  # 使用 'a' 模式以便在文件存在时进行追加
                #     keys = list(f.keys())
                    
                #     original_key = key
                #     count = 0
                #     while key in keys:
                #         print(f"Dataset with key {key} already exists. Updating key name.")
                #         count += 1
                #         key = original_key + str(count)
                        
                #     f.create_dataset(key, data=labels_embedding.detach().cpu().numpy())
                    
                #     original_label_key = 'conv_' + original_key
                #     label_key = 'labels_' + original_label_key
                #     count = 0
                #     while label_key in keys:
                #         print(f"Dataset with key {label_key} already exists. Updating key name.")
                #         count += 1
                #         label_key = label_key + str(count)
                    
                #     f.create_dataset(label_key, data=labels_embedding_conv.detach().cpu().numpy())
                #     original_label_key = 'hard_' + original_key
                #     label_key = 'labels_' + original_label_key
                #     count = 0
                #     while label_key in keys:
                #         print(f"Dataset with key {label_key} already exists. Updating key name.")
                #         count += 1
                #         label_key = label_key + str(count)
                    
                #     f.create_dataset(label_key, data=labels.detach().cpu().numpy())
                  
                    # original_label_key = 'avg_' + original_key
                    # label_key = 'labels_' + original_label_key
                    # count = 0
                    # while label_key in keys:
                    #     print(f"Dataset with key {label_key} already exists. Updating key name.")
                    #     count += 1
                    #     label_key = label_key + str(count)
                    
                    # f.create_dataset(label_key, data=labels_embedding_avg.detach().cpu().numpy())
                    
                #     original_label_key = 'max_' + original_key
                #     label_key = 'labels_' + original_label_key
                #     count = 0
                #     while label_key in keys:
                #         print(f"Dataset with key {label_key} already exists. Updating key name.")
                #         count += 1
                #         label_key = label_key + str(count)
                    
                #     f.create_dataset(label_key, data=labels_embedding_max.detach().cpu().numpy())  
                labels_embedding = rearrange(
                    labels_embedding_avg,
                'b c h w->b (h w) c')
                x = x + labels_embedding   
        return x
    def forward(
        self,
        x: torch.Tensor,
        sp_features_last: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        if sp_features_last.dim() == 3:
            b, n, c = sp_features_last.shape
            h_w = int(math.sqrt(n))
            if h_w * h_w != n:
                raise ValueError("n must be a perfect square to reshape to (h, w).")
            sp_features_last = einops.rearrange(
                sp_features_last,
                'b (h w) c -> b c h w', 
                b=b,
                c=c,
                h=h_w,
                w=h_w
                
            )
        info, sp_features, pixel_features_middle = self.forward_patchify(
            x, sp_features_last
        )
        assert sp_features.dim() == 4
        sp_features = self.sp_lift(sp_features)
        if self.pos_embed_method in ["pixel", "pixel_no_grad", "both", "both_no_grad"]:
            sp_features = self.add_pos_embed_pixel(sp_features)
        sp_features = sp_features.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if False:
            sp_features = self.add_sim_embed(sp_features,info)
        # if self.pos_embed_method == "fixed":
        if True:
            sp_features = self.add_pos_embed(sp_features)

        # sp_features = self.blocks(sp_features)

        # [0, seg_block_idx) are the blocks for segmentation
        sp_features_seg = self.forward_blocks_range(sp_features, 0, self.seg_block_idx)
        sp_features = self.forward_blocks_range(
            sp_features_seg, self.seg_block_idx, len(self.blocks)
        )

        sp_features_unflatten = None
        updated_pixel_features = None
        if self.return_updated_pixel_features:
            sp_features_unflatten = self.forward_unflatten_sp_features(sp_features)
            sp_features_unflatten_projected = self.sp_project(sp_features_unflatten)
            updated_pixel_features = self.forward_pixelify(
                x if not self.use_middle_pixel_features else pixel_features_middle,
                sp_features_unflatten_projected,
                info,
            )
            if isinstance(
                self.patch_embed, st.SuperPixelTokenizationCrossAttentionAsymmetry
            ):
                pass
            else:
                sp_features_unflatten = sp_features_unflatten_projected

        if self.unflatten_sp_features:
            if sp_features_unflatten is None:
                sp_features_unflatten = self.forward_unflatten_sp_features(sp_features)
            sp_features = sp_features_unflatten

        return (
            updated_pixel_features,
            sp_features,
            sp_features_seg,
        )

@HEADS.register_module()
class SuperformerBottleNeck_ori(BaseDecodeHead):
    def __init__(
        self,
        img_size: Sequence[int] = (224,224),
        in_channels: int = 3,
        num_classes: int = 1000,
        stem_channels_list: Sequence[int] = (64,),
        stem_kernel_sizes: Sequence[int] = (4,),
        stem_strides: Sequence[int] = (4,),
        stem_conv_types: Sequence[str] = ("patch",),
        depths: Sequence[int] = (2, 10),
        dims: Sequence[int] = (384, 384),
        heads: Sequence[int] = (6, 6),
        strides: Sequence[int] = (1, 1),
        sp_sizes: Sequence[int] = (4, 4),
        sp_heads: Sequence[int] = (1, 1),
        sp_method: str = "sp_cross",
        sp_iter: int = 2,
        sp_features_init_methods: Sequence[str] = ("avgpool", "from_feature"),
        sp_global_init_method: str = "none",
        sp_position_embedding_method: str = "depthwise",
        sp_position_embedding_stride: int = -1,
        sp_sp_position_embedding_stride: Optional[int] = None,
        sp_pixel_features_update_method: str = "residual",
        sp_kwargs: Optional[Dict[str, Any]] = None,
        conv_norm_layer=None,
        conv_act_layer=None,
        norm_layer=None,
        act_layer=None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        ls_init_value: Optional[float] = 1e-5,
        sp_ls_init_value: Optional[float] = 1e-5,
        pre_norm_pixel: bool = False,
        pixel_refine_method: str = "identity",
        sp_embed_method: str = "identity",
        sp_project_method: str = "noact",
        class_token_position_method: str = "none",
        pos_embed_position_method: str = "every_stage",
        pos_embed_method: str = "fixed",
        no_embed_class: bool = False,
        weight_init: str = "default",
        use_pixel_similarities: bool = False,
        use_middle_pixel_features: bool = False,
        seg_specific_classifier: str = None,
        seg_num_classes: int = 150,
        use_stem: bool = True,
        classification_feature: str = 'superpixel',
        pixel_projection = None,
        use_compact_loss: bool= False,
        **kwargs

    ):
        super().__init__(in_channels=3,
        channels=256,
        num_classes=seg_num_classes,
        out_channels=seg_num_classes,
        in_index=0,
**kwargs)

        if sp_pixel_features_update_method == "fixed" and use_pixel_similarities:
            raise ValueError("pixel_similarities won't be updated")
        self.img_size = img_size
        self.sp_method = sp_method
        self.sp_iter = sp_iter
        self.sp_dim=dims[0]
        self.sp_features_init_methods = sp_features_init_methods
        self.sp_position_embedding_method = sp_position_embedding_method
        self.sp_position_embedding_stride = sp_position_embedding_stride
        self.sp_sp_position_embedding_stride = sp_sp_position_embedding_stride
        self.sp_pixel_features_update_method = sp_pixel_features_update_method
        self.sp_ls_init_value = sp_ls_init_value
        self.sp_kwargs = sp_kwargs or {}
        self.use_middle_pixel_features = use_middle_pixel_features

        self.class_token_position_method = class_token_position_method
        self.no_embed_class = no_embed_class
        self.pos_embed_position_method = pos_embed_position_method
        self.pos_embed_method = pos_embed_method

        act_layer = act_layer or nn.GELU
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # norm_layer_2d = partial(get_norm_layer_2d(norm_layer), eps=1e-6)
        norm_layer_2d = partial(get_norm_layer_2d(norm_layer))

        self.norm_layer = norm_layer
        self.norm_layer_2d = norm_layer_2d
        
        self.seg_num_classes = seg_num_classes
        self.seg_specific_classifier = seg_specific_classifier

        conv_norm_layer = conv_norm_layer or partial(timm_layers.LayerNorm2d, eps=1e-6)
        conv_act_layer = conv_act_layer or act_layer
        
        stem_kernel_sizes = stem_kernel_sizes or [3] * len(stem_channels_list)
        self.use_stem=use_stem
        if self.use_stem is True:
            self.stem = st.ConvStem(
                in_channels,
                stem_channels_list,
                stem_kernel_sizes,
                stem_strides,
                conv_types=stem_conv_types,
                # norm_layer=conv_norm_layer,  # pyright: ignore [reportGeneralTypeIssues]
                # act_layer=conv_act_layer,  # pyright: ignore [reportGeneralTypeIssues]
            )

        cur_stride = int(np.prod(stem_strides))
        # pixel_output_shape = tuple([img_size // cur_stride for _ in range(2)])
        pixel_dim = stem_channels_list[-1]
        sp_dim = None
        self.sp_global_init_method = sp_global_init_method
        if self.sp_features_init_methods[0] == "from_feature":
            assert self.sp_method in ["sp_cross_asymmetry"]
            sp_dim = dims[0]
            if self.sp_global_init_method == "patch":
                self.sp_init = nn.Sequential(
                    timm_layers.create_conv2d(
                        in_channels=pixel_dim,
                        out_channels=sp_dim,
                        kernel_size=sp_sizes[0],
                        stride=sp_sizes[0],
                        padding="same",
                        bias=False,
                    ),
                    norm_layer_2d(sp_dim),
                )
            elif self.sp_global_init_method == "avgpool_lift":
                self.sp_init = nn.Sequential(
                    nn.AvgPool2d(kernel_size=sp_sizes[0], stride=sp_sizes[0]),
                    timm_layers.create_conv2d(pixel_dim, sp_dim, 1, padding="same"),
                    norm_layer_2d(sp_dim),
                    act_layer(),
                )
            elif self.sp_global_init_method == "lift_avgpool":
                self.sp_init = nn.Sequential(
                    timm_layers.create_conv2d(pixel_dim, sp_dim, 1, padding="same"),
                    norm_layer_2d(sp_dim),
                    act_layer(),
                    nn.AvgPool2d(kernel_size=sp_sizes[0], stride=sp_sizes[0]),
                )
            elif self.sp_global_init_method in ["conv"]:
                self.sp_init = nn.Sequential(
                    timm_layers.create_conv2d(
                        in_channels=pixel_dim,
                        out_channels=sp_dim,
                        kernel_size=sp_sizes[0],
                        stride=sp_sizes[0],
                        padding="same",
                        bias=False,
                    ),
                    norm_layer_2d(sp_dim),
                    act_layer(),
                )
            elif self.sp_global_init_method == "stem":
                    self.sp_init = nn.Sequential(
                    nn.Conv2d(pixel_dim, sp_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(sp_dim // 2),
                    
                    nn.Conv2d(sp_dim // 2, sp_dim // 2, 3, 1, 1),
                    nn.GELU(),
                    nn.BatchNorm2d(sp_dim // 2),
                                
                    nn.Conv2d(sp_dim // 2, sp_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(sp_dim),            
                    
                    nn.Conv2d(sp_dim, sp_dim, 3, 1, 1),
                    nn.GELU(),
                    nn.BatchNorm2d(sp_dim),
                )
            else:
                raise ValueError(
                    f"Unknown sp_global_init_method: {sp_global_init_method}"
                )
        else:
            assert self.sp_global_init_method == "none"

        assert (
            len(depths)
            == len(dims)
            == len(heads)
            == len(sp_sizes)
            == len(sp_heads)
            == len(sp_features_init_methods)
        )

        drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        cur_depth = 0

        if self.pos_embed_position_method == "every_stage":
            use_pos_embeds = [True] * len(depths)
        elif self.pos_embed_position_method == "none":
            use_pos_embeds = [False] * len(depths)
        else:
            raise ValueError(
                f"Unknown pos_embed_position_method: {self.pos_embed_position_method}"
            )

        if self.class_token_position_method == "last_stage":
            use_class_tokens = [False] * (len(depths) - 1) + [True]
        elif self.class_token_position_method == "none":
            use_class_tokens = [False] * len(depths)
        else:
            raise ValueError(
                f"Unknown class_token_position_method: {self.class_token_position_method}"
            )

        num_stages = len(depths)
        stages = []


        stage_in_dim = stem_channels_list[-1]

        for i, (depth, dim, head, sp_size, sp_head, stride) in enumerate(
            zip(depths, dims, heads, sp_sizes, sp_heads, strides)
        ):
            cur_stride *= stride
            sp_stride = sp_size * stride // sp_sizes[i - 1] if i > 0 else 1
            sp_layer, sp_shape = self._make_superpixel_layer(
                i, cur_stride, sp_size, sp_head, pixel_dim, sp_dim, sp_method
            )

            # first feature already has norm & act
            pre_norm_pixel_stage = pre_norm_pixel and i != 0
            if pre_norm_pixel_stage:
                print(f"pre_norm_pixel for stage {i}")

            # return_updated_pixel_features = i < num_stages - 1
            return_updated_pixel_features = i < num_stages 

            unflatten_sp_features = return_updated_pixel_features

            print(
                f"stage {i}: use_cls_token: {use_class_tokens[i]}, use_pos_embed: {use_pos_embeds[i]}"
            )

            stages.append(
                SuperformerStage(
                    stage_in_dim,
                    sp_dim,
                    dim,
                    stride,
                    sp_stride,
                    head,
                    depth,
                    norm_layer=norm_layer,
                    norm_layer_2d=norm_layer_2d,
                    act_layer=act_layer,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rates[cur_depth : cur_depth + depth],
                    attn_drop_rate=attn_drop_rate,
                    superpixel_layer=sp_layer,
                    # ls_init_value=None if i == 2 else ls_init_value,
                    ls_init_value=ls_init_value,
                    pre_norm_pixel=pre_norm_pixel_stage,
                    sp_embed_method=sp_embed_method,
                    sp_project_method=sp_project_method,
                    pixel_refine_method=pixel_refine_method,
                    return_updated_pixel_features=return_updated_pixel_features,
                    unflatten_sp_features=unflatten_sp_features,
                    use_pos_embed=use_pos_embeds[i],
                    use_cls_token=use_class_tokens[i],
                    pos_embed_method=pos_embed_method,
                    no_embed_class=self.no_embed_class,
                    superpixel_shape=sp_shape,
                    use_pixel_similarities=use_pixel_similarities,
                    use_middle_pixel_features=use_middle_pixel_features,
                )
            )
            cur_depth += depth

        assert cur_depth == sum(depths)
        self.stages = nn.ModuleList(stages)

        # Classifier Head
        self.embed_dim = dims[-1]

        has_class_token = class_token_position_method != "none"
        use_fc_norm = not has_class_token
        self.num_prefix_tokens = 1 if has_class_token else 0
        self.norm = norm_layer(self.embed_dim) if not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(self.embed_dim) if use_fc_norm else nn.Identity()
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.classification_feature = classification_feature
        self.pixel_projection = pixel_projection
        if self.seg_specific_classifier:
            assert seg_num_classes > 0
            # NOTE(meijieru): the token features doesn't use avgpool, so we skip
            # the fc_norm.
            
            if self.classification_feature =='superpixel':

                if self.seg_specific_classifier == "Linear":
                    self.seg_head = nn.Linear(self.embed_dim, self.seg_num_classes)
                elif self.seg_specific_classifier == "Conv":
                    self.seg_head_conv = nn.Conv2d(self.embed_dim, self.seg_num_classes,1)
                else:
                    print(self.seg_specific_classifier)
                    raise ValueError()
                self.seg_norm = norm_layer(self.embed_dim)
        
            elif self.classification_feature =='pixel':
                if self.pixel_projection:
                    self.pixel_projection = nn.Conv2d(stem_channels_list[-1], self.embed_dim //2, 1)   
                    self.seg_norm = norm_layer(self.embed_dim //2)
                    self.seg_head = nn.Linear(self.embed_dim //2, self.seg_num_classes)
                else:
                    self.seg_norm = norm_layer(stem_channels_list[-1])
                    self.seg_head = nn.Linear(stem_channels_list[-1], self.seg_num_classes)
            elif self.classification_feature =='both':
                self.seg_norm = norm_layer(stem_channels_list[-1] + self.embed_dim)
                self.seg_head = nn.Linear(stem_channels_list[-1] + self.embed_dim, self.seg_num_classes)
            elif self.classification_feature =='both_dualhead':     
                self.seg_norm_pixel = norm_layer(stem_channels_list[-1])
                self.seg_head_pixel = nn.Linear(stem_channels_list[-1], self.seg_num_classes)
                self.seg_norm = norm_layer(self.embed_dim)
                self.seg_head = nn.Linear(self.embed_dim, self.seg_num_classes)
        if weight_init != "skip":
            self.init_weights(weight_init)
        self.use_compact_loss =use_compact_loss
    def init_weights(self, mode=""):
        assert mode in (
            "jax",
            "jax_nlhb",
            "moco",
            "",
            "default",
        )
        if mode == "default" or mode == "":
            init_fn = init_superformer_weights
        else:
            raise NotImplementedError(f"Unknown init mode: {mode}")
        
        timm.models.named_apply(init_fn, self)
        # timm.models.helpers.named_apply(init_fn, self)

    def _make_superpixel_layer(
        self,
        i: int,
        pixel_stride: int,
        sp_size: int,
        sp_head: int,
        pixel_dim: int,
        sp_dim: Optional[int],
        method: str = "sp_cross",
    ) -> Tuple[Callable, Sequence[int]]:
        assert self.img_size[0] % pixel_stride == 0
        pixel_shape = [self.img_size[i] // pixel_stride for i in range(2)]
        sp_shape = [val // sp_size for val in pixel_shape]

        if (
            i != len(self.sp_features_init_methods) - 1
            and self.use_middle_pixel_features
        ):
            return_final_pixel_features = True
        else:
            return_final_pixel_features = True
        if method == "sp_cross":
            sp_fn = partial(
                st.SuperPixelTokenizationCrossAttention,
                self.sp_iter,
                sp_head,
                pixel_dim,
                pixel_shape,
                sp_shape,
                superpixel_features_init_method=self.sp_features_init_methods[i],
                position_embedding_method=self.sp_position_embedding_method,
                position_embedding_stride=self.sp_position_embedding_stride,
                sp_position_embedding_stride=self.sp_sp_position_embedding_stride,
                pixel_features_update_method=self.sp_pixel_features_update_method,
                return_final_pixel_features=return_final_pixel_features,
                layer_kwargs={
                    "ls_init_value": self.sp_ls_init_value,
                    "norm_layer": self.norm_layer_2d,
                },
                **self.sp_kwargs,
            )
        elif method == "sp_cross_asymmetry":
            assert sp_dim is not None
            sp_fn = partial(
                st.SuperPixelTokenizationCrossAttentionAsymmetry,
                self.sp_iter,
                sp_head,
                pixel_dim,
                pixel_shape,
                sp_shape,
                superpixel_features_init_method=self.sp_features_init_methods[i],
                position_embedding_method=self.sp_position_embedding_method,
                position_embedding_stride=self.sp_position_embedding_stride,
                sp_position_embedding_stride=self.sp_sp_position_embedding_stride,
                pixel_features_update_method=self.sp_pixel_features_update_method,
                return_final_pixel_features=return_final_pixel_features,
                num_channels_sp=sp_dim,
                num_heads_sp=sp_head,
                layer_kwargs={
                    "ls_init_value": self.sp_ls_init_value,
                    "norm_layer": self.norm_layer_2d,
                },
                **self.sp_kwargs,
            )
        elif method == "patch":
            sp_fn = partial(nn.AvgPool2d, kernel_size=sp_size, stride=sp_size)
        else:
            raise ValueError(f"Unknown superpixel method {method}")
        return sp_fn, sp_shape

    def init_superpixel_features(
        self, pixel_features: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.sp_features_init_methods[0] == "from_feature":
            return self.sp_init(pixel_features)
        elif self.sp_features_init_methods[0] == "leanable":
            return nn.Parameter(
                torch.randn(1, self.sp_dim, self.superpixel_shape[0],self.superpixel_shape[1]) * 0.02
            )
            return 
        return None

    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, MutableMapping[str, torch.Tensor]]:
        if self.use_stem == True:
            endpoints, pixel_features = self.stem(x)
        else:
            endpoints=None
            pixel_features=x
        sp_features_last = self.init_superpixel_features(pixel_features)
        # import h5py
        # with h5py.File('/data2/yunfei/vis/v2.h5','a') as f:
        #     keys = list(f.keys())
        #     key = "sp_features_first"
        #     original_key = key
        #     count = 0
        #     while key in keys:
        #         print(f"Dataset with key {key} already exists. Updating key name.")
        #         count += 1
        #         key = original_key + str(count)
                
        #     f.create_dataset(key, data=sp_features_last.detach().cpu().numpy())

        assert len(self.stages) > 0
        for _, stage in enumerate(self.stages):
            pixel_features, sp_features, sp_features_seg = stage(
                pixel_features,
                sp_features_last,
            )
            sp_features_last = sp_features_seg

            # # rank not consistent due to whether flatten or not
            # # res[f"sp_features_stage{i}"] = sp_features
            # if return_updated_pixel_features:
            #     endpoints[f"pixel_features_stage{i}"] = pixel_features
        return sp_features, sp_features_seg, endpoints, pixel_features

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.norm(x)
        assert x.dim() == 3
        if self.class_token_position_method == "none":
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
        else:
            x = x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward_segmentation(
        self, x: torch.Tensor, return_pixel_logits: bool = True, stride: int = 2,pixel_feature: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.classification_feature == "pixel":
            b, c, h, w = x.shape
            if self.pixel_projection:
                x = self.pixel_projection(x)
            pixel_feature = self.seg_norm(rearrange(x,
                                        "b c h w -> b (h w) c"))
            pixel_logits = self.seg_head(pixel_feature)
            pixel_logits = rearrange(pixel_logits,
                                     "b (h w) c -> b c h w",
                                     h = h,w = w)
            
            return None, pixel_logits
        elif self.classification_feature == "superpixel":
            if not self.seg_specific_classifier:
                raise ValueError("No segmentation head is found.")
            elif self.seg_specific_classifier=="Linear":
                x = self.seg_norm(x)
                b, num, c = x.shape
                x = x.reshape(b * num, c)

                sp_logits = self.seg_head(x)

                last_stage = self.stages[-1]
                last_sp_layer = last_stage.patch_embed

                if isinstance(last_sp_layer, nn.AvgPool2d):
                    sh=sw = last_sp_layer.kernel_size
                elif isinstance(last_sp_layer, st.SuperPixelTokenization):
                    sh, sw = last_sp_layer.superpixel_shape
                else:
                    raise ValueError()
                sp_logits = sp_logits.view(b, sh, sw, -1).permute(0, 3, 1, 2)
            elif self.seg_specific_classifier=="Conv":
                
                last_stage = self.stages[-1]
                last_sp_layer = last_stage.patch_embed

                if isinstance(last_sp_layer, nn.AvgPool2d):
                    sh=sw = last_sp_layer.kernel_size
                elif isinstance(last_sp_layer, st.SuperPixelTokenization):
                    sh, sw = last_sp_layer.superpixel_shape
                else:
                    raise ValueError()
                x = self.seg_norm(x)
                b, num, c = x.shape

                x = x.view(b,sh,sw,-1).permute(0, 3, 1, 2) #B,C,sh,sw
                sp_logits = self.seg_head_conv(x)

                

            pixel_logits = None
            if return_pixel_logits:
                if isinstance(last_sp_layer, nn.AvgPool2d):
                    raise NotImplementedError()
                elif isinstance(last_sp_layer, st.SuperPixelTokenization):
                    info = last_stage.tokenization_info
                    if info is None:
                        raise ValueError()
                    scale_factor = self.img_size[0] // last_sp_layer.pixel_shape[0] // stride
                    # scale_factor = 1
                    # raise NotImplementedError(
                    #     "TODO(meijier): use pixel similarities & not merge"
                    # )

                    similarities = st.get_final_similarity(info, last_sp_layer.num_blocks)
                    similarities = prepare_similarities(
                        last_sp_layer,
                        similarities,  # pyright: ignore [reportGeneralTypeIssues]
                        scale_factor=scale_factor,
                    )
                    similarities = similarities.softmax(1)
                    similarities = einops.rearrange(
                        similarities, "b n sh ph sw pw -> b n (sh ph) (sw pw)"
                    )
                    # import h5py
                    # with h5py.File("/data2/yunfei/vis/similarities.h5","a") as f:
                    #     f.create_dataset("similarities",data=similarities.detach().cpu().numpy())
                    pixel_logits = superpixel_ops.expand_superpixel_features(
                        sp_logits, similarities
                    )
                else:
                    raise ValueError()

            return sp_logits, pixel_logits
        elif self.classification_feature == "both":
            last_stage = self.stages[-1]
            last_sp_layer = last_stage.patch_embed
            if isinstance(last_sp_layer, nn.AvgPool2d):
                sh=sw = last_sp_layer.kernel_size
            elif isinstance(last_sp_layer, st.SuperPixelTokenization):
                sh, sw = last_sp_layer.superpixel_shape
            else:
                raise ValueError()
            x = rearrange(
                x,
                'b (sh sw) c -> b c sh sw',
                sh = sh, sw= sw
            )
            pixel_logits = None
            if return_pixel_logits:
                if isinstance(last_sp_layer, nn.AvgPool2d):
                    raise NotImplementedError()
                elif isinstance(last_sp_layer, st.SuperPixelTokenization):
                    info = last_stage.tokenization_info
                    if info is None:
                        raise ValueError()
                    # scale_factor = self.img_size[0] // last_sp_layer.pixel_shape[0] // stride
                    scale_factor = 1
                    # raise NotImplementedError(
                    #     "TODO(meijier): use pixel similarities & not merge"
                    # )

                    similarities = st.get_final_similarity(info, last_sp_layer.num_blocks)
                    similarities = prepare_similarities(
                        last_sp_layer,
                        similarities,  # pyright: ignore [reportGeneralTypeIssues]
                        scale_factor=scale_factor,
                    )
                    similarities = similarities.softmax(1)
                    similarities = einops.rearrange(
                        similarities, "b n sh ph sw pw -> b n (sh ph) (sw pw)"
                    )
                    pixel_logits = superpixel_ops.expand_superpixel_features(
                        x, similarities
                    )
                else:
                    raise ValueError()
            pixel_logits = torch.concat((pixel_logits,pixel_feature),dim=1)
            pixel_logits = rearrange(
                pixel_logits,
                'b c h w ->b (h w) c')
            pixel_logits = self.seg_norm(pixel_logits)
            pixel_logits = self.seg_head(pixel_logits)
            pixel_logits = rearrange(
                pixel_logits,
                'b (h w) c->b c h w',
                h=sh*4, w=sw*4)
            return None, pixel_logits
        elif self.classification_feature == "both_dualhead":
            if not self.seg_specific_classifier:
                raise ValueError("No segmentation head is found.")
            elif self.seg_specific_classifier=="Linear":
                x = self.seg_norm(x)
                b, num, c = x.shape
                x = x.reshape(b * num, c)

                sp_logits = self.seg_head(x)

                last_stage = self.stages[-1]
                last_sp_layer = last_stage.patch_embed

                if isinstance(last_sp_layer, nn.AvgPool2d):
                    sh=sw = last_sp_layer.kernel_size
                elif isinstance(last_sp_layer, st.SuperPixelTokenization):
                    sh, sw = last_sp_layer.superpixel_shape
                else:
                    raise ValueError()
                sp_logits = sp_logits.view(b, sh, sw, -1).permute(0, 3, 1, 2)
            elif self.seg_specific_classifier=="Conv":
                
                last_stage = self.stages[-1]
                last_sp_layer = last_stage.patch_embed

                if isinstance(last_sp_layer, nn.AvgPool2d):
                    sh=sw = last_sp_layer.kernel_size
                elif isinstance(last_sp_layer, st.SuperPixelTokenization):
                    sh, sw = last_sp_layer.superpixel_shape
                else:
                    raise ValueError()
                x = self.seg_norm(x)
                b, num, c = x.shape

                x = x.view(b,sh,sw,-1).permute(0, 3, 1, 2) #B,C,sh,sw
                sp_logits = self.seg_head_conv(x)

                

            pixel_logits = None
            if return_pixel_logits:
                if isinstance(last_sp_layer, nn.AvgPool2d):
                    raise NotImplementedError()
                elif isinstance(last_sp_layer, st.SuperPixelTokenization):
                    info = last_stage.tokenization_info
                    if info is None:
                        raise ValueError()
                    # scale_factor = self.img_size[0] // last_sp_layer.pixel_shape[0] // stride
                    scale_factor = 1
                    # raise NotImplementedError(
                    #     "TODO(meijier): use pixel similarities & not merge"
                    # )

                    similarities = st.get_final_similarity(info, last_sp_layer.num_blocks)
                    similarities = prepare_similarities(
                        last_sp_layer,
                        similarities,  # pyright: ignore [reportGeneralTypeIssues]
                        scale_factor=scale_factor,
                    )
                    similarities = similarities.softmax(1)
                    similarities = einops.rearrange(
                        similarities, "b n sh ph sw pw -> b n (sh ph) (sw pw)"
                    )
                    pixel_logits = superpixel_ops.expand_superpixel_features(
                        sp_logits, similarities
                    )
                else:
                    raise ValueError()
            b, c, h, w = pixel_feature.shape
            pixel_feature = self.seg_norm_pixel(rearrange(pixel_feature,
                                        "b c h w -> b (h w) c"))
            pixel_logits_ = self.seg_head_pixel(pixel_feature)
            pixel_logits_ = rearrange(pixel_logits_,
                                     "b (h w) c -> b c h w",
                                     h = h,w = w)
            pixel_logits = pixel_logits + pixel_logits_
            return sp_logits, pixel_logits
    def forward(
        self,
        x: torch.Tensor,
        generate_seg: bool = True,
        return_pixel_logits: bool = True,
        seg_stride: int = 1,

    ) -> Union[torch.Tensor, MutableMapping[str, torch.Tensor]]:
        
        sp_features, sp_features_seg, endpoints, pixel_features = self.forward_features(x)
        h = w = int(math.sqrt(sp_features_seg.shape[1]))
        if self.use_compact_loss:
            if self.classification_feature == "superpixel":
                restruct_pixel_features = self.compute_compact_loss(pixel_features,sp_features_seg)
                if generate_seg:
                        sp_logits, pixel_logits = self.forward_segmentation(
                        sp_features_seg, return_pixel_logits, stride=seg_stride
                    )
                        ret={}
                        ret["seg"] = pixel_logits
                        ret['restruct'] = restruct_pixel_features
                        ret['pixel_feature'] = rearrange(sp_features_seg,
                                                         'b (h w) c -> b c h w',
                                                         h = h, w =w)
                        
                        return ret
        elif self.classification_feature == "superpixel":
            if generate_seg:
                    sp_logits, pixel_logits = self.forward_segmentation(
                    sp_features_seg, return_pixel_logits, stride=seg_stride
                )
                    ret={}
                    ret["seg"] = pixel_logits
                    return ret["seg"]
        elif self.classification_feature == "pixel":
            if generate_seg:
                    sp_logits, pixel_logits = self.forward_segmentation(
                    pixel_features, return_pixel_logits, stride=seg_stride
                )
                    ret={}
                    ret["seg"] = pixel_logits
                    return ret["seg"]
        elif self.classification_feature == "both":
            if generate_seg:
                    sp_logits, pixel_logits = self.forward_segmentation(
                    sp_features_seg, return_pixel_logits, stride=seg_stride, pixel_feature=pixel_features
                )
                    ret={}
                    ret["seg"] = pixel_logits
                    return ret["seg"]
        elif self.classification_feature == "both_dualhead":
            if generate_seg:
                    sp_logits, pixel_logits = self.forward_segmentation(
                    sp_features_seg, return_pixel_logits, stride=seg_stride, pixel_feature=pixel_features
                )
                    ret={}
                    ret["seg"] = pixel_logits
                    return ret["seg"]
        else:
            logits = self.forward_head(sp_features)
            return logits
    def compute_compact_loss(self, x: torch.Tensor,sp_features: torch.Tensor):
        last_stage = self.stages[-1]
        last_sp_layer = last_stage.patch_embed
        assert isinstance(last_sp_layer, st.SuperPixelTokenization)
        info = last_stage.tokenization_info
        sh, sw = last_sp_layer.superpixel_shape
        sp_features = rearrange(
            sp_features,
            'b (h w) c -> b c h w',
            h = sh, w = sw
        )
        similarities_sp = st.get_final_similarity(info, last_sp_layer.num_blocks,pixel=False)
        similarities_pixel = st.get_final_similarity(info, last_sp_layer.num_blocks,pixel=True)
        if True:
            pixel_features = superpixel_ops.update_pixel_features(None, sp_features,similarities_pixel)            
            sp_features = superpixel_ops.update_superpixel_features(pixel_features, sp_features, similarities_sp,hard_assign = True)
        else:
            sp_features = superpixel_ops.update_superpixel_features(x, sp_features, similarities_sp)
            pixel_features = superpixel_ops.update_pixel_features(None, sp_features,similarities_pixel)
        # import h5py
        # with h5py.File("/data2/yunfei/vis/restruc.h5","a") as f:
        #     f.create_dataset("pixel_before",data=x[:,:10,:,:].detach().cpu().numpy())
        #     f.create_dataset("pixel_after",data=pixel_features[:,:10,:,:].detach().cpu().numpy())
        return sp_features
    def no_weight_decay(self):
        no_weight_decay = set()
        for name, _ in self.named_parameters():
            if (
                "pos_embed" in name
                or "init_superpixel_queries" in name
                or "cls_token" in name
            ):
                no_weight_decay.add(name)
        print(f"no_weight_decay: {no_weight_decay}")
        return no_weight_decay

    def visualize_superpixel(self, resize_similarities: bool = True):
        res = {}
        for i, stage in enumerate(self.stages):
            patch_embed = stage.patch_embed
            tokenization_info = stage.tokenization_info
            if resize_similarities:
                scale_factor = self.img_size[0] // stage.patch_embed.pixel_shape[0]
            else:
                scale_factor = 1
            if tokenization_info is None:
                continue
            res_i = visualize_superpixel(tokenization_info, patch_embed, scale_factor)
            res.update({f"stage{i}_{key}": val for key, val in res_i.items()})
        return res


@register_model
def asym_bottleneck_small_nofinal_head4(pretrained=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    kwargs.pop("pretrained_cfg_overlay", None)

    if pretrained:
        raise NotImplementedError()

    defaults = dict(
        sp_global_init_method="patch",
        sp_method="sp_cross_asymmetry",
        sp_features_init_methods=("from_feature", "from_feature"),
        sp_heads=(4, 4),
        sp_kwargs={
            "return_similarities_final": False,
        },
    )
    return SuperformerBottleNeck(**_update_params(defaults, **kwargs))


@register_model
def asym_bottleneck_conv4_small_nofinal_head4(**kwargs):
    return asym_bottleneck_small_nofinal_head4(
        **_update_params(dict(stem_conv_types=("conv",)), **kwargs)
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_patch77(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4(
        **_update_params(
            dict(
                sp_method="patch",
                sp_global_init_method="none",
                sp_features_init_methods=("avgpool", "avgpool"),
            ),
            **kwargs,
        )
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_patch77_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_patch77(
        sp_global_init_method="lift_avgpool",
        sp_features_init_methods=("avgpool", "avgpool"),
        **kwargs,
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_patch7(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_patch77(
        **_update_params(
            dict(
                depths=(12,),
                dims=(384,),
                heads=(6,),
                strides=(1,),
                sp_sizes=(4,),
                sp_heads=(1,),
                sp_features_init_methods=("avgpool",),
            ),
            **kwargs,
        )
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_patch7_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_patch7(
        **_update_params(
            dict(
                stem_channels_list=(64, 384),
                stem_conv_types=("conv", "lift_avgpool"),
                stem_kernel_sizes=(4, 4),
                stem_strides=(4, 4),
                sp_features_init_methods=("avgpool",),
                sp_sizes=(1,),
            ),
            **kwargs,
        )
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_patch7_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_patch7_spilavg(
        stem_kernel_sizes=(8, 4), stem_strides=(8, 4), **kwargs
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_patch7_spiavgl(**kwargs):
    return asym_bottleneck_conv8_small_nofinal_patch7_spilavg(
        stem_conv_types=("conv", "avgpool_lift"), **kwargs
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_head4_spiconv(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4(
        sp_global_init_method="conv", **kwargs
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_head4_spiavgl(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4(
        sp_global_init_method="avgpool_lift", **kwargs
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_head4_spiavgl(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spiavgl(
        stem_kernel_sizes=(8,), stem_strides=(8,), **kwargs
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_head4_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4(
        **_update_params(dict(sp_global_init_method="lift_avgpool"), **kwargs)
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_head4_spilavg_1stage(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        depths=(12,),
        dims=(384,),
        heads=(6,),
        strides=(1,),
        sp_sizes=(4,),
        sp_heads=(1,),
        sp_features_init_methods=("from_feature",),
        **kwargs,
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_head4_spilavg_iter1(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(sp_iter=1, **kwargs)


@register_model
def asym_bottleneck_conv4_small_nofinal_head2_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        **_update_params(dict(sp_heads=(2, 2)), **kwargs)
    )

    
@register_model
def asym_bottleneck_conv4_small_nofinal_head2_spilavg_seg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        **_update_params({'sp_heads': (2, 2),
                           'seg_specific_classifier': "Conv",
                           'seg_num_classes': 150},
                         **kwargs)
    )
@register_model
def asym_bottleneck_conv4_small_nofinal_head2_spilavg_seg_3stage(**kwargs): 
        return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
            **_update_params({'sp_heads': (2,2,2),
                           'seg_specific_classifier': "Conv",
                           'seg_num_classes': 150,
                            'depths':(12,12,12),
                            'dims':(384,384,384),
                            'heads':(6,6,6),
                            'strides':(1,1,1),
                            'sp_sizes':(4,4,4),
                            'sp_heads':(1,1,1),
                            'sp_features_init_methods':("from_feature","from_feature","from_feature")},
        **kwargs,
    )
    )

@register_model
def asym_bottleneck_conv4_small_nofinal_head2_spilavg_seg_4stage(**kwargs): 
        return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
            **_update_params({'sp_heads': (2,2,2,2),
                           'seg_specific_classifier': "Conv",
                           'seg_num_classes': 150,
                            'depths':(12,12,12,12),
                            'dims':(384,384,384,384),
                            'heads':(6,6,6,6),
                            'strides':(1,1,1,1),
                            'sp_sizes':(4,4,4,4),
                            'sp_heads':(2,2,2,2),
                            'sp_features_init_methods':("from_feature","from_feature","from_feature","from_feature")},
        **kwargs,
    )
    )

@register_model
def asym_bottleneck_conv4_base_nofinal_head2_spilavg_seg_2stage(**kwargs): 
        return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
            **_update_params({'sp_heads': (2,2),
                           'seg_specific_classifier': "Conv",
                           'seg_num_classes': 150,
                            'depths':(12,12),
                            'dims':(768,768),
                            'heads':(12,12),
                            'strides':(1,1,1),
                            'sp_sizes':(4,4),
                            'sp_heads':(3,3),
                            'sp_features_init_methods':("from_feature","from_feature"),
                            'stem_channels_list':(48, 96),
                            'stem_kernel_sizes':(4, 4),
                            'stem_strides':(4, 4),
                            'stem_conv_types':("conv", "conv"),
                            'pos_embed_position_method':"none"

                            },
        **kwargs,
    )
    )
@register_model
def asym_bottleneck_conv4_base_nofinal_head2_spilavg_seg_6stage(**kwargs): 
        return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
            **_update_params({'sp_heads': (2,2),
                           'seg_specific_classifier': "Conv",
                           'seg_num_classes': 150,
                            'depths':(6,6,6,6,6,6,),
                            'dims':(768,768,768,768,768,768),
                            'heads':(6,6,6,6,6,6),
                            'strides':(1,1,1,1,1,1),
                            'sp_sizes':(4,4,4,4,4,4),
                            'sp_heads':(3,3,3,3,3,3),
                            'sp_features_init_methods':("from_feature","from_feature","from_feature","from_feature","from_feature","from_feature"),
                            'stem_channels_list':(48, 96),
                            'stem_kernel_sizes':(4, 4),
                            'stem_strides':(4, 4),
                            'stem_conv_types':("conv", "conv"),
                            'pos_embed_position_method':"none"
                            },
        **kwargs,
    )
    )
@register_model
def asym_bottleneck_conv4_small_nofinal_head2_spilavg_seg_9stage(**kwargs): 
        return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
            **_update_params({'sp_heads': (2,2,2,2,2,2),
                           'seg_specific_classifier': "Conv",
                           'seg_num_classes': 150,
                            'depths':(6,6,6,3,3,3,2,2,2),
                            'dims':(384,384,384,384,384,384,384,384,384),
                            'heads':(6,6,6,6,6,6,6,6,6),
                            'strides':(1,1,1,1,1,1,1,1,1),
                            'sp_sizes':(4,4,4,4,4,4,4,4,4),
                            'sp_heads':(1,1,1,1,1,1,1,1,1),
                            'sp_features_init_methods':("from_feature","from_feature","from_feature","from_feature","from_feature","from_feature","from_feature","from_feature","from_feature"),
                            'pos_embed_position_method':"none"
},
        **kwargs,
    )
    )


@register_model
def asym_bottleneck_conv4_tiny_nofinal_head2_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head2_spilavg(
        **_update_params(
            dict(
                stem_channels_list=(32,),
                dims=(192, 192),
                heads=(3, 3),
            ),
            **kwargs,
        )
    )


@register_model
def asym_bottleneck_cc4_tiny_nofinal_head2_spilavg(**kwargs):
    return asym_bottleneck_conv4_tiny_nofinal_head2_spilavg(
        stem_kernel_sizes=(3, 3),
        stem_conv_types=("conv", "conv"),
        stem_channels_list=(16, 32),
        stem_strides=(2, 2),
        **kwargs,
    )


@register_model
def asym_bottleneck_conv4_tiny_nofinal_head1_spilavg(**kwargs):
    return asym_bottleneck_conv4_tiny_nofinal_head2_spilavg(
        sp_heads=(1, 1),
        **kwargs,
    )


@register_model
def asym_bottleneck_conv4_small_nofinal_head1_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(sp_heads=(1, 1), **kwargs)


@register_model
def asym_bottleneck_conv4_small_nofinal_head4_spilavg_psim(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        use_pixel_similarities=True, **kwargs
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_head4_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        stem_kernel_sizes=(8,), stem_strides=(8,), **kwargs
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_head2_spilavg(**kwargs):
    return asym_bottleneck_conv8_small_nofinal_head4_spilavg(
        **_update_params(dict(sp_heads=(2, 2)), **kwargs)
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_head2_spilavg_iter1(**kwargs):
    return asym_bottleneck_conv8_small_nofinal_head2_spilavg(sp_iter=1, **kwargs)


@register_model
def asym_bottleneck_conv8_small_nofinal_head2_spilavg_stage1(**kwargs):
    return asym_bottleneck_conv8_small_nofinal_head2_spilavg(
        depths=(12,),
        dims=(384,),
        heads=(6,),
        strides=(1,),
        sp_sizes=(4,),
        sp_heads=(1,),
        sp_features_init_methods=("from_feature",),
        **kwargs,
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_head2_spilavg_patchbynograd(**kwargs):
    model = asym_bottleneck_conv8_small_nofinal_head2_spilavg(**kwargs)
    for stage in model.stages:
        for block in stage.patch_embed.blocks:
            for ls in [block.pixel_ls1, block.sp_ls1]:
                if hasattr(ls, "gamma"):
                    ls.gamma.requires_grad_(False)
                    ls.gamma.zero_()
                else:
                    assert isinstance(ls, nn.Identity)

    return model


@register_model
def asym_bottleneck_conv8_small_nofinal_head2_spilavg_spprojact(**kwargs):
    return asym_bottleneck_conv8_small_nofinal_head2_spilavg(
        sp_project_method="act", **kwargs
    )


@register_model
def asym_bottleneck_conv8_small_nofinal_head1_spilavg(**kwargs):
    return asym_bottleneck_conv8_small_nofinal_head4_spilavg(sp_heads=(1, 1), **kwargs)


@register_model
def asym_bottleneck_conv8_small_nofinal_head2(**kwargs):
    return asym_bottleneck_conv8_small_nofinal_head4_spilavg(
        sp_heads=(2, 2), sp_global_init_method="patch", **kwargs
    )


@register_model
def asym_bottleneck_conv16_small_nofinal_head4_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        stem_kernel_sizes=(16,), stem_strides=(16,), **kwargs
    )


@register_model
def asym_bottleneck_cc4_small_nofinal_head4_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        **_update_params(
            dict(
                stem_kernel_sizes=(3, 3),
                stem_conv_types=("conv", "conv"),
                stem_channels_list=(32, 64),
                stem_strides=(2, 2),
            ),
            **kwargs,
        )
    )


@register_model
def asym_bottleneck_ccc8_small_nofinal_head4_spilavg(**kwargs):
    return asym_bottleneck_conv4_small_nofinal_head4_spilavg(
        **_update_params(
            dict(
                stem_kernel_sizes=(3, 3, 3),
                stem_conv_types=("conv", "conv", "conv"),
                stem_channels_list=(16, 32, 64),
                stem_strides=(2, 2, 2),
            ),
            **kwargs,
        )
    )


@register_model
def asym_bottleneck_ccc8_small_nofinal_head2_spilavg(**kwargs):
    return asym_bottleneck_ccc8_small_nofinal_head4_spilavg(sp_heads=(2, 2), **kwargs)


@register_model
def asym_bottleneck_ccc8_small_nofinal_head2_spilavg_patchbynograd(**kwargs):
    model = asym_bottleneck_ccc8_small_nofinal_head2_spilavg(**kwargs)
    for stage in model.stages:
        for block in stage.patch_embed.blocks:
            for ls in [block.pixel_ls1, block.sp_ls1]:
                if hasattr(ls, "gamma"):
                    ls.gamma.requires_grad_(False)
                    ls.gamma.zero_()
                else:
                    assert isinstance(ls, nn.Identity)

    return model


@register_model
def asym_bottleneck_ccc8_small_nofinal_head2_spilavg77(**kwargs):
    return asym_bottleneck_ccc8_small_nofinal_head4_spilavg(
        sp_heads=(2, 2), sp_sizes=(7, 7), **kwargs
    )


@register_model
def asym_bottleneck_cc4_small_nofinal_head2_spilavg(**kwargs):
    return asym_bottleneck_cc4_small_nofinal_head4_spilavg(
        **_update_params(dict(sp_heads=(2, 2)), **kwargs)
    )


@register_model
def asym_bottleneck_cc4_base_nofinal_head3_spilavg(**kwargs):
    return asym_bottleneck_cc4_small_nofinal_head2_spilavg(
        **_update_params(
            dict(
                stem_channels_list=(48, 96),
                dims=(768, 768),
                heads=(12, 12),
                sp_heads=(3, 3),
            ),
            **kwargs,
        )
    )


@register_model
def asym_bottleneck_conv4_base_nofinal_head3_spilavg(**kwargs):
    return asym_bottleneck_cc4_base_nofinal_head3_spilavg(
        **_update_params(
            dict(
                stem_channels_list=(96,),
                stem_kernel_sizes=(4,),
                stem_strides=(4,),
                stem_conv_types=("conv",),
            ),
            **kwargs,
        )
    )


# @register_model
# def asym_bottleneck_ccc8_small_nofinal_head2_spilavg(**kwargs):
#     return asym_bottleneck_cc4_small_nofinal_head2_spilavg(, **kwargs)


@register_model
def hr_bottleneck_small_nofinal(pretrained=False, **kwargs):
    kwargs.pop("pretrained_cfg", None)
    if pretrained:
        raise NotImplementedError()

    defaults = dict(
        sp_kwargs={
            "return_similarities_final": False,
            "pixel_qkv_method": "identity",
        },
    )
    return SuperformerBottleNeck(**_update_params(defaults, **kwargs))


@register_model
def hr_bottleneck_small_nofinal_pixelqkv(**kwargs):
    return hr_bottleneck_small_nofinal(
        sp_kwargs={
            "return_similarities_final": False,
        },
        **kwargs,
    )


@register_model
def hr_bottleneck_small_nofinal_pixelqkv_psim(**kwargs):
    return hr_bottleneck_small_nofinal_pixelqkv(
        **_update_params(
            dict(
                use_pixel_similarities=True,
            ),
            **kwargs,
        )
    )


@register_model
def hr_bottleneck_p8_small_nofinal_pixelqkv_psim(**kwargs):
    return hr_bottleneck_small_nofinal_pixelqkv_psim(
        stem_kernel_sizes=(8,), stem_strides=(8,), **kwargs
    )


@register_model
def hr_bottleneck_p8_small_nofinal_pixelqkv_psim_head4(**kwargs):
    return hr_bottleneck_p8_small_nofinal_pixelqkv_psim(sp_heads=(4, 4), **kwargs)


@register_model
def hr_bottleneck_p8_small_nofinal_pixelqkv_head4(**kwargs):
    return hr_bottleneck_p8_small_nofinal_pixelqkv_psim_head4(
        use_pixel_similarities=False, **kwargs
    )


@register_model
def hr_bottleneck_p8_small_nofinal_pixelqkv_psim_head2(**kwargs):
    return hr_bottleneck_p8_small_nofinal_pixelqkv_psim(sp_heads=(2, 2), **kwargs)


@register_model
def hr_bottleneck_small_nofinal_pixelqkv_head2(**kwargs):
    return hr_bottleneck_small_nofinal_pixelqkv(sp_heads=(2, 2), **kwargs)


@register_model
def hr_bottleneck_small_nofinal_pixelqkv_head4(**kwargs):
    return hr_bottleneck_small_nofinal_pixelqkv(sp_heads=(4, 4), **kwargs)


@register_model
def hr_bottleneck_small_nofinal_pixelqkv_head4_middle(**kwargs):
    return hr_bottleneck_small_nofinal_pixelqkv_head4(
        use_middle_pixel_features=True, **kwargs
    )


@register_model
def hr_bottleneck_cc4_small_nofinal_pixelqkv_head4(**kwargs):
    return hr_bottleneck_small_nofinal_pixelqkv_head4(
        stem_kernel_sizes=(3, 3),
        stem_conv_types=("conv", "conv"),
        stem_channels_list=(32, 64),
        stem_strides=(2, 2),
        **kwargs,
    )


@register_model
def hr_bottleneck_p8_small_nofinal_pixelqkv_head4_448(**kwargs):
    assert kwargs.get("img_size") == 448, kwargs["img_size"]
    raise NotImplementedError()


@register_model
def hr_bottleneck_small_nofinal_pixelqkv_head8(**kwargs):
    return hr_bottleneck_small_nofinal_pixelqkv(sp_heads=(8, 8), **kwargs)


@register_model
def hr_bottleneck_small_nofinal_patch(**kwargs):
    return hr_bottleneck_small_nofinal(
        **_update_params(
            dict(
                sp_method="patch",
                depths=(12,),
                dims=(384,),
                heads=(6,),
                strides=(1,),
                sp_sizes=(4,),
                sp_heads=(1,),
                sp_features_init_methods=("avgpool",),
            ),
            **kwargs,
        )
    )


@register_model
def hr_bottleneck_p8_small_nofinal_patch(**kwargs):
    return hr_bottleneck_small_nofinal_patch(
        stem_kernel_sizes=(8,), stem_strides=(8,), **kwargs
    )


@register_model
def hr_bottleneck_p8_small_nofinal_conv4(**kwargs):
    return hr_bottleneck_small_nofinal_patch(
        stem_kernel_sizes=(8, 4),
        stem_strides=(8, 4),
        stem_channels_list=(64, 384),
        stem_conv_types=("conv", "patch"),
        sp_sizes=(1,),
        **kwargs,
    )


@register_model
def hr_bottleneck_small_nofinal_conv4(**kwargs):
    return hr_bottleneck_small_nofinal_patch(
        stem_kernel_sizes=(4, 4),
        stem_strides=(4, 4),
        stem_channels_list=(64, 384),
        stem_conv_types=("conv", "patch"),
        sp_sizes=(1,),
        **kwargs,
    )


@register_model
def hr_bottleneck_small_nofinal_patch77(**kwargs):
    return hr_bottleneck_small_nofinal(sp_method="patch", **kwargs)


@register_model
def hr_bottleneck_p8_small_nofinal_patch77(**kwargs):
    return hr_bottleneck_small_nofinal_patch77(
        stem_kernel_sizes=(8,), stem_strides=(8,), **kwargs
    )


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table

    import models

    img_size = 224
    # img_size = 448
    model_fns = [
        models.deit_small_patch16_224_ls_1e_5,
        # hr_bottleneck_small_nofinal,
        # hr_bottleneck_p8_small_nofinal_conv4,
        # hr_bottleneck_small_nofinal_patch,
        # asym_bottleneck_small_nofinal_head4,
        # hr_bottleneck_small_nofinal_pixelqkv_head4,
        # asym_bottleneck_conv4_small_nofinal_head4_spiconv,
        # asym_bottleneck_conv4_small_nofinal_head4_spiavgl,
        # asym_bottleneck_conv4_small_nofinal_head4_spilavg,
        # models.deit_small_patch16_224,
        asym_bottleneck_conv4_small_nofinal_head2_spilavg,
        # asym_bottleneck_cc4_small_nofinal_head2_spilavg,
        # asym_bottleneck_conv8_small_nofinal_head2_spilavg,
        # asym_bottleneck_ccc8_small_nofinal_head2_spilavg,
        # models.deit_base_patch16_224,
        # models.deit_tiny_patch16_224,
        # asym_bottleneck_conv4_base_nofinal_head3_spilavg,
        # asym_bottleneck_cc4_base_nofinal_head3_spilavg,
        # asym_bottleneck_conv4_small_nofinal_patch77,
        # asym_bottleneck_conv4_small_nofinal_patch7,
        # asym_bottleneck_ccc8_small_nofinal_head2_spilavg77,
        # asym_bottleneck_conv4_small_nofinal_patch77_spilavg,
        # asym_bottleneck_conv4_small_nofinal_patch7_spilavg,
        # asym_bottleneck_conv4_small_nofinal_patch7_spilavg,
        # asym_bottleneck_conv8_small_nofinal_head2_spilavg_patchbynograd,
        # asym_bottleneck_ccc8_small_nofinal_head2_spilavg77,
        # models.deit_tiny_patch16_224,
        # asym_bottleneck_conv4_tiny_nofinal_head2_spilavg,

        # # models.deit_small_patch16_224,
        # asym_bottleneck_conv8_small_nofinal_head1_spilavg,
        # asym_bottleneck_conv8_small_nofinal_head2_spilavg_iter1,
        # asym_bottleneck_conv8_small_nofinal_head2_spilavg_stage1,
        # asym_bottleneck_conv8_small_nofinal_patch7_spilavg,
    ]

    for model_fn in model_fns:
        model = model_fn(img_size=img_size, drop_path_rate=0.1)
        # https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.nn.FlopCountAnalysis
        flops = FlopCountAnalysis(model, torch.rand(1, 3, img_size, img_size))
        print(flop_count_table(flops, max_depth=3))
        # print(flop_count_table(flops, max_depth=2))

        total = sum(
            [param.nelement() for param in model.parameters() if param.requires_grad]
        )
        print("Number of parameter: %.4fM" % (total / 1e6))

    __import__("ipdb").set_trace()

    from torch.profiler import profile, record_function, ProfilerActivity

    # model = asym_bottleneck_conv4_small_nofinal_head4_spilavg().cuda()
    model = models.deit_small_patch16_224_ls_1e_5().cuda()
    inputs = torch.randn(32, 3, 224, 224).cuda()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        with record_function("model_inference"):
            model(inputs)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
