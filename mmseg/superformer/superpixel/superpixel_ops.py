"""SuperPixel Transformer ops."""

from fileinput import filename
import functools
from typing import Optional
import warnings
import itertools
import einops
from matplotlib.offsetbox import HPacker
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import mmcv
import os.path as osp
import os
import h5py
rearrange = einops.rearrange

SOFTMAX_IN_FLOAT32 = False
def to_h5(output_directory, max_keys_per_file=None,  max_file_per_fold=None, **kwargs):
    """
    Save input tensors or numpy arrays to an H5 file in a specified directory with automatic numbering.
    Each key from kwargs gets its own sub-directory.
    
    Usage:
    >>> a = torch.tensor([1,2,3])
    >>> b = np.array([4,5,6])
    >>> to_h5(output_directory='./output', a=a, b=b)
    
    Arguments:
    output_directory: The main directory where the sub-directories and H5 files should be saved.
    max_keys_per_file: Maximum number of keys (datasets) allowed in each H5 file.
    **kwargs : Tensors or numpy arrays to save.
    
    Returns:
    None
    """
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Helper function to handle data writing
    def write_data(f, key, value,convert_4D = True):
        if torch.is_tensor(value) and value.device.type == 'cuda':
            value = value.detach().cpu().numpy()
        elif torch.is_tensor(value):
            value = value.detach().numpy()
        if value.ndim == 3 and convert_4D:
            h = w = int(math.sqrt(value.shape[1]))
            value = rearrange(
                value,
                'b (h w) c -> b h w c',
                h = h, w = w
            )
        f.create_dataset(key, data=value)
    
    for key, value in kwargs.items():
        sub_directory = osp.join(output_directory, key)
        
        if not os.path.exists(sub_directory):
            os.makedirs(sub_directory)
        
        existing_files = [f for f in os.listdir(sub_directory) if f.startswith(key) and f.endswith('.h5')]
        counts = [int(f.split(key)[1].split('.h5')[0]) for f in existing_files]
        max_count = max(counts, default=0)  # get the maximum count or set it to 0 if the folder is empty
        if max_file_per_fold and max_file_per_fold < max_count:
            break
        output_file = osp.join(sub_directory, f'{key}{max_count}.h5')
        
        with h5py.File(output_file, "a") as f:
            keys_in_file = list(f.keys())
            
            if max_keys_per_file is not None and len(keys_in_file) >= max_keys_per_file:
                max_count += 1
                output_file = osp.join(sub_directory, f'{key}{max_count}.h5')
                f.close()
                f = h5py.File(output_file, "a")
                
            write_data(f, key, value)
            f.close()
def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret



def maskout_boundary(
    similarities: torch.Tensor, inplace: bool = True, val: float = float("-inf")
) -> torch.Tensor:
    """Masks out the invalid superpixels.

    Args:
        similarities: [b, 9, sh, ph, sw, pw].
        inplace: A bool indicates whether modify the similarities inplace.
        val: Value for masked regions.

    Returns:
        Masked similarities.
    """
    if inplace:
        # [top, bottom, left, right] respectively.
        similarities[:, :3, 0, :, :, :] = val
        similarities[:, -3:, -1, :, :, :] = val
        similarities[:, ::3, :, :, 0, :] = val
        similarities[:, 2::3, :, :, -1, :] = val
    else:
        raise NotImplementedError()
    return similarities

_unfold_weights_sp = torch.eye(3**2)
_unfold_weights_sp = _unfold_weights_sp.reshape(3**2, 1, 3, 3)
_unfold_weights_pixel = torch.eye(12**2)
_unfold_weights_pixel = _unfold_weights_pixel.reshape(12**2, 1, 12, 12)


def unfold_features_sp(features: torch.Tensor) -> torch.Tensor:
    global _unfold_weights_sp
    if _unfold_weights_sp.device != features.device:
        _unfold_weights_sp = _unfold_weights_sp.to(features.device)
    b, c, h, w = features.shape
    features = F.conv2d(
        features.reshape(b * c, 1, h, w),
        _unfold_weights_sp,
        stride=1,
        padding=1,
    )
    
    # import h5py
    # with h5py.File('unfold.h5', 'a') as f:  # 使用 'a' 模式以便在文件存在时进行追加
    #     f.create_dataset("features", data=features.detach().cpu().numpy())
    #     f.create_dataset("unfold", data=rearrange(features_unfold,
    #                                               "(b c) n h w-> b c n h w",
    #                                               b = b,
    #                                               c = c,
    #                                               n = 9).detach().cpu().numpy())
    return features.reshape(b, c, 9, h, w)

def unfold_features_pixel(features: torch.Tensor) -> torch.Tensor:
    global _unfold_weights_pixel
    if _unfold_weights_pixel.device != features.device:
        _unfold_weights_pixel = _unfold_weights_pixel.to(features.device)
    b, c, h, w = features.shape
    features = F.conv2d(
        features.reshape(b * c, 1, h, w),
        _unfold_weights_pixel,
        stride=1,
        padding="same",
    )
    
    # import h5py
    # with h5py.File('unfold.h5', 'a') as f:  # 使用 'a' 模式以便在文件存在时进行追加
    #     f.create_dataset("features", data=features.detach().cpu().numpy())
    #     f.create_dataset("unfold", data=rearrange(features_unfold,
    #                                               "(b c) n h w-> b c n h w",
    #                                               b = b,
    #                                               c = c,
    #                                               n = 9).detach().cpu().numpy())
    return features.reshape(b, c, 144, h, w)
def fold_features(val: torch.Tensor) -> torch.Tensor:
    """Inverse the unfold operation.

    Args:
        val: [b, sh, sw, c, 9].

    Returns:
        A [b, c, sh, sw] tensor.
    """
    _, sh, sw, _, _ = val.shape
    val = rearrange(val, "b sh sw c n -> b (c n) (sh sw)")
    val = F.fold(val, output_size=(sh, sw), kernel_size=3, padding=1)
    return val


def expand_superpixel_features(
    sp_features: torch.Tensor, similarities: torch.Tensor
) -> torch.Tensor:
    """Expands superpixel features to pixel features.

    Args:
        sp_features (torch.Tensor): [b, c, sh, sw].
        similarities (torch.Tensor): [b, 9, h, w] normalized.

    Returns:
        A unfolded features with shape [b, c, h, w].
    """
    b, c, sh, sw = sp_features.shape
    b, _, h, w = similarities.shape
    assert h % sh == 0 and w % sw == 0
    ph, pw = h // sh, w // sw
    # [b, sh, sw, 9, c]
    unfolded_sp_features = unfold_features_sp(sp_features)
    unfolded_sp_features = rearrange(unfolded_sp_features, "b c n sh sw -> b sh sw n c")

    similarities = similarities.view(b, 9, sh, ph, sw, pw)
    similarities = rearrange(similarities, "b n sh ph sw pw -> b sh sw (ph pw) n")

    res = (similarities @ unfolded_sp_features).view(b, sh, sw, ph, pw, c)
    res = rearrange(res, "b sh sw ph pw c -> b c (sh ph) (sw pw)")
    return res
def compute_similarities_dot_product_sp(
    pixel_features: torch.Tensor, sp_features: torch.Tensor
) -> torch.Tensor:
    """SuperPixel Cross Attention.

    Args:
        pixel_features: [b, c, h, w].
        sp_features: [b, c, sh, sw].

    Returns:
        Similarities with shape [b, 9, h, w].
    """
    b, c, h, w = pixel_features.shape
    b_sp, sc, sh, sw = sp_features.shape
    assert (
        c == sc and h % sh == 0 and w % sw == 0 and b_sp in [1, b]
    ), f"{pixel_features.shape}, {sp_features.shape}"
    ph, pw = h // sh, w // sw
    

    # [b, sh * sw, c, 9]
    unfolded_sp_features = (
        unfold_features_sp(sp_features).permute(0, 3, 4, 1, 2).flatten(1, 2)
    )
    # [b, sh * sw, ph * pw, c]
    pixel_features = rearrange(
        pixel_features, "b c (sh ph) (sw pw) -> b (sh sw) (ph pw) c", sh=sh, sw=sw
    )
    # [b, sh * sw, ph * pw, 9]
    similarities = pixel_features @ unfolded_sp_features
    similarities = similarities.view(b, sh, sw, ph, pw, 9).permute(0, 5, 1, 3, 2, 4)
    similarities = maskout_boundary(similarities)
    return similarities.contiguous().view(b, 9, h, w)


def compute_similarities_dot_product_pixel(
    pixel_features: torch.Tensor, sp_features: torch.Tensor
) -> torch.Tensor:
    """SuperPixel Cross Attention.

    Args:
        pixel_features: [b, c, h, w].
        sp_features: [b, c, sh, sw].

    Returns:
        Similarities with shape [b, 9, h, w].
    """
    b, c, h, w = pixel_features.shape
    b_sp, sc, sh, sw = sp_features.shape
    assert (
        c == sc and h % sh == 0 and w % sw == 0 and b_sp in [1, b]
    ), f"{pixel_features.shape}, {sp_features.shape}"
    ph, pw = h // sh, w // sw

    # [b, sh * sw, c, 9]
    unfold_pixel_features = (
        unfold_features_pixel(pixel_features).permute(0, 3, 4, 1, 2).flatten(1, 2)
    )
    # [b, sh * sw, ph * pw, c]
    pixel_features = rearrange(
        pixel_features, "b c (sh ph) (sw pw) -> b (sh sw) (ph pw) c", sh=sh, sw=sw
    )
    # [b, sh * sw, ph * pw, 9]
    similarities = unfold_pixel_features @ sp_features
    similarities = similarities.view(b, sh, sw, ph, pw, 9).permute(0, 5, 1, 3, 2, 4)
    similarities = maskout_boundary(similarities)
    return similarities.contiguous().view(b, 9, h, w)


def similarities_to_another_perspective(
    similarities: torch.Tensor, padding_value: float = float("-inf")
) -> torch.Tensor:
    """Converts pixel perspective similarities between pixel & superpixel perspective.

    Args:
        similarities: [b, 9, sh, ph, sw, pw].

    Returns:
        [b, 9, sh, ph, sw, pw].
    """
    # Pad sh & sw.
    _, _, sh, _, sw, _ = similarities.shape
    # [b, 9, sh + 2, ph, sw + 2, pw]
    padded = F.pad(similarities, [0, 0, 1, 1, 0, 0, 1, 1], value=padding_value)

    res = []
    for i, j in itertools.product([2, 1, 0], repeat=2):
        c = i * 3 + j
        # [b, sh, ph, sw, pw]
        val = padded[:, c, (2 - i) : (2 - i + sh), :, (2 - j) : (2 - j + sw)]
        res.append(val)
    return torch.stack(res, dim=1)


def softmax_along_superpixel(similarities: torch.Tensor,hard_assign: bool) -> torch.Tensor:
    """Softmax within pixels associated to a superpixel.

    Args:
        similarities: [b, 9, sh, ph, sw, pw].

    Returns:
        Normalized similarities [b, 9, ph, pw, sh, sw].
    """
    b, _, sh, ph, sw, pw = similarities.shape
    similarities_sp_persptive = similarities_to_another_perspective(similarities)
    similarities_sp_persptive = rearrange(
        similarities_sp_persptive, "b n sh ph sw pw -> b (n ph pw) sh sw"
    )
    dtype = torch.float32 if SOFTMAX_IN_FLOAT32 else None
    if hard_assign == True :
        similarities_sp_persptive = hard_softmax(similarities_sp_persptive,dim = 1)
    else :
        similarities_sp_persptive = similarities_sp_persptive.softmax(1, dtype=dtype)
    
    return similarities_sp_persptive.view(b, 9, ph, pw, sh, sw)


def update_superpixel_features_v2(
    pixel_features: torch.Tensor, sp_features: torch.Tensor, similarities: torch.Tensor,hard_assign: bool,
) -> torch.Tensor:
    """Updates superpixel features by weighted combination of pixel features.

    Args:
        pixel_features: [b, c, h, w].
        sp_features: [b, c, sh, sw].
        similarities: [b, 9, h, w].

    Returns:
        A [b, c, sh, sw].
    """
    b, c, h, w = pixel_features.shape
    _, _, sh, sw = sp_features.shape
    ph, pw = h // sh, w // sw

    # [b, n, ph, pw, sh, sw]
    similarities = similarities.view(b, 9, sh, ph, sw, pw)
    similarities = softmax_along_superpixel(similarities,hard_assign)
    similarities = rearrange(similarities, "b n ph pw sh sw -> b n sh ph sw pw")
    similarities = similarities_to_another_perspective(similarities, 0)
    similarities = rearrange(similarities, "b n sh ph sw pw -> b (sh sw) (ph pw) n")

    # [b, sh * sw, ph * pw, c]
    pixel_features = rearrange(
        pixel_features, "b c (sh ph) (sw pw) -> b (sh sw) c (ph pw)", sh=sh, sw=sw
    )

    # [b, sh * sw, c, n]
    sp_delta = pixel_features @ similarities
    sp_delta = fold_features(sp_delta.reshape(b, sh, sw, c, 9))
    return sp_delta


update_superpixel_features = update_superpixel_features_v2
# compute_similarities_dot_product = compute_similarities_dot_product_v2
# unfold_features = unfold_features_v2


def update_pixel_features(
    pixel_features: Optional[torch.Tensor],
    sp_features: torch.Tensor,
    similarities: torch.Tensor,
) -> torch.Tensor:
    """Updates pixel features by weighted combination of superpixel features."""
    del pixel_features

    assert similarities.size(1) == 9
    dtype = torch.float32 if SOFTMAX_IN_FLOAT32 else None
    similarities = similarities.softmax(1, dtype=dtype)
    res = expand_superpixel_features(sp_features, similarities)
    return res


def _create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
        A colormap for visualizing segmentation results.
    """

    def bit_get(val, idx):
        return (val >> idx) & 1

    colormap = np.zeros((512, 3), dtype=np.int32)
    ind = np.arange(512, dtype=np.int32)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


@functools.lru_cache(5)
def create_superpixel_colormap(colormap_name: str = "random", seed=1234):
    if colormap_name == "random":
        cm = torch.from_numpy(_create_pascal_label_colormap())
        g = torch.Generator()
        g.manual_seed(seed)
        return cm[torch.randperm(cm.size(0), generator=g)]
    else:
        from matplotlib import cm

        del seed
        cm_func = getattr(cm, colormap_name, None)
        if cm_func is None:
            raise ValueError()
        cm = torch.from_numpy(cm_func(range(512))[:, :3] * 255).to(torch.int32)
        return cm

def show_result(
                img,
                result,
                classes,
                palette=None,
                opacity=0.5):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor): The semantic segmentation results to draw over
            `img`.
        palette (list[list[int]]] | np.ndarray | None): The palette of
            segmentation map. If None is given, random palette will be
            generated. Default: None
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img)
    img = img.copy()
    seg = result
    if palette is None:
        # Get random state before set seed,
        # and restore random state later.
        # It will prevent loss of randomness, as the palette
        # may be different in each iteration if not specified.
        # See: https://github.com/open-mmlab/mmdetection/issues/5844
        state = np.random.get_state()
        np.random.seed(42)
        # random palette
        palette = np.random.randint(
            0, 255, size=(len(classes), 3))
        np.random.set_state(state)
    palette = np.array(palette)
    assert palette.shape[0] == len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    # convert to BGR
    # color_seg = color_seg[..., ::-1]

    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)
        
    
    return img


def compute_grid_segments_id(sh, sw, ph, pw, dtype=torch.int32, device=None):
    grid_segments_ids = torch.arange(0, sh * sw, dtype=dtype, device=device).reshape(
        [sh, 1, sw, 1]
    )
    grid_segments_ids = grid_segments_ids + torch.zeros(
        [1, ph, 1, pw], dtype=dtype, device=device
    )
    return grid_segments_ids.reshape([sh * ph, sw * pw])


def compute_hard_association(
    association: torch.Tensor, validate: bool = True, eps: float = 1e-5
) -> torch.Tensor:
    """Computes hard association from soft association."""
    b, c, sh, ph, sw, pw = association.shape
    if c != 9:
        raise ValueError(f"Unexpected channel: {c}")
    if torch.isnan(association).any().item():
        raise ValueError("NaN detected")

    # If all superpixels have the same weights, choose the middle one.
    association = association.clone()
    association[:, 4] += eps

    offset = torch.tensor([-1, 0, 1], device=association.device)
    offset = (offset[None, :] + (offset * sw)[:, None]).flatten()
    sp_offsets = offset[association.argmax(dim=1)]
    res = compute_grid_segments_id(sh, sw, ph, pw, device=association.device)[
        None, :, :
    ] + sp_offsets.reshape([b, sh * ph, sw * pw])
    if validate:
        if torch.any(res < 0) or torch.any(res >= sh * sw):
            raise RuntimeError()
    return res

# def compute_soft_association(
#     association: torch.Tensor, validate: bool = True, eps: float = 1e-5
# ) -> torch.Tensor:
#     """Computes hard association from soft association."""
#     b, c, sh, ph, sw, pw = association.shape
#     if c != 9:
#         raise ValueError(f"Unexpected channel: {c}")
#     if torch.isnan(association).any().item():
#         raise ValueError("NaN detected")

#     # If all superpixels have the same weights, choose the middle one.
#     association = association.clone()
#     association[:, 4] += eps

#     offset = torch.tensor([-1, 0, 1], device=association.device)
#     offset = (offset[None, :] + (offset * sw)[:, None]).flatten()
#     grid_id = compute_grid_segments_id(sh, sw, ph, pw, device=association.device)[None, :, :].squeeze(0).unsqueeze(-1).expand(-1, -1, 9).to(torch.long)
    
#     # 提前计算加和结果
#     sum_result = (grid_id + offset).to(torch.long)
#     sum_result_ = sum_result.clone()
#     # 创建mask，其中元素值小于0或大于grid_id.max()
#     mask_lower = sum_result < 0
#     mask_upper = sum_result > grid_id.max()

#     # 使用mask来恢复元素到相加之前的值
#     sum_result[mask_lower] = grid_id[mask_lower]
#     sum_result[mask_upper] = grid_id[mask_upper]
#     to_h5(sum_result = sum_result,
#           sp_id = sum_result_,
#           diff = sum_result_ - sum_result,
#           filename='/data2/yunfei/id.h5')
#     sp_id = sum_result

         
#     rows = sp_id // sw
#     cols = sp_id % sh

#     association_glob = torch.zeros(size=(b, sh * ph, sw * pw, sh, sw), dtype=association.dtype, device=association.device)
#     # association[torch.isinf(association)] = 0.0

#     association_reshaped = association.reshape([b, sh * ph, sw * pw, c])
#     rows_expanded = rows[None, ...]  # shape becomes (1, sh * ph, sw * pw, sh)
#     cols_expanded = cols[None, ...]  # shape becomes (1, sh * ph, sw * pw, sw)

#     index_source = torch.arange(b, device=association.device)[:, None, None, None, None]
#     association_glob[index_source, torch.arange(sh * ph)[None, :, None, None, None], torch.arange(sw * pw)[None, None, :, None, None], rows_expanded, cols_expanded] = association_reshaped           


#     return association_glob

# def compute_soft_association(
#     association: torch.Tensor, validate: bool = True, eps: float = 1e-5
# ) -> torch.Tensor:
#     """Computes hard association from soft association."""
#     b, c, sh, ph, sw, pw = association.shape
#     if c != 9:
#         raise ValueError(f"Unexpected channel: {c}")
#     if torch.isnan(association).any().item():
#         raise ValueError("NaN detected")

#     association = association.clone()
#     association[:, 4] += eps

#     offset = torch.tensor([-1, 0, 1], device=association.device)
#     offset = (offset[None, :] + (offset * sw)[:, None]).flatten()
#     grid_id = compute_grid_segments_id(sh, sw, ph, pw, device=association.device)[None, :, :].squeeze(0).unsqueeze(-1).expand(-1, -1, 9).to(torch.long)

#     sum_result = (grid_id + offset).to(torch.long)
#     mask_lower = sum_result < 0
#     mask_upper = sum_result > grid_id.max()
#     sum_result[mask_lower] = grid_id[mask_lower]
#     sum_result[mask_upper] = grid_id[mask_upper]
#     sp_id = sum_result

#     rows = sp_id // sw
#     cols = sp_id % sh

#     association_glob = torch.zeros(size=(b, sh * ph, sw * pw, sh, sw), dtype=association.dtype, device=association.device)

#     # No need to reshape the association tensor
#     for i in range(sh * ph):
#         for j in range(sw * pw):
#             for k in range(c):
#                 association_glob[b -1, i, j, rows[i, j, k], cols[i, j, k]] = association[b -1 , k, i//ph, i%ph, j//pw, j%pw]

#     return association_glob
def compute_soft_association(
    association: torch.Tensor, validate: bool = True, eps: float = 1e-5
) -> torch.Tensor:
    """Computes hard association from soft association."""
    b, c, sh, ph, sw, pw = association.shape
    if c != 9:
        raise ValueError(f"Unexpected channel: {c}")
    if torch.isnan(association).any().item():
        raise ValueError("NaN detected")

    association = association.clone()
    association[:, 4] += eps

    offset = torch.tensor([-1, 0, 1], device=association.device)
    offset = (offset[None, :] + (offset * sw)[:, None]).flatten()
    grid_id = compute_grid_segments_id(sh, sw, ph, pw, device=association.device)[None, :, :].squeeze(0).unsqueeze(-1).expand(-1, -1, 9).to(torch.long)

    sum_result = (grid_id + offset).to(torch.long)
    mask_lower = sum_result < 0
    mask_upper = sum_result > grid_id.max()
    sum_result[mask_lower] = grid_id[mask_lower]
    sum_result[mask_upper] = grid_id[mask_upper]
    sp_id = sum_result

    rows = sp_id // sw
    cols = sp_id % sh

    association_glob = torch.zeros(size=(b, sh * ph, sw * pw, sh, sw), dtype=association.dtype, device=association.device)
    association[torch.isinf(association)] = 0.0
    association = rearrange(
        association,
        'b c sh ph sw pw -> b (sh ph) (sw pw) c'
    )

    # to_h5(association = association,filename='/data2/yunfei/ass.h5')
    # Create expanded indices
    rows_expanded = rows.unsqueeze(0).expand(b, sh*ph, sw*pw, c).reshape(b, sh*ph, sw*pw, c)
    cols_expanded = cols.unsqueeze(0).expand(b, sh*ph, sw*pw, c).reshape(b, sh*ph, sw*pw, c)
    # Create target indices
    i_indices = torch.arange(sh * ph, device=association.device).unsqueeze(1).expand(sh*ph, sw*pw).unsqueeze(0).expand(b, sh*ph, sw*pw)
    j_indices = torch.arange(sw * pw, device=association.device).unsqueeze(0).expand(sh*ph, sw*pw).unsqueeze(0).expand(b, sh*ph, sw*pw)    # Directly set values using efficient indexing
    for k in range(c):
            association_glob[:, i_indices, j_indices, rows_expanded[..., k], cols_expanded[..., k]] = association[..., k]
    association_glob[...,0] = 0
    association_glob[...,31] = 0

    return association_glob 

def resize_similarities_v1(
    similarities: torch.Tensor,
    scale_factor: float,
    mode: str = "bilinear",
    **kwargs,
) -> torch.Tensor:
    """Resizes similarities.

    TODO(meijieru): bilinear within [3 * ph, 3 * pw] instead of [ph, pw]. Could
    be done by first convert to superpixel perspective and interpolate here.

    NOTE(meijieru): would produce `NaN` if has `inf` in similarities.

    Args:
        similarities: [b, 9, sh, ph, sw, pw].
        scale_factor: Scale factor.
        mode: Algorithm used for upsampling.

    Returns:
        Resized similarities with shape [b, 9, sh, ph * size_factor, sw, pw *
                                         size_factor].
    """
    _, n, sh, _, sw, _ = similarities.shape
    assert n == 9
    # NOTE(meijieru): We must reshape to avoid mixing across sp boundaries, as
    # the same sp channel corresponds to different sp.
    res = rearrange(similarities, "b n sh ph sw pw -> b (n sh sw) ph pw")
    res = F.interpolate(res, scale_factor=scale_factor, mode=mode, **kwargs)
    return rearrange(res, "b (n sh sw) ph pw -> b n sh ph sw pw", n=n, sh=sh, sw=sw)


def resize_similarities_v2(
    similarities: torch.Tensor,
    scale_factor: float,
    mode: str = "bilinear",
    resize_version : str = 'v2',
    vis_upsample: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Resizes similarities, before softmax.

    NOTE(meijieru): would produce `NaN` if has `inf` in similarities.

    Args:
        similarities: [b, 9, sh, ph, sw, pw].
        scale_factor: Scale factor.
        mode: Algorithm used for upsampling.

    Returns:
        Resized similarities with shape [b, 9, sh, ph * size_factor, sw, pw *
                                         size_factor].
    """
    b, n, sh, _, sw, _ = similarities.shape
    assert n == 9
    # NOTE(meijieru): We must reshape to avoid mixing across sp boundaries, as
    # the same sp channel corresponds to different sp.
    similarities_sp_persptive = similarities_to_another_perspective(similarities)
    # similarities_sp_persptive = rearrange(
    #     similarities_sp_persptive, "b n sh ph sw pw -> b (n ph pw) sh sw"
    # )
    
    if resize_version == 'v2':
        if vis_upsample:
            import h5py 
            vis = similarities_sp_persptive.detach()
            vis = rearrange(
        vis,
        "b c sh ph sw pw -> b c (sh ph) (sw pw)"
    )
            with h5py.File("/data2/yunfei/sim_before.h5","a") as f:
                keys = list(f.keys())
                key = "sim_before"
                original_key = key
                count = int(0)
                while key in keys:
                    print(f"Dataset with key {key} already exists. Updating key name.")
                    count  = int(count) + 1
                    key = original_key + str(count)
                if int(count) < 5:
                    f.create_dataset(key,data=vis.detach().cpu().numpy())
            
             
        similarities_sp_persptive = rearrange(
        similarities_sp_persptive,
        "b c sh ph sw pw -> (b c) (sh sw) ph pw"
    )


        
        res = F.interpolate(
        similarities_sp_persptive, scale_factor=scale_factor, mode=mode, **kwargs
    )
        if vis_upsample:
            import h5py 
            vis = res.detach()
            vis = vis[0,...].unsqueeze(0)    
            print(vis.shape)        
            vis = rearrange(
        vis,
        " (b c) (sh sw) ph pw ->  b c (sh ph) (sw pw)",
        b = b, c = 1, sh  = sh, sw = sw
    )
            with h5py.File("/data2/yunfei/sim_after.h5","a") as f:
                keys = list(f.keys())
                key = "sim_after"
                original_key = key
                count = int(0)
                while key in keys:
                    print(f"Dataset with key {key} already exists. Updating key name.")
                    count  = int(count) + 1
                    key = original_key + str(count)
                if int(count) < 5:
                    f.create_dataset(key,data=vis.detach().cpu().numpy())

        
        res = rearrange(
        res,
        "(b c) (sh sw) ph pw -> b c sh ph sw pw",
        b=b,
        c=n,
        sh=sh,
        sw=sw
    )
    elif resize_version == 'v3':
        if vis_upsample:
            import h5py 
            vis = similarities_sp_persptive.detach()
            vis = rearrange(
        vis,
        " b c sh ph sw pw ->  b c (sh ph) (sw pw)",
        b = b, c = n, sh  = sh, sw = sw
    )
            with h5py.File("/data2/yunfei/sim_before_v3.h5","a") as f:
                keys = list(f.keys())
                key = "sim_before"
                original_key = key
                count = int(0)
                while key in keys:
                    print(f"Dataset with key {key} already exists. Updating key name.")
                    count  = int(count) + 1
                    key = original_key + str(count)
                if int(count) < 5:
                    f.create_dataset(key,data=vis.detach().cpu().numpy())
                
        
        similarities_sp_persptive = rearrange(
        similarities_sp_persptive,
        "b c sh ph sw pw -> b c (sh ph) (sw pw)"
    )
        res = F.interpolate(
        similarities_sp_persptive, scale_factor=scale_factor, mode=mode, **kwargs
    )
        
        
        res = rearrange(
        res,
        "b c (sh ph) (sw pw) -> b c sh ph sw pw",
        b=b,
        c=n,
        sh=sh,
        sw=sw
    )
        
        if vis_upsample:
            import h5py 
            vis = res.detach()
            vis = vis[:,0,...].unsqueeze(1)
            vis = rearrange(
        vis,
        " b c sh ph sw pw ->  b c (sh ph) (sw pw)",
        b = b, c = 1, sh  = sh, sw = sw
    )
            with h5py.File("/data2/yunfei/sim_after_v3.h5","a") as f:
                keys = list(f.keys())
                key = "sim_after"
                original_key = key
                count = int(0)
                while key in keys:
                    print(f"Dataset with key {key} already exists. Updating key name.")
                    count  = int(count) + 1
                    key = original_key + str(count)
                if int(count) < 5:
                    f.create_dataset(key,data=vis.detach().cpu().numpy())        
    else:
        raise(NotImplementedError)
    # else:
    #     similarities_sp_persptive = rearrange(
    #         similarities_sp_persptive,
    #         "b (ih iw) sh ph sw pw -> b (sh sw) (ih ph) (iw pw)",
    #         ih=3,
    #         iw=3,
    #     )
    #     res = F.interpolate(
    #         similarities_sp_persptive, scale_factor=scale_factor, mode=mode, **kwargs
    #     )    
    #     res = rearrange(
    #     res,
    #     "b (sh sw) (ih ph) (iw pw) -> b (ih iw) sh ph sw pw",
    #     ih=3,
    #     iw=3,
    #     sh=sh,
    #     sw=sw,
    # )
    

    # mask = (similarities_sp_persptive == float("-inf")).to(dtype=torch.float32)
    # kwargs.pop('align_corners', None)
    # mask = F.interpolate(mask, scale_factor=scale_factor, mode="nearest", **kwargs).to(
    #     dtype=torch.bool
    # )
    # res[mask] = float("-inf")

    if False:
        nan_num = 10
        inf_num = 0
        import matplotlib.pyplot as plt

        xxxxx = similarities_sp_persptive[0, 0].clone()
        xxxxx[xxxxx == float("-inf")] = inf_num

        xxxxxx = F.interpolate(
            xxxxx[None, None], scale_factor=scale_factor, mode="nearest", **kwargs
        )[0, 0]

        yyyyy = res[0, 0]
        yyyyy[yyyyy == float("-inf")] = inf_num
        yyyyy = torch.nan_to_num(yyyyy, nan_num)

        res_ref = resize_similarities_v1(similarities, scale_factor, mode, **kwargs)
        res_ref = similarities_to_another_perspective(res_ref)
        res_ref = rearrange(
            res_ref,
            "b (ih iw) sh ph sw pw -> b (sh sw) (ih ph) (iw pw)",
            ih=3,
            iw=3,
        )
        zzzzz = res_ref[0, 0]
        zzzzz[zzzzz == float("-inf")] = inf_num
        zzzzz = torch.nan_to_num(zzzzz, nan_num)

        # plt.imshow(xxxxx)
        # plt.show()

        # plt.imshow(yyyyy)
        # plt.show()
        # plt.imshow(zzzzz)
        # plt.show()

        # disable axis
        plt.axis("off")
        plt.imshow(torch.concat([xxxxxx, yyyyy, zzzzz], dim=0))
        plt.show()
        __import__("ipdb").set_trace()


    res = similarities_to_another_perspective(res)
    warnings.warn("NaNs in similarities are replaced with -inf")
    res = torch.nan_to_num(res, float("-inf"))
    return res


# resize_similarities = resize_similarities_v1
resize_similarities = resize_similarities_v2
