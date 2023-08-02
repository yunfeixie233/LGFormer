"""Test for SuperPixel Transformer OPs."""

from typing import Tuple

import time
import itertools

import einops
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import superpixel.superpixel_ops as ops
import superpixel.superpixel_ops_v1 as ops_v1


def bilinear_upsampling(x, scale_factor):
    # Get input dimensions
    batch_size, channels, height, width = x.size()

    # Calculate output dimensions
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # Create indices for upsampling
    rows = (
        torch.arange(new_height, dtype=torch.float32, device=x.device)
        .unsqueeze(1)
        .repeat(1, new_width)
    )
    cols = (
        torch.arange(new_width, dtype=torch.float32, device=x.device)
        .unsqueeze(0)
        .repeat(new_height, 1)
    )

    # Compute indices in original tensor space
    rows /= scale_factor
    cols /= scale_factor
    rows.floor_()
    cols.floor_()
    rows = rows.to(torch.int64)
    cols = cols.to(torch.int64)

    # Compute weights for interpolation
    row_fracs = rows.to(torch.float32) * scale_factor - rows.to(torch.float32)
    col_fracs = cols.to(torch.float32) * scale_factor - cols.to(torch.float32)
    row_w1 = 1 - row_fracs
    col_w1 = 1 - col_fracs
    row_w2 = row_fracs
    col_w2 = col_fracs
    __import__("ipdb").set_trace()

    # Reshape input tensor to (batch_size, channels, height, width)
    x = x.view(batch_size, channels, height, width)

    # Initialize output tensor
    y = torch.zeros(
        batch_size,
        channels,
        new_height,
        new_width,
        dtype=torch.float32,
        device=x.device,
    )

    # Apply bilinear upsampling
    y[:, :, rows, cols] += (
        x[:, :, rows, cols] * row_w1.unsqueeze(-1) * col_w1.unsqueeze(-2)
    )
    y[:, :, rows, cols + 1] += (
        x[:, :, rows, cols] * row_w1.unsqueeze(-1) * col_w2.unsqueeze(-2)
    )
    y[:, :, rows + 1, cols] += (
        x[:, :, rows, cols] * row_w2.unsqueeze(-1) * col_w1.unsqueeze(-2)
    )
    y[:, :, rows + 1, cols + 1] += (
        x[:, :, rows, cols] * row_w2.unsqueeze(-1) * col_w2.unsqueeze(-2)
    )

    # Reshape output tensor to (batch_size, channels, new_height, new_width)
    y = y.view(batch_size, channels, new_height, new_width)

    return y


def interpolate_bilinear(grid: torch.Tensor, query_points: torch.Tensor, indexing="ij"):
    grid_shape = grid.shape
    query_shape = query_points.shape
    batch_size, height, width, channels = (
        grid_shape[0],
        grid_shape[1],
        grid_shape[2],
        grid_shape[3],
    )
    num_queries = query_shape[1]

    index_order = [0, 1] if indexing == "ij" else [1, 0]
    unstacked_query_points = query_points.unbind(dim=-1)

    alphas = []
    floors = []
    ceils = []
    for i, dim in enumerate(index_order):
        with torch.no_grad():
            queries = unstacked_query_points[dim]

            size_in_indexing_dimension = grid_shape[i + 1]

            max_floor = size_in_indexing_dimension - 2
            min_floor = 0
            floor = torch.min(torch.max(torch.floor(queries), min_floor), max_floor)
            int_floor = floor.to(torch.int32)
            floors.append(int_floor)
            ceil = int_floor + 1
            ceils.append(ceil)

            alpha = queries - floor
            min_alpha = 0.0
            max_alpha = 1.0
            alpha = torch.min(torch.max(alpha, min_alpha), max_alpha)

            alpha = alpha.unsqueeze(-1)
            alphas.append(alpha)

    flattened_grid = grid.reshape(batch_size * height * width, channels)
    batch_offsets = (torch.arange(batch_size) * height * width).view(batch_size, 1)

    def gather(y_coords, x_coords, name):
        with torch.no_grad():
            linear_coordinates = batch_offsets + y_coords * width + x_coords
            gathered_values = flattened_grid.gather(0, linear_coordinates)
            return gathered_values.view(batch_size, num_queries, channels)

    top_left = gather(floors[0], floors[1], "top_left")
    top_right = gather(floors[0], ceils[1], "top_right")
    bottom_left = gather(ceils[0], floors[1], "bottom_left")
    bottom_right = gather(ceils[0], ceils[1], "bottom_right")

    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp


def get_example_data(
    batch_size: int = 2,
    sh: int = 4,
    sw: int = 3,
    ph: int = 2,
    pw: int = 2,
    channels: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pixel_feature_shape = (batch_size, channels, sh * ph, sw * pw)
    pixel_features = (
        torch.arange(np.prod(pixel_feature_shape).item())
        .reshape(pixel_feature_shape)
        .float()
    )
    sp_feature_shape = (batch_size, channels, sh, sw)
    sp_features = (
        torch.arange(np.prod(sp_feature_shape).item())
        .reshape(*sp_feature_shape)
        .float()
    )
    return pixel_features, sp_features


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size // 2)
        return x


def compute_similarities_dot_product_superpixel_perspective(
    pixel_features: torch.Tensor, sp_features: torch.Tensor
) -> torch.Tensor:
    """Naively computes dot product with respect to superpixel.

    9x memory usage than unfolding superpixel instead.
    """
    b, c, h, w = pixel_features.shape
    _, _, sh, sw = sp_features.shape
    ph, pw = [h // sh, w // sw]

    def unfold_pixel(val):
        val = F.unfold(val, (3 * ph, 3 * pw), padding=(ph, pw), stride=(ph, pw)).view(
            b, -1, 3, ph, 3, pw, sh, sw
        )
        val = einops.rearrange(val, "b c ih ph iw pw sh sw -> b c (ih iw) sh ph sw pw")
        return val

    unfolded_pixel_features = unfold_pixel(pixel_features)
    # [b, 9, sh, ph, sw, pw]
    res = (unfolded_pixel_features * sp_features.reshape(b, c, 1, sh, 1, sw, 1)).sum(1)
    mask = (
        unfold_pixel(torch.ones_like(pixel_features[:, :1])).to(torch.bool).squeeze(1)
    )
    res[~mask] = float("-inf")
    return res


class SuperPixelTransformerTest(unittest.TestCase):
    def test_create_superpixel_colormap(self):
        for cm_name in ["random", "viridis"]:
            val = ops.create_superpixel_colormap(cm_name)
            self.assertTupleEqual(val.shape, (512, 3))
            self.assertEqual(val.dtype, torch.int32)

    def test_unfold_features(self):
        _, sp_features = get_example_data()
        unfolded_sp_features = ops.unfold_features(sp_features)

        _, _, sh, sw = sp_features.shape
        padded_sp_features = F.pad(sp_features, [1, 1, 1, 1])
        # Current.
        np.testing.assert_allclose(sp_features, unfolded_sp_features[:, :, 4, :, :])
        # Right.
        np.testing.assert_allclose(
            padded_sp_features[:, :, 1 : 1 + sh, 2:],
            unfolded_sp_features[:, :, 5, :, :],
        )
        # Top left.
        np.testing.assert_allclose(
            padded_sp_features[:, :, :sh, :sw], unfolded_sp_features[:, :, 0, :, :]
        )

    def test_fold_unfold_features(self):
        def fold_unfold(val):
            val = ops.unfold_features(val)
            val = einops.rearrange(val, "b c n sh sw -> b sh sw c n")
            val = ops.fold_features(val)
            return val

        _, sp_features = get_example_data()

        scaler = fold_unfold(torch.ones_like(sp_features))

        # corners.
        for i, j in itertools.product([0, -1], repeat=2):
            np.testing.assert_allclose(scaler[:, :, i, j], 4)

        # edges.
        for i in [0, -1]:
            np.testing.assert_allclose(scaler[:, :, 1:-1, i], 6)
            np.testing.assert_allclose(scaler[:, :, i, 1:-1], 6)

        # middles.
        np.testing.assert_allclose(scaler[:, :, 1:-1, 1:-1], 9)

        res = fold_unfold(sp_features)
        np.testing.assert_allclose(res, sp_features * scaler)

    def test_maskout_boundary(self):
        b, sh, sw = 2, 4, 3
        val = ops.maskout_boundary(torch.ones((b, 9, sh, 1, sw, 1)), val=0)
        expected = ops.unfold_features(torch.ones(b, 1, sh, sw))
        np.testing.assert_allclose(val.squeeze(), expected.squeeze())

    def test_similarities_to_superpixel_perspective(self):
        pixel_features, sp_features = get_example_data()
        sp_features *= 4

        b, _, h, w = pixel_features.shape
        _, _, sh, sw = sp_features.shape
        ph, pw = [h // sh, w // sw]

        for method in [
            ops_v1.compute_similarities_dot_product_v1,
            ops.compute_similarities_dot_product_v2,
        ]:
            similarities = method(pixel_features, sp_features).reshape(
                b, 9, sh, ph, sw, pw
            )
            res = ops.similarities_to_another_perspective(similarities)
            expected = compute_similarities_dot_product_superpixel_perspective(
                pixel_features, sp_features
            )
            np.testing.assert_allclose(res, expected)

    def test_compute_similarities_dot_product_speed(self):
        pixel_features, sp_features = get_example_data(32, 7, 7, 4, 4, 160)

        for name, method in zip(
            ["v1", "v2"],
            [
                ops_v1.compute_similarities_dot_product_v1,
                ops.compute_similarities_dot_product_v2,
            ],
        ):
            start = time.time()
            for _ in range(100):
                _ = method(pixel_features, sp_features)
            print(
                f"compute_similarities_dot_product {name}: {time.time() - start:.2f} s"
            )

    def test_update_superpixel_features(self):
        pixel_features, sp_features = get_example_data(32, 7, 7, 4, 4, 160)
        similarities = ops.compute_similarities_dot_product(pixel_features, sp_features)

        np.testing.assert_allclose(
            ops_v1.update_superpixel_features_v1(
                pixel_features, sp_features, similarities
            ),
            ops.update_superpixel_features_v2(
                pixel_features, sp_features, similarities
            ),
        )

    def test_benchmark(self):
        pixel_features, sp_features = get_example_data(32, 7, 7, 4, 4, 160)
        similarities = ops.compute_similarities_dot_product(pixel_features, sp_features)

        for name, method in zip(
            ["v1", "v2"],
            [
                ops_v1.update_superpixel_features_v1,
                ops.update_superpixel_features_v2,
            ],
        ):
            start = time.time()
            for _ in range(10):
                _ = method(pixel_features, sp_features, similarities)
            print(f"update_superpixel_features {name}: {time.time() - start:.2f} s")

    def test_fold_features(self):
        # [b, sh, sw, c, 9]
        shape = (3, 4, 5, 7, 9)
        b, sh, sw, c, _ = shape
        val = torch.randn(shape)

        lhs = ops.fold_features(val)
        rhs = Fold(3)(val.permute(0, 3, 4, 1, 2).flatten(0, 1)).view(b, c, sh, sw)

        np.testing.assert_allclose(lhs, rhs)

    def test_perspective(self):
        shape = (1, 9, 2, 3, 4, 5)
        val = torch.range(1, np.prod(shape).item()).reshape(shape)
        val = ops.maskout_boundary(val)

        rhs = ops.similarities_to_another_perspective(
            ops.similarities_to_another_perspective(val)
        )
        np.testing.assert_allclose(rhs, val)

        val = torch.load("./data/bug_simil.pth", map_location="cpu")
        rhs = ops.similarities_to_another_perspective(
            ops.similarities_to_another_perspective(val)
        )
        np.testing.assert_allclose(rhs, val)

    # @unittest.skip("TODO")
    def test_perspective_real(self):
        val = torch.load("./data/bug_simil.pth", map_location="cpu")

        ref = ops.maskout_boundary(ops.resize_similarities_v1(val, 8))
        self.assertFalse(torch.isnan(ref).any())

        res = ops.maskout_boundary(ops.resize_similarities_v2(val, 8))
        self.assertFalse(torch.isnan(res).any())

    # def test_bilinear_upsample(self):
    #     val = torch.randn(1, 1, 4, 4)
    #     scale_factor = 4

    #     ref = F.interpolate(val, scale_factor=scale_factor, mode="bilinear")
    #     res = bilinear_upsampling(val, scale_factor=scale_factor)
    #     np.testing.assert_allclose(res, ref)

    # @unittest.skip("TODO")
    # def test_expand_superpixel_features(self):
    #     pass


if __name__ == "__main__":
    unittest.main()
