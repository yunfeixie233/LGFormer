import torch
import triton.language as tl
import triton
import numpy as np
import torch.nn.functional as F
import math
import unittest
@triton.autotune(
    configs=[
        triton.Config({'TILE_SIZE': 4}, num_stages=2, num_warps=32),
        triton.Config({'TILE_SIZE': 16}, num_stages=2, num_warps=32),
        triton.Config({'TILE_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'TILE_SIZE': 512}, num_stages=4, num_warps=4),
        triton.Config({'TILE_SIZE': 1024}, num_stages=4, num_warps=4),
        triton.Config({'TILE_SIZE': 2048}, num_stages=4, num_warps=4),
    ],
    key=['out_H', 'out_W', 'num_output_pixels'],  
)

@triton.jit
def bilinear_upsample_kernel(
    I, O, 
    stride_ii, stride_ij, 
    stride_oi, stride_oj,
    lut_indices, lut_fracs,  # Split LUT into indices and fractions
    num_output_pixels, 
    out_H: tl.constexpr,
    out_W: tl.constexpr,
    in_H: tl.constexpr,
    in_W: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):




    # TODO(yunfei) How to replace for-loop to speed up?
    for warp_id in range(0, TILE_SIZE):
        pid = tl.program_id(0) * TILE_SIZE + warp_id
        if pid < num_output_pixels:
            lut_offset = pid * 4  
            i0 = tl.load(lut_indices + lut_offset + 0).to(tl.int32)
            i1 = tl.load(lut_indices + lut_offset + 1).to(tl.int32)
            j0 = tl.load(lut_indices + lut_offset + 2).to(tl.int32)
            j1 = tl.load(lut_indices + lut_offset + 3).to(tl.int32)
            
            lut_offset_frac = pid * 2  
            i_frac = tl.load(lut_fracs + lut_offset_frac + 0).to(tl.float32)  
            j_frac = tl.load(lut_fracs + lut_offset_frac + 1).to(tl.float32)
            
            
            mask_i0 = (i0 >= 0) & (i0 < in_H)
            mask_i1 = (i1 >= 0) & (i1 < in_H)
            mask_j0 = (j0 >= 0) & (j0 < in_W)
            mask_j1 = (j1 >= 0) & (j1 < in_W)
            
            f00 = tl.where(mask_i0 & mask_j0, tl.load(I + i0 * stride_ii + j0 * stride_ij), 0.0)
            f01 = tl.where(mask_i0 & mask_j1, tl.load(I + i0 * stride_ii + j1 * stride_ij), 0.0)
            f10 = tl.where(mask_i1 & mask_j0, tl.load(I + i1 * stride_ii + j0 * stride_ij), 0.0)
            f11 = tl.where(mask_i1 & mask_j1, tl.load(I + i1 * stride_ii + j1 * stride_ij), 0.0)

            # Compute weights and output value
            o_value = (1 - i_frac) * (1 - j_frac) * f00 + \
                    i_frac * (1 - j_frac) * f10 + \
                    (1 - i_frac) * j_frac * f01 + \
                    i_frac * j_frac * f11

            i_out = pid // out_W  # Calculate i_out and j_out using the explicit output dimensions
            j_out = pid % out_W
            tl.store(O + i_out * stride_oi + j_out * stride_oj, o_value)

def bilinear_upsample_lut(I, scale_factor):
    
    H, W = I.shape

    i_out, j_out = torch.meshgrid(torch.arange(H*scale_factor, device=I.device), torch.arange(W*scale_factor, device=I.device))
    i_in = torch.clamp((i_out + 0.5) / scale_factor - 0.5, 0, H-1)
    j_in = torch.clamp((j_out + 0.5) / scale_factor - 0.5, 0, W-1)

    i0 = torch.floor(i_in).int()
    i1 = torch.clamp(i0 + 1, 0, H-1)
    j0 = torch.floor(j_in).int()
    j1 = torch.clamp(j0 + 1, 0, W-1)

    i_frac = (i_in - i0.float()).reshape(-1)
    j_frac = (j_in - j0.float()).reshape(-1)

    lut_indices = torch.stack([i0, i1, j0, j1], dim=-1).reshape(-1, 4).int()
    lut_fracs = torch.stack([i_frac, j_frac], dim=-1).reshape(-1, 2).float()

    return lut_indices,lut_fracs

def bilinear_upsample(I, scale_factor,lut_indices,lut_fracs):

    H, W = I.shape
    num_output_pixels = H * W * scale_factor * scale_factor

    # Allocate output array
    O = torch.empty((H * scale_factor, W * scale_factor), device=I.device, dtype=I.dtype)

    stride_ii, stride_ij = W, 1
    stride_oi, stride_oj = W*scale_factor, 1

    grid = lambda META: (
        triton.cdiv(num_output_pixels, META['TILE_SIZE']),
    )

    bilinear_upsample_kernel[grid](I, O, stride_ii, stride_ij, stride_oi, stride_oj, lut_indices, lut_fracs, num_output_pixels, H*scale_factor, W*scale_factor, H, W)
    return O




import unittest

import time

class TestUpsample(unittest.TestCase):
    def setUp(self):
        # Define the input tensors and scale factor in the setup
        self.scale_factor = 4
        self.H = 16
        self.W = 16
        self.I = torch.randn((self.H, self.W), dtype=torch.float32).cuda()
        self.I_2 = torch.empty(1, 1, self.H, self.W).cuda()
        self.I_2[:, :] = self.I
        self.num_trials = 100  # Define the number of trials for the speed test
        self.lut_indices,self.lut_fracs = bilinear_upsample_lut(self.I,self.scale_factor )
    def test_bilinear_upsample(self):
        
        O = bilinear_upsample(self.I, self.scale_factor,self.lut_indices,self.lut_fracs)
        O_2 = F.interpolate(self.I_2, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        O_2 = O_2.squeeze()
        self.assertTrue(torch.allclose(O, O_2, atol=1e-6))

    def test_execution_time(self):
        custom_time = 0
        pytorch_time = 0

        for _ in range(self.num_trials):
            start_time = time.time()
            O = bilinear_upsample(self.I, self.scale_factor,self.lut_indices,self.lut_fracs)
            end_time = time.time()
            custom_time += end_time - start_time

            start_time = time.time()
            O_2 = F.interpolate(self.I_2, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
            end_time = time.time()
            pytorch_time += end_time - start_time

        custom_time /= self.num_trials
        pytorch_time /= self.num_trials

        print(f"Average execution time of custom method: {custom_time}s")
        print(f"Average execution time of PyTorch method: {pytorch_time}s")
        print(f"custom method is {custom_time/pytorch_time} times than PyTorch method")


if __name__ == '__main__':
    unittest.main()

    # # Define scale factor
    # scale_factor = 2

    # # Create an input tensor
    # # I = torch.tensor([[1, 4], [6, 8]], dtype=torch.float32).cuda()
    # # I = I.view(2, 2)
    # H=3
    # W=3
    # I = torch.randn((H,W), dtype=torch.float32).cuda()
    # I_2 = torch.empty(1,1,H,W).cuda()
    # I_2[:,:]=I
    # # Call the upsample function
    # O = bilinear_upsample(I, scale_factor)
    # O_2=F.interpolate(I_2,scale_factor=scale_factor,mode="bilinear",align_corners=False)
    # print(O)
    # print(O_2)