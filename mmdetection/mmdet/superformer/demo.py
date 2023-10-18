import torch

from hr_superpixel_models import hr_superformer_stemp4_patch7_small_512_ls_1e_5_sp_pe as model
# 2. Initialize a tensor
input_tensor = torch.randn(1, 3, 512, 512)

# 3. Forward the tensor through the model
output=model().forward(x=input_tensor,**{'generate_seg': True,
            'return_pixel_logits': True,
            'seg_stride': 1})


print(output.shape)