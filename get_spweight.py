import os
from mmseg.apis import init_model
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import numpy as np
import re
config_path = '/data1/yunfei/SpformerV1/configs/superformer/ade/nogt/sp_extra_small_pre_reweight_spfeat.py'
directory_path = '/data1/yunfei/SpformerV1/work_dirs/sp_extra_small_pre_reweight_spfeat'

sorted_files = sorted([file for file in os.listdir(directory_path) if file.endswith('.pth')])

# Loop through all files with .pth extension in the specified directory
for filename in sorted_files:
    if filename.endswith('.pth'):
        checkpoint_path = os.path.join(directory_path, filename)
        match = re.search(r'iter_(\d+).pth', filename)
        if match:
            iteration_number = int(match.group(1))        
            model = init_model(config_path, checkpoint_path)
            for i, stage in enumerate(model.decode_head.stages):
                for j, block in enumerate(stage.patch_embed.blocks):
                    param = (0.5 + torch.sigmoid(block.reweight_pixel.reweight)).detach().cpu().numpy().astype(np.float32)
                    print(param)
                    writer.add_scalar(f'{i}_stage_{j}_block_reweight_sp', param,iteration_number )
                    # param = (0.5 + torch.sigmoid(block.reweight_pixel.reweight)).detach().cpu().numpy().astype(np.float32)
                    # print(param)
                    # writer.add_scalar(f'{i}_stage_{j}_block_reweight_pixel', param,iteration_number )                    
                                        
