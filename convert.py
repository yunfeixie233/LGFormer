import torch
import einops
rearrange = einops.rearrange
import torch.nn.functional as F

# 读取.pth文件
data = torch.load('/data3/yunfei/SpformerV1/best_p6.pth')
print(data.keys())
new_data = {}
model_data = data['state_dict']
for key, value in model_data.items():
    # key = key.replace('backbone.layers.0','backbone.stages.0.blocks.0')
    # key = key.replace('backbone.layers.1','backbone.stages.0.blocks.1')
    # key = key.replace('backbone.layers.2','backbone.stages.0.blocks.2')
    # key = key.replace('backbone.layers.3','backbone.stages.0.blocks.3')
    # key = key.replace('backbone.layers.4','backbone.stages.0.blocks.4')
    # key = key.replace('backbone.layers.5','backbone.stages.0.blocks.5')
    # key = key.replace('backbone.layers.6','backbone.stages.0.blocks.6')
    # key = key.replace('backbone.layers.7','backbone.stages.0.blocks.7')
    # key = key.replace('backbone.layers.8','backbone.stages.0.blocks.8')
    # key = key.replace('backbone.layers.9','backbone.stages.0.blocks.9')
    # key = key.replace('backbone.layers.10','backbone.stages.0.blocks.10')
    # key = key.replace('backbone.layers.11','backbone.stages.0.blocks.11')
    # key = key.replace('backbone.','decode_head.')
    # if 'pos_embed' in key:
    #     key = 'decode_head.stages.0.pos_embed'
    # new_data[key] = value
    print(key)

