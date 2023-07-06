import torch
from mmseg.models.backbones import ResNet
from mmseg.models.necks import SpformerNeck
from mmseg.models.decode_heads import SpformerHead

# 初始化Backbone, Neck, 和 Head
backbone = ResNet(depth=50).cuda()
neck =  SpformerNeck(embed_dim=[256, 256, 256, 256],
                    depths=[4, 2, 1, 1],
                    num_heads=[2,2,2, 2],
                    n_iter=[4, 4, 4, 4], 
                    stoken_size=[4, 4, 4, 4],
                    projection=256,                    
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0,
                    drop_path_rate=0.3, 
                    use_checkpoint=False,
                    checkpoint_num = [0, 0, 0, 0],
                    layerscale=[False, False, True, True],
                    init_values=1e-5,
                    seg=False,
                    downsample=False,
                    in_chans=256).cuda()

head = SpformerHead(in_channels=256,
       channels=512,
       num_classes=150,
       stoken_size=[4, 4, 4, 4]).cuda()

B=1


input_shape = (B,3,512,512) 


input_data = torch.randn(input_shape).to('cuda')


x = backbone.forward(input_data)
stoken_features,pixel_features = neck.forward(x)
output = head.forward(stoken_features,pixel_features)


print(output.shape)
