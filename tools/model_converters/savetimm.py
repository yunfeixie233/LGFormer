import timm
import torch

# 创建预训练模型
model = timm.create_model('vit_small_patch16_384', pretrained=True)

# 将模型的权重保存到本地文件
torch.save(model.state_dict(), '/data2/yunfei/vit_small_patch16_384.pth')
