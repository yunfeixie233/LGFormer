import torch
import einops
rearrange = einops.rearrange
import torch.nn.functional as F

def compare_weights(weight_file1, weight_file2):
    # 读取第一个权重文件
    data1 = torch.load(weight_file1)
    model_data1 = data1
    keys1 = set(model_data1.keys())

    # 读取第二个权重文件
    data2 = torch.load(weight_file2)
    model_data2 = data2
    keys2 = set(model_data2.keys())

    # 找出共有的key
    common_keys = keys1.intersection(keys2)
    print("共有的key:")
    for key in common_keys:
        print(key)

    # 找出第一个权重文件独有的key
    unique_keys1 = keys1.difference(keys2)
    print("\n第一个权重文件独有的key:")
    for key in unique_keys1:
        print(key)

    # 找出第二个权重文件独有的key
    unique_keys2 = keys2.difference(keys1)
    print("\n第二个权重文件独有的key:")
    for key in unique_keys2:
        print(key)

# 输入两个权重文件的路径
weight_file1 = '/data2/yunfei/SpformerV1/tiny_p9.pth'
weight_file2 = '/data2/yunfei/SpformerV1/best_p9_stem.pth'

# 比较权重
compare_weights(weight_file1, weight_file2)