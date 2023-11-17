import os
from PIL import Image
import numpy as np

def image_to_tensor(file_path):
    """ 将图片转换为张量 """
    with Image.open(file_path) as img:
        return np.array(img)

def find_max_unique_value_in_folder(folder_path):
    """ 遍历文件夹中的所有图片，转换为张量，并找出最大的唯一值 """
    max_unique_value = None
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(folder_path, filename)
            tensor = image_to_tensor(file_path)
            unique_values = np.unique(tensor)
            max_value_in_image = unique_values.min()
            if max_unique_value is None or max_value_in_image < max_unique_value:
                max_unique_value = max_value_in_image

    return max_unique_value

# 使用示例
# 使用示例
folder_path = '/data2/yunfei/SpformerV1/data/VOCdevkit/VOC2010/part_annotations/train'  # 替换为你的文件夹路径
max_unique_value = find_max_unique_value_in_folder(folder_path)
print("在所有图片中找到的最大的唯一值是:", max_unique_value)


