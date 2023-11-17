import os

def save_image_names_to_file(folder_path, output_file):
    # 确保文件夹路径存在
    if not os.path.exists(folder_path):
        print("指定的文件夹不存在")
        return

    # 打开文件以写入图片名称
    with open(output_file, 'w') as file:
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            # 检查文件是否为图片
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # 写入文件名（不包括扩展名）到文件
                file.write(os.path.splitext(filename)[0] + '\n')

# 使用示例
folder_path = '/data2/yunfei/SpformerV1/data/VOCdevkit/VOC2010/part_annotations/val'  # 替换为你的图片文件夹路径
output_file = '/data2/yunfei/SpformerV1/data/VOCdevkit/VOC2010/part_annotations/val.txt'            # 输出文件的名称
save_image_names_to_file(folder_path, output_file)
