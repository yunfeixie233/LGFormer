import os
import shutil
import json
import numpy as np
import scipy.io as scio

from PIL import Image

train_split = "/data2/yunfei/SpformerV1/data/VOCdevkit/VOC2010/ImageSets/Main/train.txt"
with open(train_split, 'r') as f:
    train_list = f.readlines()
    train_list = [_.strip('\n') for _ in train_list]

val_split = "/data2/yunfei/SpformerV1/data/VOCdevkit/VOC2010/ImageSets/Main/val.txt"
with open(val_split, 'r') as f:
    val_list = f.readlines()
    val_list = [_.strip('\n') for _ in val_list]

image_path = "/data2/yunfei/SpformerV1/data/VOCdevkit/VOC2010/JPEGImages"
imgs = os.listdir(image_path)
for img in imgs:
    if img[:-4] in train_list:
        shutil.move(os.path.join(image_path, img), os.path.join(image_path, "train", img))
    elif img[:-4] in val_list:
        shutil.move(os.path.join(image_path, img), os.path.join(image_path, "val", img))


# modes = ["train", "val"]
# image_path = "/data2/yunfei/SpformerV1/data/VOCdevkit/VOC2010/JPEGImages"
# anno_path = "/data2/yunfei/SpformerV1/data/VOCdevkit/VOC2010/Annotations"

# train_split = "/data2/yunfei/SpformerV1/data/VOCdevkit/VOC2010/ImageSets/Main/train.txt"
# with open(train_split, 'r') as f:
#     train_list = f.readlines()
#     train_list = [_.strip('\n') for _ in train_list]

# val_split = "/data2/yunfei/SpformerV1/data/VOCdevkit/VOC2010/ImageSets/Main/val.txt"
# with open(val_split, 'r') as f:
#     val_list = f.readlines()
#     val_list = [_.strip('\n') for _ in val_list]

# for mode in modes:
#     img_dir = os.path.join(image_path, mode)
#     if not os.path.exists(img_dir):
#         os.makedirs(img_dir)    
#     imgs = os.listdir(img_dir)
#     for img in imgs:
#         if not os.path.exists(os.path.join(anno_path, mode, img[:-4]+".png")):
#             print("remove" + os.path.join(img_dir, img))
#             os.remove(os.path.join(img_dir, img))


modes = ["train", "val"]
# image_path = "/mnt/sdg/ju/VOC2010/JPEGImages"
anno_path = "/data2/yunfei/SpformerV1/data/VOCdevkit/VOC2010/Annotations"
for mode in modes:
    anno_dir = os.path.join(anno_path, mode)
    anno_whole_dir = os.path.join(anno_path, mode+"_whole")
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)       
    if not os.path.exists(anno_whole_dir):
        os.makedirs(anno_whole_dir)     
    imgs = os.listdir(anno_dir)
    for img in imgs:
        img_dict = {}
        current_img = Image.open(os.path.join(anno_dir, img))
        current_img_shape = list(current_img.size)
        img_dict["imgHeight"] = current_img_shape[1]
        img_dict["imgWidth"] = current_img_shape[0]
        with open(os.path.join(anno_dir, img[:-4]+".json"), "w") as f:
            json.dump(img_dict, f)
        with open(os.path.join(anno_whole_dir, img[:-4]+".json"), "w") as f:
            json.dump(img_dict, f)



object_list = ["aeroplane", "bicycle", "bird", "bottle", "bus", "car", "cat", "cow", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "train", "tvmonitor"]

aeroplane_parts = ["body", "engine", "wing", "stern", "tail", "wheel"]
bicycle_parts = ["wheel", "head", "saddle"]
bird_parts = ["head", "foot", "wing", "tail", "torso"]
bottle_parts = ["body", "cap"]
bus_parts = ["body", "wheel", "mirror"]
car_parts = ["body", "wheel", "mirror"]
cat_parts = ["head", "foot", "tail", "torso"]
cow_parts = ["head", "foot", "tail", "torso"]
dog_parts = ["head", "foot", "tail", "torso"]
horse_parts = ["head", "foot", "tail", "torso"]
motorbike_parts = ["wheel", "head", "saddle"]
person_parts = ["head", "foot", "hand", "torso"]
pottedplant_parts = ["plant", "pot"]
sheep_parts = ["head", "foot", "tail", "torso"]
train_parts = ["coach", "head"]
tvmonitor_parts = ["screen"]

part_label_constructor = {0: aeroplane_parts, 1: bicycle_parts, 2: bird_parts, 3: bottle_parts, 4: bus_parts, 5: car_parts, 6: cat_parts, 7: cow_parts, 8: dog_parts,
                          9: horse_parts, 10: motorbike_parts, 11: person_parts, 12: pottedplant_parts, 13: sheep_parts, 14: train_parts, 15: tvmonitor_parts}
part_num = 0
part_list = {}
for i in range(len(object_list)):
    current_parts = part_label_constructor[i]
    for part in current_parts:
        part_list[object_list[i] + "_" + part] = part_num
        part_num += 1
print(part_list)
print(part_num)
print(len(part_list)) 

aeroplane_mapping = {"body": ["body"], "engine": ["engine"], "wing": ["lwing", "rwing"], "stern": ["stern"], "tail": ["tail"], "wheel": ["wheel"]}
bicycle_mapping = {"wheel": ["fwheel", "bwheel", "chainwheel"], "head": ["handlebar", "headlight"], "saddle": ["saddle"]}
bird_mapping = {"head": ["head", "leye", "reye", "beak"], "foot": ["lleg", "lfoot", "rleg", "rfoot"], "wing": ["lwing", "rwing"], "tail": ["tail"], "torso": ["torso", "neck"]}
bottle_mapping = {"body": ["body"], "cap": ["cap"]}
bus_mapping = {"body": ["frontside", "leftside", "rightside", "bikeside", "roofside", "fliplate", "bliplate", "door", "headlight", "window"], "wheel": ["wheel"], "mirror": ["leftmirror", "rightmirror"]}
car_mapping = {"body": ["frontside", "leftside", "rightside", "bikeside", "roofside", "fliplate", "bliplate", "door", "headlight", "window"], "wheel": ["wheel"], "mirror": ["leftmirror", "rightmirror"]}
cat_mapping = {"head": ["head", "leye", "reye", "lear", "rear", "nose"], "foot": ["lfleg", "lfpa", "rfleg", "rfpa", "lbleg", "lbpa", "rbleg", "rbpa"], "tail": ["tail"], "torso": ["torso", "neck"]}
cow_mapping = {"head": ["head", "leye", "reye", "lear", "rear", "muzzle", "lhorn", "rhorn"], "foot": ["lfuleg", "lflleg", "rfuleg", "rflleg", "lbuleg", "lblleg", "rbuleg", "rblleg"], "tail": ["tail"], "torso": ["torso", "neck"]}
dog_mapping = {"head": ["head", "leye", "reye", "lear", "rear", "nose", "muzzle"], "foot": ["lfleg", "lfpa", "rfleg", "rfpa", "lbleg", "lbpa", "rbleg", "rbpa"], "tail": ["tail"], "torso": ["torso", "neck"]}
horse_mapping = {"head": ["head", "leye", "reye", "lear", "rear", "muzzle"], "foot": ["lfuleg", "lflleg", "lfho", "rfuleg", "rflleg", "rfho", "lbuleg", "lblleg", "lbho", "rbuleg", "rblleg", "rbho"], "tail": ["tail"], "torso": ["torso", "neck"]}
motorbike_mapping = {"wheel": ["fwheel", "bwheel"], "head": ["handlebar", "headlight"], "saddle": ["saddle"]}
person_mapping = {"head": ["head", "leye", "reye", "lear", "rear", "lebrow", "rebrow", "nose", "mouth", "hair"], "foot": ["llleg", "luleg", "lfoot", "rlleg", "ruleg", "rfoot"], "hand": ["llarm", "luarm", "lhand", "rlarm", "ruarm", "rhand"], "torso": ["torso"]}
pottedplant_mapping = {"plant": ["plant"], "pot": ["pot"]}
sheep_mapping = {"head": ["head", "leye", "reye", "lear", "rear", "muzzle", "lhorn", "rhorn"], "foot": ["lfuleg", "lflleg", "rfuleg", "rflleg", "lbuleg", "lblleg", "rbuleg", "rblleg"], "tail": ["tail"], "torso": ["torso", "neck"]}
train_mapping = {"coach": ["coach", "cfrontside", "cleftside", "crightside", "cbackside", "croofside"], "head": ["head", "hfrontside", "hleftside", "hrightside", "hbackside", "hroofside", "headlight"]}
tvmonitor_mapping = {"screen": "screen"}

part_mapper_constructor = {0: aeroplane_mapping, 1: bicycle_mapping, 2: bird_mapping, 3: bottle_mapping, 4: bus_mapping, 5: car_mapping, 6: cat_mapping, 7: cow_mapping, 8: dog_mapping,
                          9: horse_mapping, 10: motorbike_mapping, 11: person_mapping, 12: pottedplant_mapping, 13: sheep_mapping, 14: train_mapping, 15: tvmonitor_mapping}

base_anno_path = "/data2/yunfei/SpformerV1/Annotations_Part"
base_image_path = "/data2/yunfei/SpformerV1/data/VOCdevkit/VOC2010/JPEGImages"
new_image_path = "/data2/yunfei/SpformerV1/data/VOCdevkit/VOC2010/part_annotations"

train_split = "/data2/yunfei/SpformerV1/data/VOCdevkit/VOC2010/ImageSets/Main/train.txt"
with open(train_split, 'r') as f:
    train_list = f.readlines()
    train_list = [_.strip('\n') for _ in train_list]

val_split = "/data2/yunfei/SpformerV1/data/VOCdevkit/VOC2010/ImageSets/Main/val.txt"
with open(val_split, 'r') as f:
    val_list = f.readlines()
    val_list = [_.strip('\n') for _ in val_list]

train_img_num = 0
val_img_num = 0
train_instance_num = 0
val_instance_num = 0
train_part_num = 0
val_part_num = 0

annotations = os.listdir(base_anno_path)

for i, annotation in enumerate(annotations):
    print(i)
    annotation_file = os.path.join(base_anno_path, annotation)
    data = scio.loadmat(annotation_file)
    img_prefix = data["anno"][0][0][0][0]
    if img_prefix in train_list:
        train_flag = True
        if not os.path.exists(os.path.join(base_image_path, "train", img_prefix + ".jpg")):
            continue
        current_img = Image.open(os.path.join(base_image_path, "train", img_prefix + ".jpg"))
    else:
        train_flag = False
        if not os.path.exists(os.path.join(base_image_path, "val", img_prefix + ".jpg")):
            continue
        current_img = Image.open(os.path.join(base_image_path, "val", img_prefix + ".jpg"))
    # current_img = Image.open(os.path.join(base_image_path, img_prefix + ".jpg"))
    current_img_shape = list(current_img.size)
    current_img_shape[0], current_img_shape[1] = current_img_shape[1], current_img_shape[0]
    annos = data["anno"][0][0][1][0]
    obj_img = np.ones(current_img_shape).astype(np.uint8) * len(object_list)
    part_img = np.ones(current_img_shape).astype(np.uint8) * len(part_list)
    for instance in annos:
        instance_object_name = instance[0][0]
        if instance_object_name in ["boat", "chair", "table", "sofa"]:
            continue
        if train_flag:
            train_instance_num += 1
        else:
            val_instance_num += 1
        instance_object_id = object_list.index(instance_object_name)
        instance_object_mask = instance[2]
        
        obj_img[instance_object_mask == 1] = instance_object_id

        instance_parts = instance[3]

        for parts in instance_parts:
            current_parts = []
            unduplicateable_part = True
            for part in parts:
                part_name = part[0][0]
                if part_name.find("_") != -1:
                    unduplicateable_part = False
                    if train_flag:
                        train_part_num += 1
                    else:
                        val_part_num += 1
                part_name_prefix = part_name.split("_")[0]
                current_mapping = part_mapper_constructor[instance_object_id]
                for new_part_name, old_part_name in current_mapping.items():
                    if part_name_prefix in old_part_name:
                        updated_part_name = instance_object_name + "_" +  new_part_name
                if updated_part_name not in current_parts and unduplicateable_part:
                    current_parts.append(updated_part_name)
                    if train_flag:
                        train_part_num += 1
                    else:
                        val_part_num += 1
                part_id = part_list[updated_part_name]
                part_mask = part[1]
                part_img[part_mask == 1] = part_id

    if (obj_img == len(object_list)).sum() == current_img_shape[0] * current_img_shape[1]:
        print("Image {} only contains invalid objects (with no parts).".format(img_prefix))
        continue

    obj_img = Image.fromarray(obj_img, mode="L")
    part_img = Image.fromarray(part_img, mode="L")

    if img_prefix in train_list:
        if not os.path.exists(os.path.join(new_image_path, "train_whole",)):
            os.makedirs(os.path.join(new_image_path, "train_whole",))       
        if not os.path.exists(os.path.join(new_image_path, "train")):
            os.makedirs(os.path.join(new_image_path, "train"))                  
        obj_img.save(os.path.join(new_image_path, "train_whole", img_prefix + ".png"))
        part_img.save(os.path.join(new_image_path, "train", img_prefix + ".png"))
        train_img_num += 1
    elif img_prefix in val_list:
        if not os.path.exists(os.path.join(new_image_path, "val_whole")):
            os.makedirs(os.path.join(new_image_path, "val_whole"))       
        if not os.path.exists(os.path.join(new_image_path, "val")):
            os.makedirs(os.path.join(new_image_path, "val"))          
        obj_img.save(os.path.join(new_image_path, "val_whole", img_prefix + ".png"))
        part_img.save(os.path.join(new_image_path, "val", img_prefix + ".png"))
        val_img_num += 1

print("Finish computing")
print(train_instance_num)
print(val_instance_num)
print(train_part_num)
print(val_part_num)

print("Successfully convert {} training images.".format(train_img_num))
print("Successfully convert {} validation images.".format(val_img_num))