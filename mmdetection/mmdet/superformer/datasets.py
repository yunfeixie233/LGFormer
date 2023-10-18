# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import torch
from PIL import Image

class ADE20KDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, train=True, img_suffix='.jpg', seg_map_suffix='.png', transform=None):
        split = 'training' if train else 'validation'

        self.root_dir = root_dir
        self.is_train = train
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.transform = transform

        # Locations of images and segmentation maps
        self.img_folder = os.path.join(root_dir, 'ADEChallengeData2016', 'images', split)
        self.seg_folder = os.path.join(root_dir, 'ADEChallengeData2016', 'annotations', split)

        self.img_list = [f for f in os.listdir(self.img_folder) if f.endswith(self.img_suffix)]
        self.seg_list = [f for f in os.listdir(self.seg_folder) if f.endswith(self.seg_map_suffix)]

        assert len(self.img_list) == len(self.seg_list), 'Number of images and segmentation maps must be equal!'

        # Transformations for image and segmentation map
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((2048, 512)),
                transforms.RandomCrop((512, 512)),
                transforms.RandomHorizontalFlip(0.5),
                # PhotoMetricDistortion is not directly available in torchvision
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((2048, 512)),
                transforms.ToTensor(),
            ])

        self.seg_transform = transforms.Compose([
            transforms.Resize((2048, 512)),  # Added resizing and cropping to match image transformations
            transforms.RandomCrop((512, 512)),  # Added only for training split
            transforms.ToTensor()  # Convert PIL image to PyTorch tensor
        ])
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.img_list[idx])
        seg_path = os.path.join(self.seg_folder, self.seg_list[idx])

        img = Image.open(img_path).convert("RGB")
        seg = Image.open(seg_path)

        # Apply transformations
        img = self.transform(img)
        seg = self.seg_transform(seg)

        return img, seg

    def __len__(self):
        return len(self.img_list)  
      
class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'ADE20K':
        dataset = ADE20KDataset(args.data_path,train=is_train)
        nb_classes = 150
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
