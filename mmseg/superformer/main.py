# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import os
import time
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
# from torch.distributed.elastic.multiprocessing.errors import record
import torchvision
import torchvision.transforms as transforms
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import get_state_dict, ModelEma
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from datasets import build_dataset
from datasets import build_transform
from engine import evaluate_imagenetx, train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator
import time
import torch.distributed as dist

import models
import models_v2
import superpixel_models
import hr_superpixel_models
import hr_bottleneck_models

import utils
from pathlib import Path
import filelock

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')
    parser.add_argument('--precision', default='fp16', type=str)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--unscale_lr', type=bool, default=True,
                        help='')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                            "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Superpixel params
    parser.add_argument('--cls_weight', type=float, default=1.0,
                        help='weight of cls loss')
    parser.add_argument('--superpixel_naive_weight', type=float, default=0.0,
                        help='weight of superpixel naive loss')
    parser.add_argument('--superpixel_naive_topk_pos', type=int, default=2,
                        help='topk positive superpixels')
    parser.add_argument('--superpixel_naive_topk_neg', type=int, default=0,
                        help='topk negative superpixels')
    parser.add_argument('--superpixel_loss_negative_max', type=float,
                        default=6.0, help='topk negative superpixels')
    parser.add_argument('--superpixel_simil_cls_weight', type=float, default=0.0,
                        help='weight of similarities classification loss')
    parser.add_argument('--seg_weight', type=float, default=1.0,
                        help='weight of segmentation loss')
    parser.add_argument('--seg_batchsize', type=int, default=4,
                        help='number of classes in seg')
    parser.add_argument('--seg_config', type=str,
                        help='path to seg dataset config')
    parser.add_argument('--seg_only', type=bool,
                        help='whether only use seg data')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19','ADE20K'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument("--already224", action="store_true")
    parser.add_argument("--rotate_eval", type=float, default=0.0)
    parser.add_argument("--occluded_eval", type=float, default=0.0)
    parser.add_argument("--imagenetx_eval", action="store_true")
    parser.add_argument("--imagenetv2_eval", action="store_true")

    parser.add_argument('--debug', action='store_true', help='debug mode')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--load', default='', help='load from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval_interval',type=int,default=1, help='')
    parser.add_argument('--vis',type=bool,default=False, help='visualized the result')

    parser.add_argument('--demo', type=str, help='path to the demo image')
    parser.add_argument('--eval_vis_dir', type=str, default=None, help='Visualize and break')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', default=False, type=bool,
                        help='')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


# @record
def main(args):
    utils.init_distributed_mode(args)

    if args.debug:
        print('debugging')
        torch.autograd.set_detect_anomaly(True)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    if not args.eval:
        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if not args.imagenetv2_eval:
        dataset_val, nb_classes = build_dataset(is_train=False, args=args)
    else:
        from imagenetv2_pytorch import ImageNetV2Dataset

        dataset_val = ImageNetV2Dataset("matched-frequency", transform=build_transform(False, args)) # supports matched-frequency, threshold-0.7, top-images variants
        nb_classes = 1000

    # if not hasattr(args, 'nb_classes'):
    #     args.nb_classes = nb_classes
    #     del nb_classes

    if args.already224:
        print('already224')
        __import__('ipdb').set_trace()
        
        del dataset_val.transform
        dataset_val.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        print(dataset_val.transform)

        # # NOTE(meijieru): tmp
        # del dataset_val.transform
        # dataset_val.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        # ])
        # print(dataset_val.transform)

    if args.rotate_eval:
        print(f'rotating eval images by {args.rotate_eval} degrees')
        del dataset_val.transform
        dataset_val.transform = transforms.Compose([
            transforms.RandomRotation((args.rotate_eval, args.rotate_eval)),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        print(dataset_val.transform)


    if args.seg_only is True:
        seg_cfg = None
        seg_num_classes = None
        data_loader_train, data_loader_val = None, None
        from mmseg.datasets import build_dataset as build_dataset_seg
        from mmseg.datasets import build_dataloader as build_dataloader_seg
        from mmcv.utils import Config
        from mmcv.utils.registry import build_from_cfg

        print(f'Loading segmentation config from {args.seg_config}')
        seg_cfg = Config.fromfile(args.seg_config)
        seg_num_classes = 150
        print(f'Loaded segmentation config with num_classes={seg_num_classes}')
        seg_cfg.data.train.type=seg_cfg.dataset_type
        seg_cfg.data.test.type=seg_cfg.dataset_type
    
        dataset_train = build_dataset_seg(seg_cfg.data.train)
        
        data_loader_train = build_dataloader_seg(
                dataset_train, args.batch_size, args.num_workers)



    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if not args.eval:
            if args.repeated_aug:
                sampler_train = RASampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
            else:
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # if True:    
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
    #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    if args.seg_only is False:
        if not args.eval:
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
            )
            if args.ThreeAugment:
                data_loader_train.dataset.transform = new_data_aug_generator(args)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,
    )
    # model = torch.compile(model, mode="reduce-overhead")

                    
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
        
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
        print("Use linear_scaled_lr")
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = utils.NativeScaler()

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    if args.cooldown_epochs == 0:
        assert num_epochs == args.epochs

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )



    now = datetime.datetime.now() 
    date_time = now.strftime("%m%d_%H%M%S")
    args.output_dir=args.output_dir + '_' + date_time
    output_dir = Path(args.output_dir)

    if True:
        log_dir = os.path.join(args.output_dir, 'logs')
        os.makedirs(log_dir,exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_dir = os.path.join(args.output_dir, 'logs')
        log_writer = None
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)
    if args.load:
            try:
                checkpoint = torch.load(args.load, map_location='cpu')
                model_without_ddp.load_state_dict(checkpoint['model'])
            except RuntimeError as e:
                print("Warning: There's a mismatch between the model and the checkpoint.")
                print("Trying a partial load...")

                # 获取预训练权重和当前模型的权重
                pretrained_dict = checkpoint['model']
                model_dict = model_without_ddp.state_dict()

                # 找出预训练权重中不存在于模型中的权重
                unmatched_in_pretrained = [k for k in pretrained_dict.keys() if k not in model_dict]
                print(f"Unmatched keys in pretrained dict: {unmatched_in_pretrained}")

                # 找出模型权重中不存在于预训练权重中的权重
                unmatched_in_model = [k for k in model_dict.keys() if k not in pretrained_dict]
                print(f"Unmatched keys in model dict: {unmatched_in_model}")

                # 通过匹配 'pretrained_dict' 和 'model_dict' 的键名来构建新的权重字典
                matched_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

                # 更新当前模型的权重
                model_dict.update(matched_dict)

                # 再次尝试载入权重
                try:
                    model_without_ddp.load_state_dict(model_dict)
                    print("Partial load was successful.")
                except RuntimeError as e:
                    print("Error: Loading model failed even after partial load. Please check the model architecture and the checkpoint.")
                    raise e

    if args.eval:
        visualize_dir = os.path.join(args.output_dir, args.eval_vis_dir) if args.eval_vis_dir else None
        if args.imagenetx_eval:
            visualize_dir = os.path.join(args.output_dir, 'imagenetx')
            test_stats = evaluate_imagenetx(data_loader_val, model, device, visualize_dir=visualize_dir, args=args)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            return
        else:
            test_stats = evaluate(data_loader_val, model, device, visualize_dir=visualize_dir, args=args)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            return

    print(f"Start training for {num_epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_mIoU= 0.0
    max_aAcc= 0.0
    max_mAcc= 0.0
    for epoch in range(args.start_epoch, num_epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model, criterion=criterion, data_loader=data_loader_train,pixel_data_loader=None,seg_cfg=seg_cfg,
            optimizer=optimizer, device=device, epoch=epoch, loss_scaler=loss_scaler,
            max_norm=args.clip_grad, model_ema=model_ema, mixup_fn=mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            log_writer=log_writer,
            args = args,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
             
        if epoch % args.eval_interval==0:

            if utils.is_main_process():
                    dataset_val = build_dataset_seg(seg_cfg.data.val)
                    data_loader_val = build_dataloader_seg(
                    dataset=dataset_val, samples_per_gpu=1, workers_per_gpu=args.num_workers,num_gpus=1, shuffle=False, drop_last=False,dist=False)
                    device_test="cuda:0"
                    print("test on",device_test)
                    if isinstance(data_loader_val.sampler, torch.utils.data.distributed.DistributedSampler):
                        print("A DistributedSampler is being used.")
                    else:
                        print("A DistributedSampler is not being used.")
                    model_without_ddp.to(device_test)
                    test_stats = evaluate(data_loader=data_loader_val, model=model_without_ddp, device=device_test,args=args,visualize_dir=args.output_dir)
                    
                    log_writer.add_scalar('val/mIoU', test_stats['mIoU'], epoch)
                    log_writer.add_scalar('val/aAcc', test_stats['aAcc'], epoch)
                    log_writer.add_scalar('val/mAcc', test_stats['mAcc'], epoch)
                    
                    if max_mIoU < test_stats["mIoU"]:
                        max_mIoU = test_stats["mIoU"]
                    if max_aAcc < test_stats["aAcc"]:
                        max_aAcc = test_stats["aAcc"]
                    if max_mAcc < test_stats["mAcc"]:
                        max_mAcc = test_stats["mAcc"]
                    print(f'Max mIoU: {max_mIoU:.2f}%, Max aAcc: {max_aAcc:.2f}%, Max mAcc: {max_mAcc:.2f}%')
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                    **{f'test_{k}': v for k, v in test_stats.items()},
                                    'epoch': epoch,
                                    'n_parameters': n_parameters}
                    print("Done main process")
                    dist.barrier()
            else:
                dist.barrier()
            # print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if args.seg_only is False:
                if log_writer is not None:
                    log_writer.add_scalar('val/acc1', test_stats['acc1'], epoch)
                    log_writer.add_scalar('val/acc5', test_stats['acc5'], epoch)
                    if 'acc1_sim' in test_stats:
                        log_writer.add_scalar('val/acc1_sim', test_stats['acc1_sim'], epoch)
                    log_writer.add_scalar('val/loss', test_stats['loss'], epoch)
                
                if max_accuracy < test_stats["acc1"]:
                    max_accuracy = test_stats["acc1"]
                print(f'Max accuracy: {max_accuracy:.2f}%')

            # if args.output_dir:
            #         checkpoint_paths = [output_dir / 'best_checkpoint.pth']
            #         for checkpoint_path in checkpoint_paths:
            #             utils.save_on_master({
            #                 'model': model_without_ddp.state_dict(),
            #                 'optimizer': optimizer.state_dict(),
            #                 'lr_scheduler': lr_scheduler.state_dict(),
            #                 'epoch': epoch,
            #                 'model_ema': get_state_dict(model_ema),
            #                 'scaler': loss_scaler.state_dict(),
            #                 'args': args,
            #             }, checkpoint_path)
             
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
