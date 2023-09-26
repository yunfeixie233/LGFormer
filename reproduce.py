# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from pathlib import Path

import torch
from mmengine import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope

from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0 to use this script.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('config1', help='train config file path')
    parser.add_argument('config2', help='train config file path')
    
    # parser.add_argument(
    #     '--shape',
    #     type=int,
    #     nargs='+',
    #     default=[2048, 1024],
    #     help='input image size')
    # parser.add_argument(
    #     '--cfg-options',
    #     nargs='+',
    #     action=DictAction,
    #     help='override some settings in the used config, the key-value pair '
    #     'in xxx=yyy format will be merged into config file. If the value to '
    #     'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
    #     'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
    #     'Note that the quotation marks are necessary and that no white space '
    #     'is allowed.')
    args = parser.parse_args()
    return args



def inference(args: argparse.Namespace, logger: MMLogger) -> dict:
        
    config_name1 = Path(args.config1)

    if not config_name1.exists():
        logger.error(f'Config file {config_name1} does not exist')

    cfg: Config = Config.fromfile(config_name1)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('scope', 'mmseg'))

    # if len(args.shape) == 1:
    #     input_shape = (3, args.shape[0], args.shape[0])
    # elif len(args.shape) == 2:
    #     input_shape = (3, ) + tuple(args.shape)
    # else:
    #     raise ValueError('invalid input shape')
    result = {}

    model1: BaseSegmentor = MODELS.build(cfg.model)
    print(model1)
    if hasattr(model1, 'auxiliary_head'):
        model1.auxiliary_head = None
    if torch.cuda.is_available():
        model1.cuda()
    model1 = revert_sync_batchnorm(model1)
    model1.eval()

    config_name2 = Path(args.config2)

    if not config_name2.exists():
        logger.error(f'Config file {config_name2} does not exist')

    cfg: Config = Config.fromfile(config_name2)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('scope', 'mmseg'))

    # if len(args.shape) == 1:
    #     input_shape = (3, args.shape[0], args.shape[0])
    # elif len(args.shape) == 2:
    #     input_shape = (3, ) + tuple(args.shape)
    # else:
    #     raise ValueError('invalid input shape')
    result = {}

    model2 :BaseSegmentor = MODELS.build(cfg.model)
    print(model2)
    if hasattr(model1, 'auxiliary_head'):
        model2.auxiliary_head = None
    if torch.cuda.is_available():
        model2.cuda()
    model2 = revert_sync_batchnorm(model2)
    model2.eval()
    
    checkpoint1 = torch.load('/root/autodl-tmp/new/SpformerV1/work_dirs/regproxy-s16-sub4+implicit-mid-4+512x512+160k+adamw-poly+ade20k_sgd/iter_1.pth')
    checkpoint2 = torch.load('/root/autodl-tmp/vit_init.pth')
    # print(checkpoint1.keys())
    model1 = partial_load(model1, checkpoint1['state_dict'])
    # print(checkpoint2.keys())
    model2 = partial_load(model2, checkpoint2)    
    data = torch.randn((1,3,512,512)).cuda()
    outputs1=model1.forward(data) 

    outputs2=model2.forward(data) 
       
    return print(outputs1 - outputs2)

import torch

def partial_load(model, checkpoint_path):
    """
    Partially loads model weights from a checkpoint, ignoring keys that don't match.

    :param model: The model into which you want to load the weights.
    :param checkpoint_path: Path to the checkpoint.
    :return: The model with the weights loaded.
    """
    model_dict = model.state_dict()  
    pretrained_dict = checkpoint_path  
    mismatched_keys = []

    matched_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if model_dict[k].size() == v.size():
                if 'patch' in k:
                    print("load")
                matched_dict[k] = v
            else:
                mismatched_keys.append(k)
        else:
            mismatched_keys.append(k)

    if mismatched_keys:
        print("Mismatched keys:", mismatched_keys)

    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)

    return model


def main():

    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')

    result = inference(args, logger)


if __name__ == '__main__':
    main()
