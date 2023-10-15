# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
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
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def inference(args: argparse.Namespace, logger: MMLogger) -> dict:
    config_name = Path(args.config)

    if not config_name.exists():
        logger.error(f'Config file {config_name} does not exist')

    cfg: Config = Config.fromfile(config_name)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('scope', 'mmseg'))

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    result = {}

    model: BaseSegmentor = MODELS.build(cfg.model)
    if hasattr(model, 'auxiliary_head'):
        model.auxiliary_head = None
    if torch.cuda.is_available():
        model.cuda()
    model = revert_sync_batchnorm(model)

    data = torch.randn(input_shape).unsqueeze(0).cuda()

    model.eval()
    
    
    register_hooks(model)

    # 前向传递
    start_time = time.time()
    with torch.no_grad():
        _ = model(data)
    end_time = time.time()
    print(end_time - start_time)
    # 根据执行时间排序并打印
    # sorted_time_stats = sorted(time_stats.items(), key=lambda x: x[1], reverse=True)
    
    # # 格式化打印为表格形式
    # print("\nTime statistics for each module:")
    # print(f"{'Module Name':<25} | {'Execution Time (s)':<20}")
    # print('-'*50)
    # for module_name, exec_time in sorted_time_stats:
    #     print(f"{module_name:<25} | {exec_time:.6f}")
    # with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
    #     with torch.no_grad():
    #         _ = model(data)
    # print(prof)



import time

# 用于存储每个子模块的执行时间
time_stats = {}

def hook_fn(module, input, output):
    """Hook function to register"""
    end_time = time.time()
    exec_time = end_time - module.start_time
    # 只存储模块的简要名称
    module_name = module.__class__.__name__
    time_stats[module_name] = exec_time
def register_hooks(module):
    """递归地为每个子模块注册hook"""
    if not isinstance(module, nn.Sequential) and \
       not isinstance(module, nn.ModuleList):
        module.register_forward_pre_hook(record_start_time)
        module.register_forward_hook(hook_fn)

    for child in module.children():
        register_hooks(child)

def record_start_time(module, input):
    module.start_time = time.time()


def main():

    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')

    inference(args, logger)
    split_line = '=' * 30



if __name__ == '__main__':
    main()


