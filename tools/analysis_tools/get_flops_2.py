# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from pathlib import Path
import torch
from mmengine import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.registry import init_default_scope
from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
from fvcore.nn import FlopCountAnalysis, flop_count_table
from mmengine.model import revert_sync_batchnorm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Load a segmentor model from config')
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
        help='override some settings in the used config')
    args = parser.parse_args()
    return args

def load_model_from_cfg(args: argparse.Namespace, logger: MMLogger):
    config_name = Path(args.config)

    if not config_name.exists():
        logger.error(f'Config file {config_name} does not exist')

    cfg: Config = Config.fromfile(config_name)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('scope', 'mmseg'))

    model: BaseSegmentor = MODELS.build(cfg.model)
    return model

def count_parameters(model, input_shape):
    # Calculate FLOPs using fvcore.nn
    dummy_input = torch.rand(1, 3, *input_shape)
    flops = FlopCountAnalysis(model, dummy_input)
    print(flop_count_table(flops, max_depth=3))

    # Count the total parameters
    total_params = sum(
        [param.nelement() for param in model.parameters() if param.requires_grad]
    )
    print(f"Number of parameters: {total_params / 1e6:.4f}M")

def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    
    model = load_model_from_cfg(args, logger)
    model = revert_sync_batchnorm(model)
    print("\nModel Loaded Successfully\n")

    # Count parameters and FLOPs
    count_parameters(model, args.shape)

if __name__ == '__main__':
    main()
