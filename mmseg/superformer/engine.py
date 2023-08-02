# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""

from functools import partial
import math
import sys
import os
from typing import Any, Iterable, Optional
from mmseg.core.evaluation.metrics import pre_eval_to_metrics
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from timm.loss import SoftTargetCrossEntropy

from losses import DistillationLoss, SoftTargetCrossEntropyTokenIntuitive
from superpixel.superpixel_ops import create_superpixel_colormap
from superpixel_models import Superformer
from hr_superpixel_models import SuperformerHR
from hr_bottleneck_models import SuperformerBottleNeck
import utils
from mmseg.core.evaluation import mean_iou
import mmcv
from collections import OrderedDict
from prettytable import PrettyTable

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Optional[Iterable], pixel_data_loader: Optional[Iterable],
                    seg_cfg: Optional[Any],
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, log_writer=None, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    if args.seg_only is False:
        # use_seg = args.seg_weight > 0.0 and epoch < seg_cfg.get('num_epochs', 1e10)
        use_seg=True
        seg_criterion = None
        if use_seg:
            from mmseg.models import build_loss
            seg_criterion = build_loss(dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
            seg_criterion = partial(seg_criterion, ignore_index=255)
            pixel_iter = iter(pixel_data_loader)

        token_criterion = SoftTargetCrossEntropyTokenIntuitive(
                args.superpixel_naive_topk_pos, args.superpixel_naive_topk_neg,
                args.superpixel_loss_negative_max)

        optimizer.zero_grad()
        for step, (samples, targets) in enumerate(
                metric_logger.log_every(data_loader, print_freq, header)):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            if args.bce_loss:
                targets = targets.gt(0.0).type(targets.dtype)

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

            enable_autocast = args.precision != 'fp32'
            precision = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.precision]

            loss_seg = None
            if use_seg:
                try:
                    val = next(pixel_iter)
                except StopIteration:
                    pixel_iter = iter(pixel_data_loader)
                    val = next(pixel_iter)

                seg_samples = val['img'].data[0].to(device, non_blocking=True)
                seg_targets = val['gt_semantic_seg'].data[0].to(device, non_blocking=True).squeeze(1)

                # TODO(yunfei) conflict with torch version? & how to use model.module
                # with torch.cuda.amp.autocast(enabled=enable_autocast, dtype=precision):
                with torch.cuda.amp.autocast(enabled=enable_autocast):
                    if isinstance(model.module, Superformer):
                    # if model.__class__.__name__ == "Superformer":
                        kwargs = {'generate_seg': True,
                                    'generate_simil_pred': False,
                                    'return_pixel_logits': True,
                                    'seg_stride': 1}
                    elif isinstance(model.module, SuperformerHR):
                    # elif model.__class__.__name__ == "SuperformerHR":
                        kwargs = {'generate_seg': True,
                                    'return_pixel_logits': True,
                                    'seg_stride': 1}
                    else:
                        raise ValueError()
                    seg_outputs = model(seg_samples, **kwargs)
                    loss_seg = seg_criterion(seg_outputs['seg'], seg_targets) * args.seg_weight
                    
                    # Additional outputs
                    for key in seg_outputs:
                        if key != "seg":
                            loss_seg += 0. * seg_outputs[key].sum()
                    
                # NOTE: temporary annotating for using loss_scaler for only one time
                # loss_scaler(loss_seg, optimizer, clip_grad=max_norm,
                #             parameters=model.parameters(), create_graph=is_second_order,
                #             # NOTE(meijieru): don't update params here.
                #             update_grad=False)
            
            # # with torch.cuda.amp.autocast(enabled=enable_autocast, dtype=precision):
            # with torch.cuda.amp.autocast(enabled=enable_autocast):
            #     if model.__class__.__name__ == "Superformer":
            #         kwargs = {'generate_seg': args.superpixel_naive_weight > 0.0,
            #                   'generate_simil_pred': args.superpixel_simil_cls_weight > 0.0,
            #                   'return_pixel_logits': False}
            #     if model.__class__.__name__ == "SuperformerHR":
            #         if args.superpixel_naive_weight > 0.0 or args.superpixel_simil_cls_weight > 0.0:
            #             raise ValueError()
            #         kwargs = {}
            #     else:
            #         kwargs = {}
            #     outputs = model(samples, **kwargs)

            #     if  model.__class__.__name__==  "SuperformerHR" or "Superformer":
            #         cls_logits = outputs['cls']
            #         sp_logits = outputs.pop('sp_cls', None)
            #         cls_logits_sim = outputs.get('cls_similarities', None)
            #     else:
            #         cls_logits = outputs
            #         sp_logits = None
            #         cls_logits_sim = None
            #     # loss_cls = criterion(samples, cls_logits, targets) * args.cls_weight
            #     loss_cls = None
            #     loss_token, loss_token_pos, loss_token_neg = None, None, None
            #     if sp_logits is not None:
            #         assert args.superpixel_naive_weight > 0.0
            #         if not isinstance(criterion.base_criterion,
            #                           SoftTargetCrossEntropy):
            #             raise NotImplementedError()
            #         loss_token_pos, loss_token_neg = [
            #                 val * args.superpixel_naive_weight
            #                 if val is not None else None
            #                 for val in token_criterion(sp_logits, targets)]
            #         loss_token = loss_token_pos - (
            #                 loss_token_neg if loss_token_neg is not None else 0)

            #     loss_cls_similarities = None
            #     if cls_logits_sim is not None:
            #         assert args.superpixel_simil_cls_weight > 0.0
            #         loss_cls_similarities = (
            #                 criterion(samples, cls_logits_sim, targets)
            #                 * args.superpixel_simil_cls_weight)

            #     loss_total = sum([val for val in
            #                       [loss_cls, loss_token, loss_cls_similarities]
            #                       if val is not None])
            
            loss_total=loss_seg
            loss_value = loss_total.item()

            if not math.isfinite(loss_value):
                dump_path = os.path.join(args.output_dir, 'error_dump.pth')
                torch.save({
                    'model': model.module.state_dict(),
                    'samples': samples,
                    'targets': targets,
                    'optimizer': optimizer.state_dict(),
                }, dump_path)
                raise ValueError("Cls loss is {}, stopping training".format(loss_value))

            if loss_seg is not None and not math.isfinite(loss_seg.item()):
                error_dump(model, samples, targets)
                raise ValueError("Seg loss is {}, stopping training".format(loss_seg.item()))

            loss_scaler(loss_total, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order, update_grad=True)
            optimizer.zero_grad()

            torch.cuda.synchronize()
            if model_ema is not None:
                model_ema.update(model)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            # loss_names = ['cls', 'cls_simil', 'token', 'token_pos', 'token_neg', 'seg', 'total']
            # loss_values = [lv.item() if lv is not None else 0.0
            #                for lv in [loss_cls, loss_cls_similarities, loss_token,
            #                           loss_token_pos, loss_token_neg, loss_seg,
            #                           loss_total if loss_seg is None else loss_total + loss_seg]]

            # NOTE(meijieru): we only log cls loss for now.
            # NOTE: log seg for now
            loss_names = ['seg']
            loss_values = [lv.item() if lv is not None else 0.0 for lv in [loss_seg]]

            for name, lv in zip(loss_names, loss_values):
                metric_logger.update(**{f'loss_{name}': lv})

            losses_reduce = [utils.all_reduce_mean(lv) for lv in loss_values]
            if log_writer is not None:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((step / len(data_loader) + epoch) * 1000)
                for name, value in zip(loss_names, losses_reduce):
                    log_writer.add_scalar(f'train/loss/{name}', value, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    else:
                # use_seg = args.seg_weight > 0.0 and epoch < seg_cfg.get('num_epochs', 1e10)
        use_seg=True
        seg_criterion = None
        if use_seg:
            from mmseg.models import build_loss
            seg_criterion = build_loss(dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
            seg_criterion = partial(seg_criterion, ignore_index=255)
            pixel_iter = iter(data_loader)

        token_criterion = SoftTargetCrossEntropyTokenIntuitive(
                args.superpixel_naive_topk_pos, args.superpixel_naive_topk_neg,
                args.superpixel_loss_negative_max)

        optimizer.zero_grad()
        for step, val in enumerate(
                metric_logger.log_every(data_loader, print_freq, header)):
            seg_samples = val['img'].data[0].to(device, non_blocking=True)
            seg_targets = val['gt_semantic_seg'].data[0].to(device, non_blocking=True).squeeze(1)

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

            enable_autocast = args.precision != 'fp32'
            precision = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.precision]

            loss_seg = None

                # TODO(yunfei) conflict with torch version? & how to use model.module
                # with torch.cuda.amp.autocast(enabled=enable_autocast, dtype=precision):
            with torch.cuda.amp.autocast(enabled=enable_autocast):
                try:
                    is_Superformer = isinstance(model.module, Superformer)
                except AttributeError:
                    is_Superformer = model.__class__.__name__ == "Superformer"
                if is_Superformer:
                    kwargs = {'generate_seg': True,
                                'generate_simil_pred': False,
                                'return_pixel_logits': True,
                                'seg_stride': 1}
                try:
                    is_SuperformerBN = isinstance(model.module, SuperformerBottleNeck)
                except AttributeError:
                    is_SuperformerBN = model.__class__.__name__ == "SuperformerBottleNeck"
                if is_SuperformerBN:
                    kwargs = {'generate_seg': True,
                                'return_pixel_logits': True,
                                'seg_stride': 1}

                try:
                    is_SuperformerHR = isinstance(model.module, SuperformerHR)
                except AttributeError:
                    is_SuperformerHR = model.__class__.__name__ == "SuperformerHR"
                if is_Superformer or is_SuperformerHR or is_SuperformerBN is False:
                    raise ValueError()
                
                if 'kwargs' in locals():
                    seg_outputs = model(seg_samples, **kwargs)
                else:
                    seg_outputs = model(seg_samples)
                loss_seg = seg_criterion(seg_outputs['seg'], seg_targets) * args.seg_weight
                
                # Additional outputs
                for key in seg_outputs:
                    if key != "seg":
                        loss_seg += 0. * seg_outputs[key].sum()
                    
                # NOTE: temporary annotating for using loss_scaler for only one time
                # loss_scaler(loss_seg, optimizer, clip_grad=max_norm,
                #             parameters=model.parameters(), create_graph=is_second_order,
                #             # NOTE(meijieru): don't update params here.
                #             update_grad=False)
            
            # # with torch.cuda.amp.autocast(enabled=enable_autocast, dtype=precision):
            # with torch.cuda.amp.autocast(enabled=enable_autocast):
            #     if model.__class__.__name__ == "Superformer":
            #         kwargs = {'generate_seg': args.superpixel_naive_weight > 0.0,
            #                   'generate_simil_pred': args.superpixel_simil_cls_weight > 0.0,
            #                   'return_pixel_logits': False}
            #     if model.__class__.__name__ == "SuperformerHR":
            #         if args.superpixel_naive_weight > 0.0 or args.superpixel_simil_cls_weight > 0.0:
            #             raise ValueError()
            #         kwargs = {}
            #     else:
            #         kwargs = {}
            #     outputs = model(samples, **kwargs)

            #     if  model.__class__.__name__==  "SuperformerHR" or "Superformer":
            #         cls_logits = outputs['cls']
            #         sp_logits = outputs.pop('sp_cls', None)
            #         cls_logits_sim = outputs.get('cls_similarities', None)
            #     else:
            #         cls_logits = outputs
            #         sp_logits = None
            #         cls_logits_sim = None
            #     # loss_cls = criterion(samples, cls_logits, targets) * args.cls_weight
            #     loss_cls = None
            #     loss_token, loss_token_pos, loss_token_neg = None, None, None
            #     if sp_logits is not None:
            #         assert args.superpixel_naive_weight > 0.0
            #         if not isinstance(criterion.base_criterion,
            #                           SoftTargetCrossEntropy):
            #             raise NotImplementedError()
            #         loss_token_pos, loss_token_neg = [
            #                 val * args.superpixel_naive_weight
            #                 if val is not None else None
            #                 for val in token_criterion(sp_logits, targets)]
            #         loss_token = loss_token_pos - (
            #                 loss_token_neg if loss_token_neg is not None else 0)

            #     loss_cls_similarities = None
            #     if cls_logits_sim is not None:
            #         assert args.superpixel_simil_cls_weight > 0.0
            #         loss_cls_similarities = (
            #                 criterion(samples, cls_logits_sim, targets)
            #                 * args.superpixel_simil_cls_weight)

            #     loss_total = sum([val for val in
            #                       [loss_cls, loss_token, loss_cls_similarities]
            #                       if val is not None])
            
            loss_total=loss_seg
            loss_value = loss_total.item()

            if not math.isfinite(loss_value):
                dump_path = os.path.join(args.output_dir, 'error_dump.pth')
                torch.save({
                    'model': model.module.state_dict(),
                    'samples': samples,
                    'targets': targets,
                    'optimizer': optimizer.state_dict(),
                }, dump_path)
                raise ValueError("Cls loss is {}, stopping training".format(loss_value))

            if loss_seg is not None and not math.isfinite(loss_seg.item()):
                error_dump(model, samples, targets)
                raise ValueError("Seg loss is {}, stopping training".format(loss_seg.item()))

            loss_scaler(loss_total, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order, update_grad=True)
            optimizer.zero_grad()

            torch.cuda.synchronize()
            if model_ema is not None:
                model_ema.update(model)

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)

            # loss_names = ['cls', 'cls_simil', 'token', 'token_pos', 'token_neg', 'seg', 'total']
            # loss_values = [lv.item() if lv is not None else 0.0
            #                for lv in [loss_cls, loss_cls_similarities, loss_token,
            #                           loss_token_pos, loss_token_neg, loss_seg,
            #                           loss_total if loss_seg is None else loss_total + loss_seg]]

            # NOTE(meijieru): we only log cls loss for now.
            # NOTE: log seg for now
            loss_names = ['seg']
            loss_values = [lv.item() if lv is not None else 0.0 for lv in [loss_seg]]

            for name, lv in zip(loss_names, loss_values):
                metric_logger.update(**{f'loss_{name}': lv})

            losses_reduce = [utils.all_reduce_mean(lv) for lv in loss_values]
            if log_writer is not None:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((step / len(data_loader) + epoch) * 1000)
                for name, value in zip(loss_names, losses_reduce):
                    log_writer.add_scalar(f'train/loss/{name}', value, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        

@torch.no_grad()
def evaluate(data_loader, model, device, visualize_dir: Optional[str] = None, args=None):
    if args.seg_only is True:
        import numpy as np
        from PIL import Image

        postfix=''
        alpha = 0.5
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        from mmseg.models import build_loss
        seg_criterion = build_loss(dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        seg_criterion = partial(seg_criterion, ignore_index=0)
        pixel_iter=iter(data_loader)
        model.eval()

        results=[]
        loader_indices = data_loader.batch_sampler
        prog_bar = mmcv.ProgressBar(len(data_loader.dataset))

        # for i, val in enumerate(metric_logger.log_every(pixel_iter, 10, header)):
        # for i, val in enumerate(pixel_iter):
        for batch_indices, val in zip(loader_indices, data_loader):

            seg_samples = val['img'].data[0].to(device, non_blocking=True)
            # seg_targets = val['gt_semantic_seg'].data[0].to(device, non_blocking=True).squeeze(1)
            
            with torch.cuda.amp.autocast():
                # if isinstance(model.module, Superformer):
                try:
                    is_Superformer = isinstance(model.module, Superformer)
                except AttributeError:
                    is_Superformer = model.__class__.__name__ == "Superformer"
                if is_Superformer:
                    kwargs = {'generate_seg': True,
                                'generate_simil_pred': False,
                                'return_pixel_logits': True,
                                'seg_stride': 1}
                try:
                    is_SuperformerBN = isinstance(model.module, SuperformerBottleNeck)
                except AttributeError:
                    is_SuperformerBN = model.__class__.__name__ == "SuperformerBottleNeck"
                if is_SuperformerBN:
                    kwargs = {'generate_seg': True,
                                'return_pixel_logits': True,
                                'seg_stride': 1}

                try:
                    is_SuperformerHR = isinstance(model.module, SuperformerHR)
                except AttributeError:
                    is_SuperformerHR = model.__class__.__name__ == "SuperformerHR"
                if is_Superformer or is_SuperformerHR or is_SuperformerBN is False:
                    raise ValueError()
                
                if 'kwargs' in locals():
                    seg_outputs = model(seg_samples, **kwargs)
                else:
                    seg_outputs = model(seg_samples)
                # loss_seg = seg_criterion(seg_outputs['seg'], seg_targets) * args.seg_weight
                result = torch.argmax(seg_outputs['seg'].squeeze(0), dim=0)
                if args.vis ==True:
                    vis_result=result.cpu().numpy()
                result = [result.cpu().numpy()]
                result = data_loader.dataset.pre_eval(result, indices=batch_indices)
                results.extend(result)
                if args.vis == True:
                    from timm.data.constants import ADE20K_DEFAULT_MEAN, ADE20K_DEFAULT_STD
                    from superpixel.superpixel_ops import show_result
                    # seg_colormap = create_superpixel_colormap('random')
                    #origin image
                    _, _, ih, iw = seg_samples.shape
                    mean=np.array(ADE20K_DEFAULT_MEAN, dtype=np.float32)
                    std=np.array(ADE20K_DEFAULT_STD, dtype=np.float32)
                    vis_images = mmcv.imdenormalize(seg_samples[0].permute(1,2,0).cpu().numpy().astype(np.float32),mean,std,True)

                    #truth ground
                    seg_map = data_loader.dataset.get_gt_seg_map_by_idx(batch_indices[0])
                    seg_map[seg_map == 0] = 255
                    seg_map = seg_map - 1
                    seg_map[seg_map == 254] = 255

                    if seg_map.shape != seg_samples.shape[2:]:
                        seg_map= mmcv.imresize(seg_map,(512,512), interpolation='nearest')
                    
                    im_image = show_result(vis_images,seg_map,data_loader.dataset.CLASSES,data_loader.dataset.PALETTE,opacity=0.5)
                    im_image = Image.fromarray(im_image.astype(np.uint8))

                    im_seg = show_result(vis_images,vis_result,data_loader.dataset.CLASSES,data_loader.dataset.PALETTE,opacity=0.5)                    
                    im_seg = Image.fromarray(im_seg.astype(np.uint8))
                    width, height = im_image.size[0], im_image.size[1]
                    im_concat = Image.new('RGB', (width*2, height))
                    im_concat.paste(im_image, (0, 0))
                    im_concat.paste(im_seg, (width, 0))
                    im_concat.save(f'{visualize_dir}/vis_{batch_indices}_seg{postfix}.png')
                    
                    resize_similarities = True
                    tmp = model.visualize_superpixel(resize_similarities)
                    for key, val in tmp.items():
                        im = val[0].squeeze().numpy().astype(np.uint8)
                        resize_output = not resize_similarities or im.shape[0] != ih
                        im = Image.fromarray(im)
                        if resize_output:
                            print('resize_output')
                            im = im.resize((ih, iw), Image.Resampling.NEAREST)
                        im.save(f'{visualize_dir}/vis_{batch_indices}_{key}_{resize_similarities}{postfix}.png')

                        im_alpha = np.array(im_image, dtype=np.float32) * alpha + np.array(im, dtype=np.float32) * (1 - alpha)
                        im_alpha = Image.fromarray(im_alpha.astype(np.uint8))
                        im_alpha.save(f'{visualize_dir}/vis_{batch_indices}_alpha_{key}_{resize_similarities}{postfix}.png')



                # ret_metrics = mean_iou(
                #     seg_outputs['seg'].squeeze(0).detach().cpu().numpy(), seg_targets.detach().cpu().numpy(), 150, ignore_index=255, nan_to_num=-1)
            batch_size = seg_samples.shape[0]
            # metric_logger.update(loss=loss_seg.item())
            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()
                
        ret_metrics = pre_eval_to_metrics(pre_eval_results=results, metrics=['mIoU'])
                # Because dataset.CLASSES is required for per-eval.
        eval_results = {}

        class_names = data_loader.dataset.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print('per class results:')
        print(class_table_data)
        print('Summary:')
        print(summary_table_data)
        
        # each metric dict
        
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })


        metric_logger.meters['mIoU'].update(eval_results['mIoU'], n=batch_size)
        metric_logger.meters['aAcc'].update(eval_results['aAcc'], n=batch_size)
        metric_logger.meters['mAcc'].update(eval_results['mAcc'], n=batch_size)

        # metric_logger.synchronize_between_processes()
        print('* mIoU {mIoU.global_avg:.3f}  aAcc{aAcc.global_avg:.3f} mAcc{mAcc.global_avg:.3f}'
                .format(mIoU=metric_logger.mIoU, aAcc=metric_logger.aAcc,mAcc=metric_logger.mAcc))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    else:
        criterion = torch.nn.CrossEntropyLoss()

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'

        # switch to evaluation mode
        model.eval()

        for i, (images, target) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            postfix=''
            if args is not None and args.occluded_eval > 0:
                patch_size = int(args.occluded_eval)
                y, x = 80, 80
                images[:, :, y:y + patch_size, x:x + patch_size] = 0
                # __import__('ipdb').set_trace()

            # compute output
            model_without_ddp = getattr(model, 'module', model)
            with torch.cuda.amp.autocast():
                if isinstance(model_without_ddp, Superformer):
                    kwargs = {
                        'generate_seg':
                            visualize_dir is not None and
                            model_without_ddp.token_specific_classifier,
                        'generate_simil_pred':
                            model_without_ddp.similarities_classifier
                    }
                elif isinstance(model_without_ddp, SuperformerHR):
                    # TODO(meijieru): fix this
                    kwargs = {}
                else:
                    kwargs = {}
                output = model(images, **kwargs)

                if isinstance(model_without_ddp, (Superformer, SuperformerHR)):
                    cls_logits_sim = output.get('cls_similarities', None)
                    seg_logits = output.get('seg', None)
                    cls_logits = output['cls']
                else:
                    cls_logits_sim = None
                    seg_logits = None
                    cls_logits = output
                loss = criterion(cls_logits, target)

            if visualize_dir is not None:
                import os
                from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
                from PIL import Image
                import numpy as np

                os.makedirs(visualize_dir, exist_ok=True)

                _, _, ih, iw = images.shape
                
                vis_images = images.cpu() * torch.tensor(IMAGENET_DEFAULT_STD).reshape(
                    1, 3, 1, 1) + torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(
                        1, 3, 1, 1)
                vis_images *= 255.0
                im_image = Image.fromarray(vis_images[0].squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                im_image.save(f'{visualize_dir}/vis_{i}_image{postfix}.png')

                if seg_logits is not None:
                    seg_pred = seg_logits.argmax(1).cpu()
                    seg_colormap = create_superpixel_colormap('random')
                    vis = seg_colormap[seg_pred[0] % seg_colormap.size(0)].cpu().numpy()
                    im_seg = Image.fromarray(vis.astype(np.uint8))
                    im_seg.save(f'{visualize_dir}/vis_{i}_seg{postfix}.png')

                resize_similarities = True
                tmp = model.visualize_superpixel(resize_similarities)
                for key, val in tmp.items():
                    im = val[0].squeeze().numpy().astype(np.uint8)
                    resize_output = not resize_similarities or im.shape[0] != ih
                    im = Image.fromarray(im)
                    if resize_output:
                        print('resize_output')
                        im = im.resize((ih, iw), Image.Resampling.NEAREST)
                    im.save(f'{visualize_dir}/vis_{i}_{key}_{resize_similarities}{postfix}.png')

                    alpha = 0.5
                    im_alpha = np.array(im_image, dtype=np.float32) * alpha + np.array(im, dtype=np.float32) * (1 - alpha)
                    im_alpha = Image.fromarray(im_alpha.astype(np.uint8))
                    im_alpha.save(f'{visualize_dir}/vis_{i}_alpha_{key}_{resize_similarities}{postfix}.png')

                if i > 10:
                    break

            acc1, acc5 = accuracy(cls_logits, target, topk=(1, 5))
            acc1_sim = 0
            if cls_logits_sim is not None:
                acc1_sim, = accuracy(cls_logits_sim, target, topk=(1,))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            if cls_logits_sim is not None:
                metric_logger.meters['acc1_sim'].update(acc1_sim.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
        if cls_logits_sim is not None:
            print('* Sim Acc@1 {top1.global_avg:.3f}'
                    .format(top1=metric_logger.acc1_sim))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_imagenetx(data_loader, model, device, visualize_dir: str, args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = torch.zeros(len(data_loader.dataset), dtype=torch.int64)
    probs = torch.zeros(len(data_loader.dataset))
    num_examples = 0

    # switch to evaluation mode
    model.eval()

    from timm.data.real_labels import RealLabelsImagenet

    real_labels = RealLabelsImagenet([val[0] for val in data_loader.dataset.samples], real_json='./data/real.json')

    for i, (images, target) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        model_without_ddp = getattr(model, 'module', model)
        with torch.cuda.amp.autocast():
            kwargs = {}
            output = model(images, **kwargs)

            cls_logits = output
            loss = criterion(cls_logits, target)

        prob = output.softmax(1)
        prob, pred = prob.topk(1, 1, True, True)

        preds[num_examples: num_examples + output.size(0)] = pred.cpu().detach().squeeze()
        probs[num_examples: num_examples + output.size(0)] = prob.cpu().detach().squeeze()
        num_examples += output.size(0)

        real_labels.add_result(cls_logits)

        acc1, acc5 = accuracy(cls_logits, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    assert num_examples == len(data_loader.dataset)

    print(real_labels.get_accuracy())
    __import__('ipdb').set_trace()

    import csv

    __import__('ipdb').set_trace()
    os.makedirs(visualize_dir, exist_ok=True)
    with open(os.path.join(visualize_dir, "imagenetx.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "predicted_class", "predicted_probability"])
        writer.writeheader()

        for item, pred, prob in zip(data_loader.dataset.samples, preds, probs):
            filename = os.path.basename(item[0])
            row = {"file_name": filename, "predicted_class": pred.item(), "predicted_probability": prob.item()}
            writer.writerow(row)

    from imagenet_x import get_factor_accuracies, error_ratio
    from imagenet_x import plots
    import numpy as np

    # path = "./remote_ckpt/asym_bottleneck_conv4_small_nofinal_head2_spilavg_epoch300_clsweight1.0_bs256/eval/imagenetx_v2_tmp/"
    factor_accs = get_factor_accuracies(visualize_dir)
    error_ratios = error_ratio(factor_accs)

    plots.model_comparison(
        factor_accs.reset_index(), fname=os.path.join(visualize_dir, "./imagenetx.pdf")
    )
    np.savetxt(os.path.join(visualize_dir, 'values.txt'), error_ratios.values, fmt='%f')


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
