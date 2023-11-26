# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .encoder_decoder import EncoderDecoder
import mmcv
from mmseg.structures import SegDataSample
from ..utils import resize
@MODELS.register_module()
class EncoderDecoderJoint(EncoderDecoder):
    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels
        self.out_channels_part = self.decode_head.out_channels_part    
        self.out_channels_obj = self.decode_head.out_channels_obj    
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501
    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        out_channels_part = self.out_channels_part
        out_channels_obj = self.out_channels_obj
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds_part = inputs.new_zeros((batch_size, out_channels_part, h_img, w_img))
        preds_obj = inputs.new_zeros((batch_size, out_channels_obj, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        seg_logits_dict = dict()
        seg_logits_dict['preds_part'] = preds_part
        seg_logits_dict['preds_obj'] = preds_obj
        
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                if crop_img.shape[-1] !=  crop_img.shape[-2]:
                    max_side = max(crop_img.shape[2], crop_img.shape[3])
                    crop_img = F.interpolate(crop_img, size = max_side,mode = 'bicubic')                
                crop_seg_logit_dict = self.encode_decode(crop_img, batch_img_metas)
                
                seg_logits_dict['preds_part'] += F.pad(crop_seg_logit_dict['part'],
                               (int(x1), int(seg_logits_dict['preds_part'].shape[3] - x2), int(y1),
                                int(seg_logits_dict['preds_part'].shape[2] - y2)))
                seg_logits_dict['preds_obj'] += F.pad(crop_seg_logit_dict['obj'],
                               (int(x1), int(seg_logits_dict['preds_obj'].shape[3] - x2), int(y1),
                                int(seg_logits_dict['preds_obj'].shape[2] - y2)))                                

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits_dict['preds_obj']  = seg_logits_dict['preds_obj']  / count_mat
        seg_logits_dict['preds_part']  = seg_logits_dict['preds_part']  / count_mat
        return seg_logits_dict

    def postprocess_result(self,
                           seg_logits_dict: dict,
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        seg_logits_part = seg_logits_dict['preds_part']
        seg_logits_obj = seg_logits_dict['preds_obj']
        batch_size, C_part, H, W = seg_logits_part.shape
        _, C_obj, _, _ = seg_logits_obj.shape
        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits_part = seg_logits_part[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]
                i_seg_logits_obj = seg_logits_obj[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]                

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits_part = i_seg_logits_part.flip(dims=(3, ))
                        i_seg_logits_obj = i_seg_logits_obj.flip(dims=(3, ))                        
                    else:
                        i_seg_logits_obj = i_seg_logits_obj.flip(dims=(2, ))
                        i_seg_logits_part = i_seg_logits_part.flip(dims=(2, ))                        

                # resize as original shape
                i_seg_logits_obj = resize(
                    i_seg_logits_obj,
                    size=img_meta['ori_shape'],              
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
                i_seg_logits_part = resize(
                    i_seg_logits_part,
                    size=img_meta['ori_shape'],              
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)                
            else:
                i_seg_logits_part = seg_logits_part[i]
                i_seg_logits_obj = seg_logits_obj[i]
            if C_part > 1:
                i_seg_pred_part = i_seg_logits_part.argmax(dim=0, keepdim=True)
            else:
                i_seg_pred_part = i_seg_pred_part.sigmoid()
                i_seg_pred_part = (i_seg_logits_part >
                              self.decode_head.threshold).to(i_seg_logits_part)                
            if C_obj > 1:
                i_seg_pred_obj = i_seg_logits_obj.argmax(dim=0, keepdim=True)
            else:
                i_seg_pred_obj = i_seg_pred_obj.sigmoid()
                i_seg_pred_obj = (i_seg_logits_obj >
                              self.decode_head.threshold).to(i_seg_logits_obj)               

            data_samples[i].set_data({
                'seg_logits_part':
                PixelData(**{'data': i_seg_logits_part}),
                'seg_logits_obj':
                PixelData(**{'data': i_seg_logits_obj}),                
                'pred_sem_seg_part':
                PixelData(**{'data': i_seg_pred_part}),
                'pred_sem_seg_obj':
                PixelData(**{'data': i_seg_pred_obj})                
            })

        return data_samples
