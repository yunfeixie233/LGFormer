from typing import Callable, Optional, Sequence, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import superpixel_ops

BaseModule = nn.Module


def pad_sequence_with_none(sequence, target_length):
    return list(sequence) + [None] * (target_length - len(sequence))


def reshape_and_transpose_for_attention_operation(
    inputs: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Sequentially reshapes and transposes the tensor.

    Args:
      inputs: An input [b, num_heads * c, h, w] tensor.
      num_heads: An integer, the number of attention heads.

    Returns:
      output: An output [b, num_heads, c, h, w] tensor.
    """
    b, c, h, w = inputs.shape
    assert c % num_heads == 0
    return inputs.view([b, num_heads, c // num_heads, h, w])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "glu":
        return F.glu
    elif activation == "none":
        return lambda x: x
    else:
        raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class Conv2D(BaseModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bn: bool = True,
        bn_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.use_bn = use_bn

        self.conv = nn.Conv2d(
            in_features, out_features, kernel_size=1, bias=not use_bn
        )
        if self.use_bn:
            self.bn = bn_layer(out_features)
        self.activation = _get_activation_fn(activation)

    def forward(self, x: torch.Tensor):
        # x: [b, c, h, w]
        res = self.conv(x)
        if self.use_bn:
            res = self.bn(res)
        return self.activation(res)


class DualPathTransformerLayer(BaseModule):
    """Applies a transformer layer, as proposed in MaX-DeepLab models."""

    def __init__(
        self,
        in_features: int,
        activation="relu",
        filters=128,
        num_heads=8,
        bottleneck_expansion=2,
        key_expansion=1,
        value_expansion=2,
        feed_forward_network_channels=2048,
        use_memory_self_attention=True,
        use_memory2pixel_feedback_attention=True,
        use_pixel2memory_feedback_attention=True,
        use_kmeans_cross_attention=False,
        use_superpixel_cross_attention=False,
        transformer_activation="softmax",
        bn_layer=nn.BatchNorm2d,
        conv_kernel_weight_decay=0.0,
        memory_pre_activation: bool = False,
        pixel_shape=None,
        superpixel_shape=None,
        position_embedding_method='none',
        position_embedding_stride=4,
    ):
        """Initializes a DualPathTransformerLayer.

        This function implements a dual path transformer layer between a pixel space
        and a memory space, as described in the MaX-DeepLab paper. In this dual path
        transformer, the memory2pixel cross attention and the memory self-attention
        share a single activation, e.g. softmax.

        The flag "use_kmeans_cross_attention" enables k-means cross-attention
        proposed in the kMaX-DeepLab paper, which regards the memory (object query)
        as cluster center, and updates them in a k-means clustering manner.

        Reference:
          MaX-DeepLab: "End-to-End Panoptic Segmentation with Mask Transformers",
            CVPR 2021. https://arxiv.org/abs/2012.00759
              Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

          k-means Mask Transformer, ECCV 2022.
            Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu,
            Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

        Args:
          name: A string, the name of this dual path transformer layer.
          activation: A string, type of activation function to apply.
          filters: An integer, the base number of channels for the layer.
          num_heads: An integer, the number of heads in multi-head attention.
          bottleneck_expansion: A float, the channel expansion ratio for the
            bottleneck.
          key_expansion: A float, the channel expansion ratio for keys.
          value_expansion: A float, the channel expansion ratio for values.
          feed_forward_network_channels: An integer, the number of channels for the
            feed_forward_network. Zero means no feed_forward_network will be
            applied.
          use_memory_self_attention: A boolean, whether to apply the memory space
            self-attention.
          use_memory2pixel_feedback_attention: A boolean, whether to apply the
            memory2pixel feedback attention.
          use_pixel2memory_feedback_attention: A boolean, whether to apply the
            pixel2memory feedback attention.
          use_kmeans_cross_attention: A boolean, whether to apply the kmeans
            cross-attention.
          transformer_activation: A string, type of activation function for
            self-attention. Support 'sigmoid' and 'softmax'.
          bn_layer: A tf.keras.layers.Layer that computes the normalization
            (default: tf.keras.layers.BatchNormalization).
          conv_kernel_weight_decay: A float, the weight decay for convolution
            kernels.
          auxiliary_predictor_func: A callable function that returns an
            initialization of auxiliary predictor.

        Raises:
          ValueError: If filters * key_expansion is not divisible by num_heads.
          ValueError: If filters * value_expansion is not divisible by num_heads.
          ValueError: If both use_memory2pixel_feedback_attention and
            use_kmeans_cross_attention are False.
          ValueError: If use_kmeans_cross_attention is True but
            auxiliary_predictor_func is None.
        """
        super(DualPathTransformerLayer, self).__init__()

        bottleneck_channels = int(round(filters * bottleneck_expansion))
        total_key_depth = int(round(filters * key_expansion))
        total_value_depth = int(round(filters * value_expansion))

        if total_key_depth % num_heads:
            raise ValueError(
                "Total_key_depth should be divisible by num_heads."
            )

        if total_value_depth % num_heads:
            raise ValueError(
                "Total_value_depth should be divisible by num_heads."
            )

        if not (
            use_memory2pixel_feedback_attention or use_kmeans_cross_attention
        ):
            raise ValueError(
                "At least one of use_memory2pixel_feedback_attention or"
                " use_kmeans_cross_attention needs to be enabled."
            )

        # Compute query key value with one convolution and a batch norm layer. The
        # initialization std is standard transformer initialization (without batch
        # norm), as used in SASA and ViT. In our case, we use batch norm by default,
        # so it does not require careful tuning. If one wants to remove all batch
        # norms in axial attention, this standard initialization should still be
        # good, but a more careful initialization is encouraged.
        # FIXME(meijieru): initialization
        # initialization_std = bottleneck_channels**-0.5

        memory_in_features = in_features
        self._memory_conv1_bn_act = Conv2D(
            memory_in_features,
            bottleneck_channels,
            use_bn=True,
            bn_layer=bn_layer,
            activation=activation,
        )

        self._pixel_conv1_bn_act = Conv2D(
            in_features,
            bottleneck_channels,
            use_bn=True,
            bn_layer=bn_layer,
            activation=activation,
        )

        # We always compute the query for memory space, since it gathers information
        # from the pixel space and thus cannot be removed. We compute the key and
        # value for memory space only when they are necessary (i.e. either
        # use_memory_self_attention or use_pixel2memory_feedback_attention).
        if use_memory_self_attention or use_pixel2memory_feedback_attention or use_superpixel_cross_attention:
            # TODO(meijieru): tf initialization.
            self._memory_qkv_conv_bn = Conv2D(
                bottleneck_channels,
                total_key_depth * 2 + total_value_depth,
                use_bn=True,
                bn_layer=bn_layer,
                activation="none",
            )
        elif use_memory2pixel_feedback_attention:
            raise NotImplementedError("removed from tf")
        else:
            raise NotImplementedError()

        # For the pixel space, we always compute the key and value, since they
        # provide information for the memory space and thus cannot be removed. We
        # compute the query for pixel space only when it is necessary (i.e.
        # use_pixel2memory_feedback_attention is True).
        if use_pixel2memory_feedback_attention or use_superpixel_cross_attention:
            self._pixel_qkv_conv_bn = Conv2D(
                bottleneck_channels,
                total_key_depth * 2 + total_value_depth,
                use_bn=True,
                bn_layer=bn_layer,
                activation="none",
            )
        elif use_memory2pixel_feedback_attention:
            raise NotImplementedError("removed from tf")
        else:
            raise NotImplementedError("removed from tf")

        self._use_memory_self_attention = use_memory_self_attention
        self._use_memory2pixel_feedback_attention = (
            use_memory2pixel_feedback_attention
        )
        self._use_pixel2memory_feedback_attention = (
            use_pixel2memory_feedback_attention
        )
        self._use_kmeans_cross_attention = use_kmeans_cross_attention
        self._use_superpixel_cross_attention = use_superpixel_cross_attention
        self._bottleneck_channels = bottleneck_channels
        self._total_key_depth = total_key_depth
        self._total_value_depth = total_value_depth
        self._num_heads = num_heads
        self._bn_layer = bn_layer
        self._conv_kernel_weight_decay = conv_kernel_weight_decay
        self._activation = activation
        self._activation_fn = _get_activation_fn(activation)
        self._feed_forward_network_channels = feed_forward_network_channels
        self._memory_pre_activation = memory_pre_activation
        self._position_embedding_method = position_embedding_method
        self._position_embedding_stride = position_embedding_stride

        # Here we follow ResNet bottleneck blocks: we apply a batch norm with gamma
        # initialized at zero, followed by drop path and an activation function.
        # Initializing this gamma at zero ensures that at random initialization of
        # the model, the skip connections dominate all residual blocks. In this way,
        # all the skip connections construct an identity mapping that passes the
        # gradients (without any distortion from the randomly initialized blocks) to
        # all residual blocks. This helps training at early epochs.
        # Reference: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour".
        # https://arxiv.org/abs/1706.02677
        if (
            self._use_memory2pixel_feedback_attention
            or self._use_memory_self_attention
        ):
            self._memory_conv3_bn = Conv2D(
                self._total_value_depth,
                in_features,
                use_bn=True,
                bn_layer=self._bn_layer,
                # bn_gamma_initializer="zeros",
                activation="none",
                # conv_kernel_weight_decay=self._conv_kernel_weight_decay,
            )

        if self._feed_forward_network_channels > 0:
            # Again, we follow ResNet bottleneck blocks: we apply a batch norm with
            # gamma initialized at zero, followed by drop path and an activation
            # function.
            self._memory_ffn = nn.Sequential(
                *[
                    Conv2D(
                        memory_in_features,
                        self._feed_forward_network_channels,
                        use_bn=True,
                        bn_layer=self._bn_layer,
                        activation=self._activation,
                        # conv_kernel_weight_decay=self._conv_kernel_weight_decay,
                    ),
                    Conv2D(
                        self._feed_forward_network_channels,
                        memory_in_features,
                        use_bn=True,
                        bn_layer=self._bn_layer,
                        # bn_gamma_initializer="zeros",
                        activation="none",
                        # conv_kernel_weight_decay=self._conv_kernel_weight_decay,
                    ),
                ]
            )
        if self._use_pixel2memory_feedback_attention:
            self._pixel_conv3_bn = Conv2D(
                self._total_value_depth,
                in_features,
                use_bn=True,
                bn_layer=self._bn_layer,
                # bn_gamma_initializer="zeros",
                activation="none",
                # conv_kernel_weight_decay=self._conv_kernel_weight_decay,
            )

        if self._position_embedding_method == 'none':
            pass
        elif self._position_embedding_method == 'learnable':
            self.sp_pos_embed = self._create_pos_embed(memory_in_features,
                                                       superpixel_shape)
            self.pixel_pos_embed = self._create_pos_embed(
                in_features, pixel_shape)
            print(f'sp_pos_embed shape: {self.sp_pos_embed.shape}, '
                  f'pixel_pos_embed shape: {self.pixel_pos_embed.shape}')
        else:
            raise ValueError()

    def _create_pos_embed(self, dim, shape) -> torch.Tensor:
        pos_embed_shape = [
            int(math.ceil(val / self._position_embedding_stride))
            for val in shape
        ]
        for val in pos_embed_shape:
            assert val > 1
        pos_embed = nn.Parameter(torch.zeros(1, dim, *pos_embed_shape))
        return pos_embed

    def _add_pos_embed(self, val, pos_embed) -> torch.Tensor:
        _, _, h, w = val.shape
        return val + F.interpolate(
            pos_embed, size=(h, w), mode='bilinear', align_corners=False)

    def _split_qkv(self, val: torch.Tensor) -> Sequence[Optional[torch.Tensor]]:
        query, key, value = torch.split(
            val,
            [
                self._total_key_depth,
                self._total_key_depth,
                self._total_value_depth,
            ],
            dim=1,
        )

        query, key, value = [
            reshape_and_transpose_for_attention_operation(val, self._num_heads)
            for val in [query, key, value]
        ]

        return query, key, value

    def _expand_single_sp_features(self, val: torch.Tensor, b: int):
        # TODO(meijieru): temporarily solution to deal with the heads.
        assert val.shape[0] == 1
        return val.expand(b, -1, -1, -1, -1)

    def _superpixel_update_step(
        self,
        sp_query: torch.Tensor,
        pixel_key: torch.Tensor,
        pixel_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsp, _, _, sh, sw = sp_query.shape
        b, num_heads, c_key, h, w = pixel_key.shape
        _, _, c_value, _, _ = pixel_value.shape
        if bsp == 1:
            sp_query = self._expand_single_sp_features(sp_query, b)
        else:
            assert b == bsp
        scale = c_key**-0.5
        # [b * num_heads]
        sp_query, pixel_key, pixel_value = [
            val.flatten(end_dim=1) for val in [sp_query, pixel_key, pixel_value]
        ]
        similarities = (
            superpixel_ops.compute_similarities_dot_product(pixel_key, sp_query)
            * scale
        )
        sp_delta = superpixel_ops.update_superpixel_features(
            pixel_value, sp_query, similarities
        )
        sp_delta = self._memory_conv3_bn(
            sp_delta.reshape(b, num_heads * c_value, sh, sw)
        )
        return sp_delta, similarities.reshape(b, num_heads, 9, h, w)

    def _superpixel_assign_step(
        self,
        pixel_query: torch.Tensor,
        sp_key: torch.Tensor,
        sp_value: torch.Tensor,
    ) -> torch.Tensor:
        bsp, _, _, _, _ = sp_key.shape
        b, num_heads, c_key, h, w = pixel_query.shape
        _, _, c_value, _, _ = sp_value.shape
        if bsp == 1:
            sp_key = self._expand_single_sp_features(sp_key, b)
            sp_value = self._expand_single_sp_features(sp_value, b)
        else:
            assert b == bsp
        scale = c_key**-0.5
        pixel_query, sp_key, sp_value = [
            val.flatten(end_dim=1) for val in [pixel_query, sp_key, sp_value]
        ]
        similarities = (
            superpixel_ops.compute_similarities_dot_product(pixel_query, sp_key)
            * scale
        )
        pixel_delta = superpixel_ops.update_pixel_features(
            None, sp_value, similarities
        )
        pixel_delta = self._pixel_conv3_bn(
            pixel_delta.reshape(b, num_heads * c_value, h, w)
        )
        return pixel_delta

    def forward(self, inputs):
        """Performs a forward pass.

        We have to define drop_path_masks outside the layer call and pass it into
        the layer call, because recompute_grad (gradient checkpointing) does not
        allow any randomness within the function call. In addition, recompute_grad
        only supports float tensors as inputs. For this reason, the training flag
        should be also passed as a float tensor. For the same reason, we cannot
        support passing drop_path_random_mask as None. Instead, we ask the users to
        pass only the first two tensors when drop path is not used.

        Args:
          inputs: A tuple of 4 or 8 tensors, containing
            pixel_space_input should be a [batch, h, w
              pixel_space_channels] tensor.
            memory_space_input should be a [batch, sh, sw,
              memory_space_channels] tensor.
            auxiliary_outputs should be a tuple containing auxiliary outputs, where
              each element has the dictionary type.
            float_tensor_training should be a float tensor of 0.0 or 1.0, whether
              the model is in training mode.
            (optional) pixel_space_drop_path_mask is a drop path mask tensor of
              shape [batch, 1, 1] for the pixel space.
            (optional) memory_space_attention_drop_path_mask is a drop path mask
              tensor of shape [batch, 1, 1] for the memory space.
            (optional) memory_kmeans_attention_drop_path_mask is a drop path mask
              tensor of shape [batch, 1, 1] for the memory space.
            (optional) memory_space_feed_forward_network_drop_path_mask is a drop
              path mask tensor of shape [batch, 1, 1] for the memory space feed
              forward network.

        Returns:
          pixel_space_output: A [batch, num_pixel, pixel_space_channels] tensor.
          memory_space_output: A [batch, sh, sw, memory_space_channels]
            tensor.
          auxiliary_outputs: A tuple containing auxiliary outputs, where each
            element has the dictionary type.

        Raises:
          ValueError: If the length of inputs is not 4 or 8.
        """
        if len(inputs) not in (2, 5):
            raise ValueError("The length of inputs should be either 2 or 5.")

        # Unpack the inputs.
        (
            pixel_space_input,
            memory_space_input,
            pixel_space_drop_path_mask,
            memory_space_attention_drop_path_mask,
            memory_space_feed_forward_network_drop_path_mask,
        ) = pad_sequence_with_none(inputs, target_length=5)

        # Similar to the ResNet bottleneck design, we do an input down projection
        # in both the pixel space and the memory space.
        # [b, csp, sh, sw]
        # NOTE(meijieru): If the b == 1, degrade to instance normalization.
        memory_space_embeded = memory_space_input
        pixel_space_embeded = pixel_space_input
        if self._position_embedding_method != 'none':
            memory_space_embeded = self._add_pos_embed(memory_space_embeded,
                                                       self.sp_pos_embed)
            pixel_space_embeded = self._add_pos_embed(pixel_space_embeded,
                                                      self.pixel_pos_embed)
        memory_space = self._memory_conv1_bn_act(memory_space_embeded)

        # NOTE(meijieru): Diff from segmentation impl. activation removed.
        # [b, cp, h, w]
        pixel_space = self._pixel_conv1_bn_act(pixel_space_embeded)

        if (
            self._use_memory_self_attention
            or self._use_pixel2memory_feedback_attention
            or self._use_superpixel_cross_attention
        ):
            memory_space_qkv = self._memory_qkv_conv_bn(memory_space)
            memory_query, memory_key, memory_value = self._split_qkv(
                memory_space_qkv
            )
        elif self._use_memory2pixel_feedback_attention:
            raise NotImplementedError("removed from tf")
        else:
            raise NotImplementedError()

        if self._use_pixel2memory_feedback_attention or self._use_superpixel_cross_attention or self._use_superpixel_cross_attention:
            pixel_space_qkv = self._pixel_qkv_conv_bn(pixel_space)
            pixel_query, pixel_key, pixel_value = self._split_qkv(
                pixel_space_qkv
            )
        elif self._use_memory2pixel_feedback_attention:
            raise NotImplementedError("removed from tf")
        else:
            raise NotImplementedError("removed from tf")

        memory_space_output = memory_space_input
        pixel_space_output = pixel_space_input

        # Perform kmeans cross-attention.
        if self._use_kmeans_cross_attention:
            raise NotImplementedError("removed from tf")
        if not self._use_superpixel_cross_attention:
            raise NotImplementedError("removed from tf")
        if self._use_memory_self_attention:
            raise NotImplementedError("removed from tf")

        similarities_multi_head = None
        if self._use_superpixel_cross_attention:
            sp_feature_delta, similarities_multi_head = self._superpixel_update_step(
                memory_query, pixel_key, pixel_value
            )
            if memory_space_attention_drop_path_mask is not None:
                memory_space = (
                    memory_space * memory_space_attention_drop_path_mask
                )
            memory_space_output = memory_space_output + sp_feature_delta

        memory_space_output = self._activation_fn(memory_space_output)

        # Apply an optional feed-forward network to the memory space.
        if self._feed_forward_network_channels > 0:
            memory_space = self._memory_ffn(memory_space_output)
            if memory_space_feed_forward_network_drop_path_mask is not None:
                memory_space = (
                    memory_space
                    * memory_space_feed_forward_network_drop_path_mask
                )
            memory_space_output = self._activation_fn(
                memory_space_output + memory_space
            )

        # Perform P2M attention.
        # Compute pixel space attention and the output projection only when
        # pixel2memory_feedback_attention is used.
        if self._use_pixel2memory_feedback_attention:
            pixel_space = self._superpixel_assign_step(
                pixel_query, memory_key, memory_value
            )
            if pixel_space_drop_path_mask is not None:
                pixel_space = pixel_space * pixel_space_drop_path_mask
            pixel_space_output = pixel_space_input + pixel_space

        return pixel_space_output, memory_space_output, similarities_multi_head

    # def init_weights(self):
    #     # initialization_std = self._bottleneck_channels**-0.5
    #     initialization_std = 0.02
    #     for conv in [
    #         self._memory_qkv_conv_bn.conv,
    #         self._pixel_qkv_conv_bn.conv,
    #     ]:
    #         nn.init.trunc_normal_(conv.weight, std=initialization_std)
    #         if conv.bias is not None:
    #             nn.init.zeros_(conv.bias)

    #     # for bn in [
    #     #     self._memory_conv3_bn.bn,
    #     #     # self._memory_ffn[-1].bn,
    #     #     self._pixel_conv3_bn.bn,
    #     # ]:
    #     #     nn.init.constant_(bn.weight, 1e-3)
