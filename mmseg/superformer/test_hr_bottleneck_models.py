"""Test for SuperPixel Transformer Head."""
import unittest
import numpy as np
import torch
import torch.nn as nn
import timm.models.layers as layers

import hr_bottleneck_models as sp


class HRSuperPixelTransformerTest(unittest.TestCase):
    def test_forward(self):
        b, h, w = 1, 224, 224
        for tokenization_method in ["sp_cross", "patch"]:
            x = torch.zeros((b, 3, h, w))
            module = sp.hr_bottleneck_small_nofinal(sp_method=tokenization_method)
            res = module(x)
            self.assertTupleEqual(res.shape, (b, 1000))

    def test_no_weight_decay(self):
        module = sp.hr_bottleneck_small_nofinal()
        no_weight_decay = module.no_weight_decay()
        for name in no_weight_decay:
            self.assertTrue(name.endswith("pos_embed"))

    def test_model(self):
        module = sp.hr_bottleneck_small_nofinal(img_size=224)
        self.assertEqual(len(module.stages[0].blocks), 2)
        self.assertEqual(len(module.stages[1].blocks), 10)
        self.assertEqual(module.stem.conv_layers[0][0].kernel_size, (4, 4))
        self.assertEqual(module.stem.conv_layers[0][0].stride, (4, 4))
        self.assertEqual(module.use_middle_pixel_features, False)
        for stage in module.stages:
            self.assertEqual(stage.use_pixel_similarities, False)
            self.assertEqual(stage.patch_embed.num_heads, 1)
            self.assertSequenceEqual(stage.patch_embed.superpixel_shape, (14, 14))


if __name__ == "__main__":
    sp.SKIP_CONFIRM = True
    unittest.main()
