"""Test for SuperPixel Transformer Head."""
import unittest
import numpy as np
import torch

import superpixel_models as sp


class SuperPixelTransformerTest(unittest.TestCase):
    def test_superpixel_stem(self):
        b, h, w = 1, 224, 224
        for tokenization_method in ["patch", "identity"]:
            x = torch.zeros((b, 3, h, w))
            module = sp.superformer_small_patch16_224(
                superpixel_kwargs=dict(tokenization_method=tokenization_method)
            )
            res = module(x)
            self.assertTupleEqual(res['cls'].shape, (b, 1000))

            tokenization_info = module.tokenization_info
            self.assertIsInstance(tokenization_info, (type(None), dict))

    def test_no_weight_decay(self):
        module = sp.superformer_small_patch16_224(
            superpixel_kwargs=dict(
                tokenization_method="identity",
                superpixel_features_init_method="learnable",
            )
        )
        no_weight_decay = module.no_weight_decay()
        self.assertSetEqual(
            no_weight_decay,
            {
                "pos_embed",
                "dist_token",
                "cls_token",
                "patch_embed.superpixel_tokenize.init_superpixel_queries",
            },
        )

    def test_no_hypercolumn(self):
        module = (
            sp.superformer_tiny_iter2_stride8_patch32_224_notoken_prenorm_dualpath()
        )
        self.assertEqual(
            len(list(module.patch_embed.hyper_column.parameters())), 0
        )


if __name__ == "__main__":
    unittest.main()
