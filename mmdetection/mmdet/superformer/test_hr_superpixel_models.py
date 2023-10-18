"""Test for SuperPixel Transformer Head."""
import unittest
import numpy as np
import torch
import torch.nn as nn
import timm.models.layers as layers

import hr_superpixel_models as sp
import hr_superpixel_prev_models as sp_prev


class HRSuperPixelTransformerTest(unittest.TestCase):
    def test_forward(self):
        b, h, w = 1, 224, 224
        for tokenization_method in ["sp_cross", "patch"]:
            x = torch.zeros((b, 3, h, w))
            module = sp_prev.hr_superformer_tiny_224(sp_method=tokenization_method)
            res = module(x)
            self.assertTupleEqual(res["cls"].shape, (b, 1000))

            # tokenization_info = module.tokenization_info
            # self.assertIsInstance(tokenization_info, (type(None), dict))

    def test_no_weight_decay(self):
        module = sp_prev.hr_superformer_tiny_224_posembeds2()
        no_weight_decay = module.no_weight_decay()
        for name in no_weight_decay:
            self.assertTrue(name.endswith("pos_embed"))

    def test_sp_sizes(self):
        module = sp_prev.hr_superformer_tiny_224_posembeds2()
        self.assertEqual(module.stages[0].patch_embed.superpixel_shape, [14, 14])
        self.assertEqual(module.stages[1].patch_embed.superpixel_shape, [7, 7])

    def test_sp_skip_connection(self):
        module = sp_prev.hr_superformer_tiny_spres_224_posembeds2()
        self.assertIsInstance(module.stages[0].sp_downsample, nn.Identity)
        self.assertEqual(module.stages[0].sp_stride, 1)
        self.assertIsInstance(module.stages[1].sp_downsample, nn.Sequential)
        self.assertEqual(module.stages[1].sp_stride, 2)

    def test_patch_network(self):
        module = sp_prev.hr_superformer_stems8_patch7_7_tiny_224()
        self.assertIsInstance(module.stages[0].patch_embed, nn.AvgPool2d)
        self.assertIsInstance(module.stages[1].patch_embed, nn.AvgPool2d)

    @unittest.skip("No seg now")
    def test_segclassifier(self):
        seg_num_classes = 21
        module = sp_prev.hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw_segclassifier(
            seg_num_classes=seg_num_classes
        )
        self.assertEqual(module.stages[-1].seg_block_idx, 9)
        module = sp_prev.hr_superformer_stems8_hyper13_spsize7_7_tiny_spres_224_posembeddw_segclassifierf3(
            seg_num_classes=seg_num_classes
        )
        self.assertEqual(module.stages[-1].seg_block_idx, 9 - 3)

        b, h, w = 1, 224, 224
        x = torch.zeros((b, 3, h, w))
        res = module(x, generate_seg=True, seg_stride=1)
        self.assertTupleEqual(res["seg"].shape, (b, 21, h, w))

    def test_prenorm_pixel(self):
        module = (
            sp_prev.hr_superformer_stems8_hyper13_spsize7_7_pixelqkvidentity_prenormpixel_tiny_spres_224_nofinal_posembeddw()
        )
        self.assertIsInstance(module.stages[0].downsample, nn.Identity)
        self.assertIsInstance(module.stages[1].downsample, layers.LayerNorm2d)

    @unittest.skip("Not implemented")
    def test_sp_sp_position_embedding_stride(self):
        raise NotImplementedError()

    def test_pixel_refine(self):
        module = (
            sp_prev.hr_superformer_stems8_hyper13_pixelrefinek3_spsize7_7_pixelqkvidentity_tiny_spres_224_nofinal_posembeddw()
        )
        self.assertIsInstance(module.stages[0].pixel_refine, sp.PixelRefineSepConv)
        self.assertIsInstance(module.stages[1].pixel_refine, nn.Identity)

    @unittest.skip("Not implemented")
    def test_fix_drop_path_rate(self):
        module = (
            sp.hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate()
        )

    def test_class_token_and_pos_embed(self):
        module = (
            sp.hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_clstokencor()
        )
        self.assertIsNone(module.stages[0].cls_token)
        self.assertIsNotNone(module.stages[1].cls_token)
        self.assertIsNone(module.stages[0].pos_embed)
        self.assertIsNone(module.stages[1].pos_embed)
        self.assertIsInstance(module.fc_norm, nn.Identity)
        self.assertIsInstance(module.norm, nn.LayerNorm)

        module = (
            sp.hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery()
        )
        self.assertIsNone(module.stages[0].cls_token)
        self.assertIsNone(module.stages[1].cls_token)
        self.assertIsNotNone(module.stages[0].pos_embed)
        self.assertIsNotNone(module.stages[1].pos_embed)
        self.assertIsInstance(module.fc_norm, nn.LayerNorm)
        self.assertIsInstance(module.norm, nn.Identity)

    def test_sp_ls_init(self):
        module = (
            sp.hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery()
        )
        for stage in module.stages:
            num_blocks = len(stage.patch_embed.blocks)
            for i, block in enumerate(stage.patch_embed.blocks):
                self.assertIsInstance(block.sp_ls1, sp.LayerScale2d)
                if i == num_blocks - 1:
                    self.assertIsInstance(block.pixel_ls1, nn.Identity)
                else:
                    self.assertIsInstance(block.sp_ls1, sp.LayerScale2d)

        module = (
            sp.hr_superformer_stemp8_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_small_224_ls_1e_5_fixdroppathrate_posembedevery()
        )

        for stage in module.stages:
            for block in stage.patch_embed.blocks:
                self.assertIsInstance(block.sp_ls1, nn.Identity)
                self.assertIsInstance(block.pixel_ls1, nn.Identity)

    def test_use_pixel_similarities(self):
        module = (
            sp.hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery()
        )

        for stage in module.stages:
            self.assertTrue(stage.use_pixel_similarities)
            # self.assertIsInstance(stage.pixel_ls1, nn.Identity)

    def test_pixelify(self):
        module = (
            sp.hr_superformer_stemp8h6_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery()
        )
        self.assertTrue(module.stages[0].patch_embed.blocks[0]._pixelify)
        self.assertFalse(module.stages[0].patch_embed.blocks[1]._pixelify)
        self.assertTrue(module.stages[1].patch_embed.blocks[0]._pixelify)
        self.assertFalse(module.stages[1].patch_embed.blocks[1]._pixelify)

        module = (
            sp.hr_superformer_stemp8h6_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_middlepixel_small_224_ls_1e_5_fixdroppathrate_posembedevery()
        )

        self.assertTrue(module.stages[0].patch_embed.blocks[0]._pixelify)
        self.assertTrue(module.stages[0].patch_embed.blocks[1]._pixelify)
        self.assertTrue(module.stages[1].patch_embed.blocks[0]._pixelify)
        self.assertFalse(module.stages[1].patch_embed.blocks[1]._pixelify)

        module = (
            sp_prev.hr_superformer_stemp8h6_spsize7_7_pixelqkvidentity_fixpixel_spres_nofinal_posembeddw_ls5_small_224_ls_1e_5_fixdroppathrate_posembedevery()
        )
        self.assertFalse(module.stages[0].patch_embed.blocks[0]._pixelify)
        self.assertFalse(module.stages[0].patch_embed.blocks[1]._pixelify)
        self.assertFalse(module.stages[1].patch_embed.blocks[0]._pixelify)
        self.assertFalse(module.stages[1].patch_embed.blocks[1]._pixelify)

    def test_merge_multihead_similarities(self):
        module = (
            sp.hr_superformer_stemp8h6_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_small_224_ls_1e_5_fixdroppathrate_posembedevery()
        )

        for stage in module.stages:
            self.assertFalse(stage.merge_multihead_similarities)

    def test_use_middle_pixel_features(self):
        b, h, w = 1, 224, 224
        x = torch.zeros((b, 3, h, w))
        module = (
            sp.hr_superformer_stemp8h6_spsize7_7_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_psim_middlepixel_small_224_ls_1e_5_fixdroppathrate_posembedevery()
        )
        for stage in module.stages:
            self.assertTrue(stage.use_pixel_similarities)
            self.assertFalse(stage.merge_multihead_similarities)
        res = module(x)
        self.assertTupleEqual(res["cls"].shape, (b, 1000))

    def test_model(self):
        module = sp.hr_superformer_stemp8h6_spsize14_14depth2_10_head6cor_pixelqkvidentity_spres_nofinal_posembeddw_ls5_small_448_ls_1e_5_fixdroppathrate_posembedevery(
            img_size=448
        )
        self.assertEqual(len(module.stages[0].blocks), 2)
        self.assertEqual(len(module.stages[1].blocks), 10)
        self.assertEqual(module.stem.conv_layers[0][0].kernel_size, (8, 8))
        self.assertEqual(module.stem.conv_layers[0][0].stride, (8, 8))
        self.assertEqual(module.use_middle_pixel_features, False)
        for stage in module.stages:
            self.assertEqual(stage.use_pixel_similarities, False)
            self.assertEqual(stage.merge_multihead_similarities, False)
            self.assertEqual(stage.patch_embed.num_heads, 6)
            self.assertSequenceEqual(stage.patch_embed.superpixel_shape, (14, 14))


if __name__ == "__main__":
    sp.SKIP_CONFIRM = True
    unittest.main()
