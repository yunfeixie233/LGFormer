import torch
import torch.nn as nn
from unittest import TestCase
from functools import partial
from mmseg.models import build_loss

class TestIgnoreIndex(TestCase):
    def test_ignore_index(self):
        # Construct CrossEntropyLoss object with ignore_index=0
        seg_criterion = build_loss(dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        seg_criterion = partial(seg_criterion, ignore_index=0)

        # Construct simple prediction and target tensors
        # Prediction tensor should have shape (N, C, H, W), in this case (1, 2, 2, 2)
        pred = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]])

        # Target tensor should have shape (N, H, W), in this case (1, 2, 2)
        target = torch.tensor([[[0, 1], [1, 0]]])  # 0s are indices to be ignored

        # Compute loss
        loss = seg_criterion(pred, target)

        # Check if loss matches expected value.
        # Since we are ignoring the indices with 0 in the target tensor, we would expect the loss to only consider
        # the indices where the target is 1. Therefore, our expected loss is -log(exp(0.6)/(exp(0.2)+exp(0.6)))/2
        expected_loss = -torch.log(torch.exp(torch.tensor(0.6)) / (torch.exp(torch.tensor(0.2)) + torch.exp(torch.tensor(0.6)))) / 2
        self.assertTrue(torch.isclose(loss, expected_loss))

