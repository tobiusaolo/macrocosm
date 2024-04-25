import pretrain as pt
import torch
import unittest
import constants


class TestValidation(unittest.TestCase):
    def test_compute_wins(self):
        """Test compute wins in a standard case."""

        # Use 3 uids.
        uids = [1, 2, 3]
        # Use 2 batches each.
        losses_per_uid = {1: [4.0, 4.1], 2: [6.0, 6.1], 3: [8.1, 8.2]}
        # Batches are tensors but we only use it for enumeration
        batches = [torch.tensor([0]), torch.tensor([0])]
        # Each uid is at a different block.
        uid_to_block = {1: 1000, 2: 2000, 3: 3000}
        # Each uid is at the same extrinsic index since they are different blocks.
        uid_to_extrinsic_index = {1: 25, 2: 25, 3: 25}

        wins, win_rate = pt.validation.compute_wins(
            uids, losses_per_uid, batches, uid_to_block, uid_to_extrinsic_index
        )

        self.assertEqual(wins, {1: 4, 2: 2, 3: 0})
        self.assertEqual(win_rate, {1: 1.0, 2: 0.5, 3: 0.0})

    def test_compute_wins_adjustment_earlier_block(self):
        """Test compute wins when the adjustment for an earlier block matters."""

        # Use 2 uids.
        uids = [1, 2]
        # Use 4 batches each. 2 has better losses every time, but only once enough to beat the adjustment.
        worse_loss = 4.00
        better_loss = 3.99
        best_loss = 3
        losses_per_uid = {
            1: [worse_loss, worse_loss, worse_loss, worse_loss],
            2: [better_loss, better_loss, better_loss, best_loss],
        }
        # Batches are tensors but we only use it for enumeration
        batches = [
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
        ]
        # Each uid is at a different block. 1 is the earliest block
        uid_to_block = {1: 1000, 2: 2000}
        # Each uid is at the same extrinsic index since they are different blocks.
        uid_to_extrinsic_index = {1: 25, 2: 25}

        wins, win_rate = pt.validation.compute_wins(
            uids, losses_per_uid, batches, uid_to_block, uid_to_extrinsic_index
        )

        self.assertEqual(wins, {1: 3, 2: 1})
        self.assertEqual(win_rate, {1: 0.75, 2: 0.25})

    def test_compute_wins_adjustment_earlier_extrinsic_index(self):
        """Test compute wins when the adjustment for an earlier extrinsic index matters."""

        # Use 2 uids.
        uids = [1, 2]
        # Use 4 batches each. 2 has better losses every time, but only once enough to beat the adjustment.
        worse_loss = 4.00
        better_loss = 3.99
        best_loss = 3
        losses_per_uid = {
            1: [worse_loss, worse_loss, worse_loss, worse_loss],
            2: [better_loss, better_loss, better_loss, best_loss],
        }
        # Batches are tensors but we only use it for enumeration
        batches = [
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
        ]
        # Each uid is at the same block so extrinsic index will break the tie.
        uid_to_block = {1: 1000, 2: 1000}
        uid_to_extrinsic_index = {1: 15, 2: 25}

        wins, win_rate = pt.validation.compute_wins(
            uids, losses_per_uid, batches, uid_to_block, uid_to_extrinsic_index
        )

        self.assertEqual(wins, {1: 3, 2: 1})
        self.assertEqual(win_rate, {1: 0.75, 2: 0.25})
