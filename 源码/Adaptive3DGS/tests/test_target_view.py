import unittest

import numpy as np
import torch

from adaptive3dgs.models import TargetViewUNet, compute_target_view_loss
from adaptive3dgs.target_view import forward_splat_source_to_target, target_camera_conditioning_in_source


class TargetViewTests(unittest.TestCase):
    def test_identity_forward_splat_preserves_rgbd(self):
        rgb = np.arange(36, dtype=np.float32).reshape(3, 4, 3) / 35
        depth = np.full((3, 4), 2.0, dtype=np.float32)
        k = np.array([[10.0, 0, 1.5], [0, 10.0, 1.0], [0, 0, 1]], dtype=np.float64)
        warped_rgb, warped_depth, valid = forward_splat_source_to_target(
            rgb, depth, k, np.eye(4), k, np.eye(4)
        )
        np.testing.assert_allclose(warped_rgb, rgb)
        np.testing.assert_allclose(warped_depth, depth)
        self.assertTrue(valid.all())

    def test_target_camera_condition_changes_with_pose(self):
        k = np.array([[10.0, 0, 1.5], [0, 10.0, 1.0], [0, 0, 1]], dtype=np.float64)
        first_rays, first_origin = target_camera_conditioning_in_source(3, 4, k, np.eye(4), np.eye(4))
        moved = np.eye(4); moved[0, 3] = -1
        second_rays, second_origin = target_camera_conditioning_in_source(3, 4, k, np.eye(4), moved)
        np.testing.assert_allclose(first_rays, second_rays)
        self.assertFalse(np.allclose(first_origin, second_origin))

    def test_model_outputs_two_independent_support_heads(self):
        model = TargetViewUNet(base_channels=8)
        shape = (1, 3, 32, 48)
        output = model(
            torch.rand(shape), torch.ones(1, 1, 32, 48), torch.ones(1, 1, 32, 48),
            torch.rand(shape), torch.zeros(shape), torch.ones(1, 1, 1, 1) * 2,
        )
        self.assertEqual(tuple(output.rgb.shape), shape)
        self.assertTrue(torch.all(output.depth_z > 0))
        self.assertNotEqual(model.occlusion_support_head.weight.data_ptr(), model.outside_support_head.weight.data_ptr())

    def test_loss_ignores_unclassified_truth(self):
        model = TargetViewUNet(base_channels=8)
        shape = (1, 3, 32, 48)
        output = model(
            torch.rand(shape), torch.ones(1, 1, 32, 48), torch.ones(1, 1, 32, 48),
            torch.rand(shape), torch.zeros(shape), torch.ones(1, 1, 1, 1) * 2,
        )
        rgb = torch.rand(shape); depth = torch.ones(1, 1, 32, 48) * 3
        observed = torch.zeros_like(depth, dtype=torch.bool); observed[:, :, :8] = True
        occluded = torch.zeros_like(depth, dtype=torch.bool); occluded[:, :, 8:16] = True
        outside = torch.zeros_like(depth, dtype=torch.bool); outside[:, :, 16:24] = True
        first = compute_target_view_loss(output, rgb, depth, observed, occluded, outside)
        rgb[:, :, 24:] = 100; depth[:, :, 24:] = -999
        second = compute_target_view_loss(output, rgb, depth, observed, occluded, outside)
        self.assertTrue(torch.equal(first.total, second.total))


if __name__ == "__main__":
    unittest.main()
