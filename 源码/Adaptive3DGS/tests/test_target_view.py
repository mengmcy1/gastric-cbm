import unittest

import numpy as np
import torch

from adaptive3dgs.models import SourceFeatureTargetViewNet, TargetViewUNet, compute_target_view_loss
from adaptive3dgs.target_view import (
    forward_splat_source_to_target,
    forward_splat_source_to_target_with_grid,
    source_plane_proxy_grid,
    target_camera_conditioning_in_source,
)


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

    def test_identity_feature_grid_samples_source_pixels(self):
        rgb = np.arange(36, dtype=np.float32).reshape(3, 4, 3) / 35
        depth = np.full((3, 4), 2.0, dtype=np.float32)
        k = np.array([[10.0, 0, 1.5], [0, 10.0, 1.0], [0, 0, 1]], dtype=np.float64)
        _, _, valid, grid = forward_splat_source_to_target_with_grid(rgb, depth, k, np.eye(4), k, np.eye(4))
        sampled = torch.nn.functional.grid_sample(
            torch.from_numpy(rgb).permute(2, 0, 1)[None], torch.from_numpy(grid)[None], align_corners=True
        )[0].permute(1, 2, 0).numpy()
        np.testing.assert_allclose(sampled, rgb, atol=1e-6)
        self.assertTrue(valid.all())

    def test_identity_source_plane_proxy_grid_is_pixel_identity(self):
        height, width = 3, 4
        k = np.array([[10.0, 0, 1.5], [0, 10.0, 1.0], [0, 0, 1]], dtype=np.float64)
        rays, origin = target_camera_conditioning_in_source(height, width, k, np.eye(4), np.eye(4))
        grid, valid = source_plane_proxy_grid(rays, origin, 2.0, k, (width, height))
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        expected = np.stack((2 * xx / (width - 1) - 1, 2 * yy / (height - 1) - 1), axis=-1)
        np.testing.assert_allclose(grid, expected, atol=1e-6)
        self.assertTrue(valid.all())

    def test_source_plane_proxy_grid_marks_backward_intersections_invalid(self):
        rays = np.zeros((2, 3, 3), dtype=np.float32); rays[:, :, 2] = -1
        origin = np.zeros_like(rays)
        k = np.array([[10.0, 0, 1.0], [0, 10.0, 0.5], [0, 0, 1]], dtype=np.float64)
        grid, valid = source_plane_proxy_grid(rays, origin, 2.0, k, (3, 2))
        self.assertFalse(valid.any())
        self.assertTrue(np.isfinite(grid).all())

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

    def test_source_feature_model_conditions_uncovered_pixels_on_source(self):
        torch.manual_seed(3)
        model = SourceFeatureTargetViewNet(base_channels=8).eval()
        shape = (1, 3, 32, 48)
        grid = torch.full((1, 32, 48, 2), -2.0)
        common = (
            grid, torch.zeros(shape), torch.zeros(1, 1, 32, 48), torch.zeros(1, 1, 32, 48),
            torch.rand(shape), torch.zeros(shape), torch.ones(1, 1, 1, 1) * 2,
        )
        with torch.no_grad():
            first = model(torch.zeros(shape), *common).rgb
            second = model(torch.ones(shape), *common).rgb
        self.assertGreater(float(torch.mean(torch.abs(first - second))), 1e-6)


if __name__ == "__main__":
    unittest.main()
