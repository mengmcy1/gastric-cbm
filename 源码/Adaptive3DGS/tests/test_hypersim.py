import unittest

import numpy as np

from adaptive3dgs.datasets.hypersim import (
    HYPERSIM_TO_OPENCV,
    calibration_from_hypersim,
    intrinsics_from_hypersim,
    radial_depth_to_z,
    tonemap_hypersim,
)


class HypersimTests(unittest.TestCase):
    def test_standard_intrinsics(self) -> None:
        matrix = np.diag([0.57735029, 0.43301272, -1.0])
        intrinsics = intrinsics_from_hypersim(matrix, 1024, 768)
        self.assertAlmostEqual(intrinsics[0, 0], 886.810, places=2)
        self.assertAlmostEqual(intrinsics[1, 1], 886.810, places=2)
        self.assertAlmostEqual(intrinsics[0, 2], 511.5)
        self.assertAlmostEqual(intrinsics[1, 2], 383.5)

    def test_center_radial_depth_equals_z(self) -> None:
        intrinsics = np.array([[100.0, 0, 1.0], [0, 100.0, 1.0], [0, 0, 1.0]], dtype=np.float64)
        radial = np.full((3, 3), 2.0, dtype=np.float32)
        depth_z = radial_depth_to_z(radial, intrinsics)
        self.assertEqual(depth_z.dtype, np.float32)
        self.assertAlmostEqual(float(depth_z[1, 1]), 2.0)
        self.assertLess(float(depth_z[0, 0]), 2.0)

    def test_tonemap_returns_uint8(self) -> None:
        color = np.full((2, 2, 3), 0.5, dtype=np.float32)
        mapped, scale = tonemap_hypersim(color, np.ones((2, 2), dtype=bool))
        self.assertEqual(mapped.dtype, np.uint8)
        self.assertTrue(np.isfinite(scale))
        self.assertTrue((mapped == mapped[0, 0]).all())

    def test_tilted_projection_is_factored_into_k_and_rotation(self) -> None:
        matrix = np.array(
            [[0.58094111, 0.0, 0.0], [0.0, 0.44704303, 0.0], [0.14044158, -0.03296602, -1.02545373]],
            dtype=np.float64,
        )
        width, height = 1024, 768
        k, rotation = calibration_from_hypersim(matrix, width, height)
        pixel_to_uv = np.array(
            [[2 / width, 0, 1 / width - 1], [0, -2 / height, 1 - 1 / height], [0, 0, 1]],
            dtype=np.float64,
        )
        original_projection = np.linalg.inv(HYPERSIM_TO_OPENCV @ matrix @ pixel_to_uv)
        reconstructed = k @ rotation
        original_projection /= original_projection[2, 2]
        reconstructed /= reconstructed[2, 2]
        np.testing.assert_allclose(reconstructed, original_projection, atol=1e-6)
        self.assertFalse(np.allclose(rotation, np.eye(3)))


if __name__ == "__main__":
    unittest.main()
