import unittest

import numpy as np

from adaptive3dgs import TargetRGBDObservation, ValidationError, build_source_hidden_evidence


def observation(name: str, depth: np.ndarray, rgb_value: int) -> TargetRGBDObservation:
    height, width = depth.shape
    return TargetRGBDObservation(
        observation_id=name,
        rgb_uint8=np.full((height, width, 3), rgb_value, dtype=np.uint8),
        depth_z_float32=depth.astype(np.float32),
        intrinsics_3x3_float64=np.eye(3, dtype=np.float64),
        world_to_camera_4x4_float64=np.eye(4, dtype=np.float64),
    )


class HiddenSupervisionTests(unittest.TestCase):
    def test_nearest_positive_hidden_surface_is_selected(self) -> None:
        source = np.full((2, 3), 2.0, dtype=np.float32)
        farther = observation("farther", np.full((2, 3), 5.0, dtype=np.float32), 50)
        nearer = observation("nearer", np.full((2, 3), 4.0, dtype=np.float32), 200)
        evidence = build_source_hidden_evidence(
            source,
            np.eye(3, dtype=np.float64),
            np.eye(4, dtype=np.float64),
            (farther, nearer),
        )
        np.testing.assert_array_equal(evidence.positive_mask_uint8, np.ones((2, 3), dtype=np.uint8))
        np.testing.assert_allclose(evidence.depth_z_float32, 4.0)
        np.testing.assert_array_equal(evidence.rgb_uint8, np.full((2, 3, 3), 200, dtype=np.uint8))
        np.testing.assert_array_equal(evidence.observation_count_uint16, np.full((2, 3), 2, dtype=np.uint16))
        np.testing.assert_allclose(evidence.depth_spread_float32, 1.0)

    def test_front_surface_is_not_a_hidden_positive(self) -> None:
        source = np.full((2, 2), 2.0, dtype=np.float32)
        visible = observation("visible", np.full((2, 2), 2.005, dtype=np.float32), 100)
        evidence = build_source_hidden_evidence(
            source,
            np.eye(3, dtype=np.float64),
            np.eye(4, dtype=np.float64),
            (visible,),
            relative_behind_margin=0.01,
            absolute_behind_margin=0.01,
        )
        self.assertFalse(np.any(evidence.positive_mask_uint8))
        self.assertTrue(np.isnan(evidence.depth_z_float32).all())
        self.assertTrue(np.all(evidence.source_view_coverage_uint8 == 1))

    def test_unknown_is_not_encoded_as_negative_support(self) -> None:
        source = np.full((2, 2), 2.0, dtype=np.float32)
        depth = np.full((2, 2), np.nan, dtype=np.float32)
        evidence = build_source_hidden_evidence(
            source,
            np.eye(3, dtype=np.float64),
            np.eye(4, dtype=np.float64),
            (observation("empty", depth, 0),),
        )
        self.assertFalse(np.any(evidence.positive_mask_uint8))
        self.assertFalse(np.any(evidence.source_view_coverage_uint8))

    def test_float32_rounding_onto_relative_boundary_becomes_unknown(self) -> None:
        source = np.full((1, 1), 2.0, dtype=np.float32)
        target = observation("rounding-boundary", np.full((1, 1), 2.0, dtype=np.float32), 123)
        target_w2c = np.eye(4, dtype=np.float64)
        target_w2c[2, 3] = -0.02000000001
        object.__setattr__(target, "world_to_camera_4x4_float64", target_w2c)
        evidence = build_source_hidden_evidence(
            source,
            np.eye(3, dtype=np.float64),
            np.eye(4, dtype=np.float64),
            (target,),
            relative_behind_margin=0.01,
            absolute_behind_margin=0.01,
        )
        self.assertEqual(int(evidence.positive_mask_uint8[0, 0]), 0)
        self.assertTrue(np.isnan(evidence.depth_z_float32[0, 0]))
        self.assertEqual(int(evidence.observation_count_uint16[0, 0]), 0)

    def test_bad_rgb_shape_is_rejected(self) -> None:
        bad = observation("bad", np.ones((2, 2), dtype=np.float32), 0)
        object.__setattr__(bad, "rgb_uint8", np.zeros((2, 2), dtype=np.uint8))
        with self.assertRaisesRegex(ValidationError, "RGB"):
            build_source_hidden_evidence(
                np.ones((2, 2), dtype=np.float32),
                np.eye(3, dtype=np.float64),
                np.eye(4, dtype=np.float64),
                (bad,),
            )


if __name__ == "__main__":
    unittest.main()
