import unittest

import numpy as np

from adaptive3dgs import (
    GeometryPrediction,
    GeometryTarget,
    TargetViewPrediction,
    ValidationError,
    evaluate_occlusion_geometry,
    evaluate_target_view_prediction,
)


def target() -> GeometryTarget:
    return GeometryTarget(
        depth_z_float32=np.full((2, 3), 2.0, dtype=np.float32),
        occlusion_hidden_mask=np.array([[True, True, False], [True, True, False]], dtype=bool),
        outside_source_fov_mask=np.array([[False, False, True], [False, False, True]], dtype=bool),
        observed_from_source_mask=np.zeros((2, 3), dtype=bool),
    )


class EvaluationTests(unittest.TestCase):
    def test_perfect_supported_hidden_geometry(self) -> None:
        prediction = GeometryPrediction(
            depth_z_float32=np.full((2, 3), 2.0, dtype=np.float32),
            support_probability_float32=np.ones((2, 3), dtype=np.float32),
            geometry_confidence_float32=np.ones((2, 3), dtype=np.float32),
        )
        result = evaluate_occlusion_geometry(target(), prediction)
        self.assertEqual(result["hidden_coverage"], 1.0)
        self.assertEqual(result["geometry"]["abs_rel"], 0.0)
        self.assertEqual(result["geometry"]["delta_1_10"], 1.0)
        self.assertEqual(result["confidence"]["ece"], 0.0)
        self.assertEqual(result["confidence"]["selective_aurc_abs_rel"], 0.0)

    def test_support_is_not_confidence(self) -> None:
        prediction = GeometryPrediction(
            depth_z_float32=np.full((2, 3), 4.0, dtype=np.float32),
            support_probability_float32=np.ones((2, 3), dtype=np.float32),
            geometry_confidence_float32=np.zeros((2, 3), dtype=np.float32),
        )
        result = evaluate_occlusion_geometry(target(), prediction)
        self.assertEqual(result["hidden_coverage"], 1.0)
        self.assertEqual(result["geometry"]["abs_rel"], 1.0)
        self.assertEqual(result["geometry"]["delta_1_10"], 0.0)
        self.assertEqual(result["confidence"]["ece"], 0.0)
        self.assertEqual(result["confidence"]["selective_aurc_abs_rel"], 1.0)

    def test_shape_mismatch_rejected(self) -> None:
        prediction = GeometryPrediction(
            depth_z_float32=np.ones((2, 2), dtype=np.float32),
            support_probability_float32=np.ones((2, 2), dtype=np.float32),
            geometry_confidence_float32=np.ones((2, 2), dtype=np.float32),
        )
        with self.assertRaisesRegex(ValidationError, "share one HxW"):
            evaluate_occlusion_geometry(target(), prediction)

    def test_target_view_oracle_is_perfect_in_both_novel_regions(self) -> None:
        truth = target()
        rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        prediction = TargetViewPrediction(
            rgb_uint8=rgb.copy(),
            depth_z_float32=truth.depth_z_float32.copy(),
            support_probability_float32=np.ones((2, 3), dtype=np.float32),
            geometry_confidence_float32=np.ones((2, 3), dtype=np.float32),
            appearance_confidence_float32=np.ones((2, 3), dtype=np.float32),
        )
        result = evaluate_target_view_prediction(truth, rgb, prediction)
        self.assertEqual(result["support_on_excluded_truth_pixels"], 0)
        for name in ("occlusion_hidden", "outside_source_fov"):
            region = result["regions"][name]
            self.assertEqual(region["coverage"], 1.0)
            self.assertEqual(region["geometry"]["abs_rel"], 0.0)
            self.assertEqual(region["appearance"]["mae_0_255"], 0.0)
            self.assertEqual(region["appearance"]["exact_match_fraction"], 1.0)

    def test_target_view_masks_must_be_disjoint(self) -> None:
        truth = target()
        truth.outside_source_fov_mask[0, 0] = True
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        prediction = TargetViewPrediction(
            rgb_uint8=rgb,
            depth_z_float32=truth.depth_z_float32.copy(),
            support_probability_float32=np.ones((2, 3), dtype=np.float32),
            geometry_confidence_float32=np.ones((2, 3), dtype=np.float32),
            appearance_confidence_float32=np.ones((2, 3), dtype=np.float32),
        )
        with self.assertRaisesRegex(ValidationError, "pairwise disjoint"):
            evaluate_target_view_prediction(truth, rgb, prediction)


if __name__ == "__main__":
    unittest.main()
