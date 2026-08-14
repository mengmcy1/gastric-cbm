import unittest

import numpy as np

from adaptive3dgs import ProviderContext, SupportType, validate_result
from adaptive3dgs.adapters.flash3d import (
    FLASH3D_SH_C0,
    Flash3DReadOnlyAdapter,
    Flash3DSecondLayerOutput,
    flash3d_render_to_geometry_prediction,
)


class FakeBackend:
    backend_id = "fake-frozen-flash3d"

    def predict_second_layer(self, context: ProviderContext) -> Flash3DSecondLayerOutput:
        layer1 = np.ones((2, 3), dtype=np.float32)
        layer2 = np.full((2, 3), 2.0, dtype=np.float32)
        layer2[0, 0] = 0.5
        features = np.zeros((2, 3, 3), dtype=np.float32)
        features[1, 2] = 0.5 / FLASH3D_SH_C0
        return Flash3DSecondLayerOutput(
            layer1_depth_z_float32=layer1,
            layer2_depth_z_float32=layer2,
            layer2_alpha_float32=np.full((2, 3), 0.75, dtype=np.float32),
            layer2_features_dc_hwc_float32=features,
            intrinsics_3x3_float64=np.eye(3, dtype=np.float64),
            world_to_camera_4x4_float64=np.eye(4, dtype=np.float64),
        )


class Flash3DAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ProviderContext(
            sample_id="sample",
            rgb_uint8=np.zeros((2, 3, 3), dtype=np.uint8),
            intrinsics_3x3_float64=np.eye(3, dtype=np.float64),
            world_to_camera_4x4_float64=np.eye(4, dtype=np.float64),
        )

    def test_second_layer_maps_to_uncalibrated_hidden_clb(self) -> None:
        result = Flash3DReadOnlyAdapter(FakeBackend()).predict(self.context)
        report = validate_result(result, expected_support=SupportType.OCCLUSION_HIDDEN)
        patch = result.patches[0]
        self.assertEqual(report.valid_pixel_count, 5)
        self.assertEqual(result.unsupported_regions, (SupportType.OUTSIDE_SOURCE_FOV.value,))
        self.assertTrue(np.all(patch.geometry_confidence_float32 == 0.0))
        self.assertTrue(np.all(patch.appearance_confidence_float32 == 0.0))
        self.assertEqual(patch.valid_mask_uint8[0, 0], 0)
        self.assertEqual(patch.rgb_uint8[1, 2].tolist(), [255, 255, 255])
        self.assertEqual(patch.metadata["anchor_camera"], "center")
        self.assertFalse(result.diagnostics["confidence_calibrated"])

    def test_native_render_normalization_keeps_alpha_separate_from_confidence(self) -> None:
        prediction = flash3d_render_to_geometry_prediction(
            np.array([[1.0, 0.0], [4.0, 3.0]], dtype=np.float32),
            np.array([[0.5, 0.0], [1.0, 0.75]], dtype=np.float32),
            model_depth_units_per_metric_unit=2.0,
        )
        np.testing.assert_allclose(prediction.depth_z_float32[[0, 1, 1], [0, 0, 1]], [1.0, 2.0, 2.0])
        self.assertTrue(np.isnan(prediction.depth_z_float32[0, 1]))
        self.assertTrue(np.all(prediction.geometry_confidence_float32 == 0.0))
        np.testing.assert_array_equal(
            prediction.support_probability_float32,
            np.array([[0.5, 0.0], [1.0, 0.75]], dtype=np.float32),
        )

    def test_out_of_range_alpha_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "alpha"):
            flash3d_render_to_geometry_prediction(
                np.ones((2, 2), dtype=np.float32),
                np.full((2, 2), 1.1, dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
