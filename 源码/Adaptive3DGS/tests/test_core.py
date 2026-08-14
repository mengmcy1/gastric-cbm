import unittest

import numpy as np

from adaptive3dgs import (
    CanonicalLayerPatch,
    Provenance,
    ProviderRegistry,
    ProviderResult,
    SupportType,
    ValidationError,
    validate_result,
)


def hidden_patch(*, provenance: Provenance = Provenance.PREDICTED_HIDDEN_GEOMETRY) -> CanonicalLayerPatch:
    height, width = 3, 4
    return CanonicalLayerPatch(
        patch_id="patch-1",
        support_type=SupportType.OCCLUSION_HIDDEN,
        rgb_uint8=np.zeros((height, width, 3), dtype=np.uint8),
        depth_z_float32=np.full((height, width), 2.0, dtype=np.float32),
        alpha_float32=np.full((height, width), 0.8, dtype=np.float32),
        valid_mask_uint8=np.ones((height, width), dtype=np.uint8),
        provenance_uint8=np.full((height, width), provenance, dtype=np.uint8),
        geometry_confidence_float32=np.full((height, width), 0.6, dtype=np.float32),
        appearance_confidence_float32=np.full((height, width), 0.5, dtype=np.float32),
        intrinsics_3x3_float64=np.eye(3, dtype=np.float64),
        world_to_camera_4x4_float64=np.eye(4, dtype=np.float64),
    )


class CoreTests(unittest.TestCase):
    def test_valid_hidden_result(self) -> None:
        report = validate_result(
            ProviderResult(provider_id="test", patches=[hidden_patch()]),
            expected_support=SupportType.OCCLUSION_HIDDEN,
        )
        self.assertEqual(report.patch_count, 1)
        self.assertEqual(report.valid_pixel_count, 12)

    def test_alpha_does_not_substitute_confidence(self) -> None:
        patch = hidden_patch()
        object.__setattr__(patch, "geometry_confidence_float32", np.full((3, 4), np.nan, dtype=np.float32))
        with self.assertRaisesRegex(ValidationError, "geometry_confidence"):
            validate_result(
                ProviderResult(provider_id="test", patches=[patch]),
                expected_support=SupportType.OCCLUSION_HIDDEN,
            )

    def test_synthetic_oracle_rejected_in_deployment(self) -> None:
        with self.assertRaisesRegex(ValidationError, "synthetic oracle"):
            validate_result(
                ProviderResult(provider_id="oracle", patches=[hidden_patch(provenance=Provenance.SYNTHETIC_ORACLE)]),
                expected_support=SupportType.OCCLUSION_HIDDEN,
            )

    def test_invalid_pixel_does_not_hide_bad_alpha(self) -> None:
        patch = hidden_patch()
        patch.valid_mask_uint8[0, 0] = 0
        patch.alpha_float32[0, 0] = np.nan
        with self.assertRaisesRegex(ValidationError, "alpha"):
            validate_result(
                ProviderResult(provider_id="test", patches=[patch]),
                expected_support=SupportType.OCCLUSION_HIDDEN,
            )

    def test_provider_cannot_cross_support_type(self) -> None:
        with self.assertRaisesRegex(ValidationError, "outside outside_source_fov"):
            validate_result(
                ProviderResult(provider_id="test", patches=[hidden_patch()]),
                expected_support=SupportType.OUTSIDE_SOURCE_FOV,
            )

    def test_registry_refuses_duplicate(self) -> None:
        registry: ProviderRegistry[object] = ProviderRegistry()
        registry.register("provider", object)
        with self.assertRaises(KeyError):
            registry.register("provider", object)
        self.assertEqual(registry.available(), ("provider",))


if __name__ == "__main__":
    unittest.main()
