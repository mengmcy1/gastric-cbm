import unittest

import torch

from adaptive3dgs.models import (
    OcclusionHiddenUNet,
    compute_hidden_geometry_loss,
    positive_verified_negative_logistic_loss,
)


class OcclusionHiddenModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = OcclusionHiddenUNet(base_channels=8)
        self.rgb = torch.rand(2, 3, 32, 48)
        self.base = torch.rand(2, 1, 32, 48) * 4 + 1

    def test_three_heads_have_independent_valid_semantics(self):
        output = self.model(self.rgb, self.base)
        self.assertEqual(tuple(output.delta_ratio.shape), (2, 1, 32, 48))
        self.assertTrue(torch.all(output.delta_ratio > 0))
        self.assertTrue(torch.all(output.hidden_depth_z > self.base))
        self.assertTrue(torch.all((output.support_probability > 0) & (output.support_probability < 1)))
        self.assertTrue(torch.all(output.geometry_uncertainty > 0))
        self.assertTrue(torch.all((output.geometry_confidence > 0) & (output.geometry_confidence < 1)))
        self.assertNotEqual(self.model.support_head.weight.data_ptr(), self.model.uncertainty_head.weight.data_ptr())

    def test_invalid_base_depth_is_rejected(self):
        bad = self.base.clone()
        bad[0, 0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            self.model(self.rgb, bad)

    def test_bad_camera_ray_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "camera_rays_xy"):
            self.model(self.rgb, self.base, torch.zeros(2, 3, 32, 48))

    def test_unknown_truth_values_do_not_affect_geometry_losses(self):
        output = self.model(self.rgb, self.base)
        positive = torch.zeros_like(self.base, dtype=torch.bool)
        positive[:, :, 5:10, 8:15] = True
        source = torch.full_like(self.base, 2.0)
        hidden = torch.full_like(self.base, 3.0)
        first = compute_hidden_geometry_loss(output, source, hidden, positive, class_prior=0.1)
        source[~positive] = float("nan")
        hidden[~positive] = -9999
        second = compute_hidden_geometry_loss(output, source, hidden, positive, class_prior=0.1)
        self.assertTrue(torch.equal(first.relative_depth, second.relative_depth))
        self.assertTrue(torch.equal(first.uncertainty_nll, second.uncertainty_nll))
        self.assertTrue(torch.equal(first.support_pu, second.support_pu))

    def test_nonpositive_hidden_ratio_is_rejected(self):
        output = self.model(self.rgb, self.base)
        positive = torch.zeros_like(self.base, dtype=torch.bool)
        positive[:, :, 0, 0] = True
        truth = torch.ones_like(self.base)
        with self.assertRaisesRegex(ValueError, "strictly behind"):
            compute_hidden_geometry_loss(output, truth, truth, positive, class_prior=0.1)

    def test_small_positive_ratio_has_finite_loss_and_gradient(self):
        output = self.model(self.rgb, self.base)
        positive = torch.zeros_like(self.base, dtype=torch.bool)
        positive[:, :, 3:6, 4:8] = True
        source = torch.full_like(self.base, 2.0)
        hidden = source * 1.01
        loss = compute_hidden_geometry_loss(output, source, hidden, positive, class_prior=0.05)
        self.assertTrue(torch.isfinite(loss.total))
        loss.total.backward()
        self.assertTrue(torch.isfinite(self.model.delta_head.weight.grad).all())

    def test_verified_support_loss_ignores_remaining_unknown_logits(self):
        logits = torch.randn(1, 1, 4, 5)
        positive = torch.zeros_like(logits, dtype=torch.bool)
        negative = torch.zeros_like(logits, dtype=torch.bool)
        positive[:, :, 0, 0] = True
        negative[:, :, 1, 1] = True
        first = positive_verified_negative_logistic_loss(logits, positive, negative)
        changed = logits.clone()
        unknown = ~(positive | negative)
        changed[unknown] = 1000
        second = positive_verified_negative_logistic_loss(changed, positive, negative)
        self.assertTrue(torch.equal(first, second))

    def test_verified_support_masks_must_be_disjoint(self):
        logits = torch.zeros(1, 1, 2, 2)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, :, 0, 0] = True
        with self.assertRaisesRegex(ValueError, "disjoint"):
            positive_verified_negative_logistic_loss(logits, mask, mask)

    def test_missing_verified_negative_keeps_positive_only_loss(self):
        logits = torch.zeros(1, 1, 2, 2)
        positive = torch.zeros_like(logits, dtype=torch.bool)
        positive[:, :, 0, 0] = True
        negative = torch.zeros_like(logits, dtype=torch.bool)
        loss = positive_verified_negative_logistic_loss(logits, positive, negative)
        self.assertAlmostEqual(float(loss), float(torch.log(torch.tensor(2.0))))


if __name__ == "__main__":
    unittest.main()
