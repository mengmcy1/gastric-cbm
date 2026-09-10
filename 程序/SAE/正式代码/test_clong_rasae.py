"""RA-SAE原型数学与现有C-long损失/干预接口检查；仅CPU，无文件输出。"""

import io
import unittest
from types import SimpleNamespace

import torch

from clong_rasae_core import ArchetypalMatryoshkaSAE, calibrate_initial_encoder
from clong_rpc_core import intervention_block, remove_feature_group
from clong_s2c_matryoshka import joint_layer_losses


class RASAEChecks(unittest.TestCase):
    """验证凸组合、投影、配对初始化、梯度、保存恢复和干预数学。"""

    def setUp(self):
        torch.manual_seed(7)
        self.points = torch.randn(7, 8)
        self.center = torch.randn(8)

    def model(self, constrained=True, delta=0.2, seed=8):
        return ArchetypalMatryoshkaSAE(self.points, self.center, 12, (2, 4, 8),
                                       delta, constrained, seed)

    def test_convex_combination_even_when_dictionary_wider_than_points(self):
        model = self.model(delta=0)
        weights = model.mixture_logits.softmax(-1)
        torch.testing.assert_close(weights.sum(1), torch.ones(12))
        torch.testing.assert_close(model.decoder_weight, weights @ model.points)
        self.assertTrue(bool((weights >= 0).all()))
        self.assertEqual(torch.unique(model.decoder_weight.detach(), dim=0).shape[0], 12)

    def test_relaxation_projection_including_zero_radius(self):
        for delta in (0, 0.2):
            model = self.model(delta=delta)
            with torch.no_grad():
                model.relaxation.fill_(10)
            model.normalize_decoder()
            self.assertTrue(bool((model.relaxation.norm(dim=1) <= delta + 1e-6).all()))
            self.assertTrue(bool(torch.isfinite(model.decoder_weight).all()))

    def test_paired_arms_have_identical_initial_outputs(self):
        free, ra = self.model(False), self.model(True)
        x = torch.randn(2, 49, 8)
        for k in ra.k_list:
            torch.testing.assert_close(free(x)[k][0], ra(x)[k][0])
        before = free.decoder_weight.detach().clone()
        free.normalize_decoder()
        torch.testing.assert_close(before, free.decoder_weight)

    def test_seed_changes_initial_dictionary(self):
        self.assertFalse(torch.equal(self.model(seed=8).decoder_weight,
                                     self.model(seed=9).decoder_weight))

    def test_encoder_initialization_accounts_for_atom_length(self):
        model = self.model()
        atom = model.decoder_weight
        torch.testing.assert_close((atom * model.encoder.weight).sum(1), torch.ones(12))

    def test_unique_initialization_requires_enough_points(self):
        with self.assertRaisesRegex(ValueError, '代表点'):
            ArchetypalMatryoshkaSAE(self.points, self.center, 12, (2, 4), .2, True, 8, 'unique')

    def test_unique_initialization_does_not_reuse_dominant_points(self):
        model = ArchetypalMatryoshkaSAE(torch.eye(16), torch.zeros(16), 12, (2, 4), .2, True, 8, 'unique')
        self.assertEqual(len(model.mixture_logits.argmax(1).unique()), 12)

    def test_shared_scale_matches_direct_reconstruction_and_preserves_support(self):
        model = self.model()
        x = torch.randn(4, 49, 8)
        attention = torch.softmax(torch.randn(4, 49), 1)
        weights = torch.tensor([.1, .2, .3, .4])
        old_hidden = {k: h.detach().clone() for k, h in model.encode(x).items()}
        old_dictionary = model.decoder_weight.detach().clone()
        result = calibrate_initial_encoder(model, x, attention, weights, 2)
        for k, (rebuilt, hidden) in model(x).items():
            torch.testing.assert_close(hidden, old_hidden[k] * result['scale'])
            position_weights = .5 + .5 * 49 * attention
            mse = (((rebuilt - x).square().mean(2) * position_weights).mean(1) * weights).sum()
            self.assertAlmostEqual(float(mse.detach()), result['per_k'][str(k)]['after'], places=5)
        torch.testing.assert_close(model.decoder_weight, old_dictionary)
        self.assertLessEqual(sum(v['after'] for v in result['per_k'].values()),
                             sum(v['before'] for v in result['per_k'].values()) + 1e-6)

    def test_nested_hidden(self):
        model = self.model()
        hidden = model.encode(torch.randn(2, 49, 8))
        for small, large in ((2, 4), (4, 8)):
            self.assertTrue(bool((~hidden[small].gt(0) | hidden[large].gt(0)).all()))
            self.assertTrue(bool((hidden[small].gt(0).sum(-1) <= small).all()))

    def test_existing_joint_loss_backpropagates_and_center_stays_fixed(self):
        model = self.model()
        head = torch.nn.Conv2d(8, 1, 1).requires_grad_(False)
        x = torch.randn(2, 49, 8)
        a = torch.softmax(head(x.transpose(1, 2).reshape(2, 8, 7, 7)).flatten(1), 1)
        pool = (x * a.unsqueeze(-1)).sum(1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        totals, _ = joint_layer_losses(model, x, pool, a, SimpleNamespace(attention_head=head),
                                      torch.randn(8), 1.0, 0.5)
        torch.stack(list(totals.values())).mean().backward()
        for parameter in (model.mixture_logits, model.relaxation, model.encoder.weight):
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        optimizer.step()
        model.normalize_decoder()
        torch.testing.assert_close(model.decoder_bias, self.center)

    def test_eval_and_state_roundtrip(self):
        model = self.model().eval()
        x = torch.randn(2, 8)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = self.model(seed=9).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(model(x)[4][0], restored(x)[4][0])

    def test_raw_decoder_intervention_matches_explicit_features(self):
        model = self.model()
        x = torch.randn(2, 49, 8)
        hidden = model.encode(x, 4)[:, :, :3]
        dictionary = model.decoder_weight[:3]
        aw, mw = torch.randn(8), torch.randn(8)
        result = intervention_block(x, hidden, dictionary, aw, torch.tensor(0.),
                                    mw, torch.tensor(0.), 0.)
        changed = x - hidden[:, :, 0, None] * dictionary[0]
        direct = (torch.softmax(changed @ aw, 1) * (changed @ mw)).sum(1)
        torch.testing.assert_close(direct - result['original_margin'], result['delta_margin'][:, 0])

    def test_group_removal_is_simultaneous_sum_of_member_components(self):
        spatial = torch.randn(2, 49, 8)
        hidden = torch.rand(2, 49, 12)
        decoder = torch.randn(12, 8)
        changed = remove_feature_group(spatial, hidden, decoder, [1, 7, 9])
        expected = spatial - sum(hidden[:, :, j, None] * decoder[j] for j in (1, 7, 9))
        torch.testing.assert_close(changed, expected)


if __name__ == '__main__':
    unittest.main()
