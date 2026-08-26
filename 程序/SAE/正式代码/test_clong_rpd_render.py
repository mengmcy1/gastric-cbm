#!/usr/bin/env python3
"""RP-D v1渲染数值和资产边界测试。"""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from clong_rpd_render_core import (
    display_response,
    positive_q99,
    render_heatmap,
    render_overlay,
    save_case_assets,
)
from render_clong_rpd_atlas import (
    CODE_ROOT,
    TECHNICAL_FONT_PATH,
    select_debug_anchors,
    technical_font,
)


class RPDRenderTests(unittest.TestCase):
    def test_positive_q99_and_all_zero(self) -> None:
        self.assertEqual(positive_q99(np.zeros(10)), (0.0, "all_zero"))
        value, status = positive_q99(np.arange(1, 101))
        self.assertEqual(status, "ok")
        self.assertGreater(value, 98)

    def test_display_response_uses_external_q99(self) -> None:
        raw = np.arange(49, dtype=np.float32).reshape(7, 7)
        result = display_response(raw, 24)
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[-1, -1], 1)
        self.assertAlmostEqual(float(result[3, 3]), 1.0)

    def test_display_response_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            display_response(np.ones((8, 8)), 1)

    def test_render_sizes_match_original(self) -> None:
        original = Image.new("RGB", (80, 60), "gray")
        response = np.ones((7, 7), dtype=np.float32)
        heatmap = render_heatmap(response, original.size)
        overlay = render_overlay(original, heatmap, response, 0.55)
        self.assertEqual(heatmap.size, original.size)
        self.assertEqual(overlay.size, original.size)

    def test_independent_assets_and_raw_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jpg"
            Image.new("RGB", (100, 80), "white").save(source)
            raw = np.arange(49, dtype=np.float32).reshape(7, 7)
            paths = save_case_assets(source, raw, 48, root / "case/seed42", 64, 0.55, True)
            self.assertTrue(paths["original"].exists())
            self.assertTrue(paths["heatmap"].exists())
            self.assertTrue(paths["overlay"].exists())
            np.testing.assert_array_equal(np.load(paths["raw_7x7"]), raw)

    def test_render_protocol_blind_boundary_is_frozen(self) -> None:
        protocol = json.loads((CODE_ROOT / "rpd_render_protocol_v1.json").read_text())
        self.assertEqual(protocol["status"], "frozen_before_image_rendering_2026-08-25")
        self.assertTrue(protocol["blind_package"]["diagnosis_label_hidden"])
        self.assertTrue(protocol["blind_package"]["source_hidden"])
        self.assertTrue(protocol["render_failure"]["replacement_case_forbidden"])

    def test_debug_selection_covers_heavy_and_light(self) -> None:
        anchors = pd.DataFrame({
            "anchor_id": ["a3", "a2", "a1"],
            "heavy_atlas": [True, True, False],
        })
        selected = select_debug_anchors(anchors, 2)
        self.assertEqual(selected.heavy_atlas.tolist(), [True, False])

    def test_frozen_technical_font_supports_chinese(self) -> None:
        self.assertTrue(TECHNICAL_FONT_PATH.is_file())
        self.assertIs(technical_font(), technical_font())


if __name__ == "__main__":
    unittest.main()
