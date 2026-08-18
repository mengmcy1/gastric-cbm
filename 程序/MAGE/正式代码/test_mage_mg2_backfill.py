#!/usr/bin/env python3
"""Unit tests for the MG2 coordinate backfill, loss masking and cache binding.

Coverage required by the frozen MG2 protocol (2026-08-18):

1. backfilled mass sums to 1 (error < 1e-6);
2. backfilled mass lies entirely inside the actual crop_box;
3. horizontal-flip consistency (backfilling the flipped attention with the
   flipped crop equals the mirror of the unflipped backfill);
4. synthetic round-trips (crop == full image reproduces the teacher attention;
   a known sub-rectangle is compared against hand-computed values);
5. student attention sums to 1;
6. no GAP bypass (the classifier output numerically equals dropout+Linear of
   the attention-weighted sum, and logits carry gradient through attention);
7. cancer/non-cancer attention-loss masking (non-cancer contributes nothing);
8. teacher-cache SHA binding and per-image alignment validation.

Runnable both as ``pytest test_mage_mg2_backfill.py`` and directly as
``python test_mage_mg2_backfill.py`` (executes every ``test_*`` in order).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_mage_mg1b_attention_teacher import (  # noqa: E402
    GRID_SIZE,
    AttentionPoolingTeacher,
    cell_overlap_map,
)
from train_mage_mg2_student import (  # noqa: E402
    ALPHA,
    CACHE_FORMAT,
    FROZEN_TEACHER_SHA256,
    IMAGE_SIZE,
    TAU,
    backfill_attention_to_full,
    compute_mg2_losses,
    flip_box_horizontal,
    validate_teacher_cache,
)

G = GRID_SIZE


def _random_attention(rng: np.random.Generator) -> np.ndarray:
    """Return one random normalized [7,7] attention map."""
    values = rng.random((G, G)) + 1e-3
    return values / values.sum()


def test_backfill_mass_sum_to_one() -> None:
    """回填质量总和=1（误差<1e-6），覆盖随机注意力与多种crop矩形。"""
    rng = np.random.default_rng(0)
    crops = [
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([0.1, 0.2, 0.9, 0.8]),
        np.array([0.5, 0.0, 1.0, 1.0]),
        np.array([0.33, 0.17, 0.61, 0.49]),
        np.array([0.0, 0.0, 0.05, 0.05]),
    ]
    for crop in crops:
        for _ in range(5):
            attention = _random_attention(rng)
            target = backfill_attention_to_full(attention, crop)
            assert abs(float(target.sum()) - 1.0) < 1e-6, (
                f"crop={crop} 回填总和={target.sum()}"
            )


def test_backfill_mass_inside_crop() -> None:
    """回填质量全部位于crop_box内：与crop零相交的学生cell质量必须为0。"""
    rng = np.random.default_rng(1)
    for _ in range(20):
        x1, x2 = sorted(rng.random(2) * 0.8)
        y1, y2 = sorted(rng.random(2) * 0.8)
        crop = np.array([x1, y1, x2 + 0.1, y2 + 0.1])
        attention = _random_attention(rng)
        target = backfill_attention_to_full(attention, crop)
        outside_cells = cell_overlap_map(crop) <= 0.0
        assert float(target[outside_cells].sum()) == 0.0, (
            f"crop={crop} 在零相交cell上存在质量 {target[outside_cells].sum()}"
        )


def test_backfill_flip_consistency() -> None:
    """水平翻转一致性：backfill(翻转注意力, 翻转crop) == 翻转backfill结果。"""
    rng = np.random.default_rng(2)
    for _ in range(20):
        x1 = rng.random() * 0.6
        y1 = rng.random() * 0.6
        crop = np.array([x1, y1,
                         min(1.0, x1 + 0.2 + rng.random() * 0.3),
                         min(1.0, y1 + 0.2 + rng.random() * 0.3)])
        attention = _random_attention(rng)
        direct = backfill_attention_to_full(attention, crop)
        flipped = backfill_attention_to_full(
            np.fliplr(attention), flip_box_horizontal(crop)
        )
        assert np.allclose(flipped, np.fliplr(direct), atol=1e-6), (
            f"翻转不一致: crop={crop}, 最大差={np.abs(flipped - np.fliplr(direct)).max()}"
        )


def test_backfill_roundtrip_full_crop() -> None:
    """合成round-trip：crop=全图时回填必须等于原教师注意力。"""
    rng = np.random.default_rng(3)
    for _ in range(10):
        attention = _random_attention(rng)
        target = backfill_attention_to_full(
            attention, np.array([0.0, 0.0, 1.0, 1.0])
        )
        assert np.allclose(target, attention, atol=1e-6), (
            f"全图round-trip最大差={np.abs(target - attention).max()}"
        )


def test_backfill_known_subrectangle() -> None:
    """合成用例手算对比：crop=[0.5,0,1,1]右半幅、均匀教师注意力。

    均匀注意力密度为1/0.5=2；学生cell宽1/7。x列0-2与crop零相交→0；
    第3列（x∈[3/7,4/7]）相交宽1/14→每格质量2*(1/7)*(1/14)=1/49；
    第4-6列完全在内→每格2*(1/7)*(1/7)=2/49。7行相同，总和=7*(1+6)/49=1。
    """
    crop = np.array([0.5, 0.0, 1.0, 1.0])
    attention = np.full((G, G), 1.0 / (G * G))
    target = backfill_attention_to_full(attention, crop)
    expected = np.zeros((G, G))
    expected[:, 3] = 1.0 / 49.0
    expected[:, 4:] = 2.0 / 49.0
    assert np.allclose(target, expected, atol=1e-6), (
        f"右半幅手算对比最大差={np.abs(target - expected).max()}"
    )

    # 单点质量：crop=[2/7,3/7,5/7,6/7]与第3象限对齐，教师cell(0,0)位于
    # crop左上角，矩形x∈[14/49,17/49]、y∈[21/49,24/49]；它与学生cell
    # (3,2)（x∈[14/49,21/49]、y∈[21/49,28/49]）完全重合，故全部质量
    # 恰好落回该学生cell，其余为0。
    crop_aligned = np.array([2 / 7, 3 / 7, 5 / 7, 6 / 7])
    onehot = np.zeros((G, G))
    onehot[0, 0] = 1.0
    target_onehot = backfill_attention_to_full(onehot, crop_aligned)
    expected_onehot = np.zeros((G, G))
    expected_onehot[3, 2] = 1.0
    assert np.allclose(target_onehot, expected_onehot, atol=1e-6), (
        f"单点手算对比最大差={np.abs(target_onehot - expected_onehot).max()}"
    )


def test_student_attention_sum_to_one() -> None:
    """学生空间softmax注意力逐图总和=1（误差<1e-6）。"""
    model = AttentionPoolingTeacher(pretrained=False)
    model.eval()
    with torch.inference_mode():
        _, attention = model(torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert attention.shape == (4, 1, G, G)
    sums = attention.sum(dim=(1, 2, 3))
    assert torch.allclose(sums, torch.ones(4), atol=1e-6), f"注意力总和={sums}"


def test_no_gap_bypass() -> None:
    """无GAP旁路：分类logits数值上等于attention加权和经dropout+Linear。

    eval模式下关闭dropout随机性，手工重算 ``classifier(sum(feature*attention))``
    并与forward输出对比；另验证logits对attention存在非零梯度依赖。
    """
    torch.manual_seed(0)
    model = AttentionPoolingTeacher(pretrained=False)
    model.eval()
    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    logits, attention = model(images)
    with torch.inference_mode():
        feature = model.features(images)
        manual_attention = torch.softmax(
            model.attention_head(feature).flatten(1), dim=1
        ).view(-1, 1, G, G)
        pooled = (feature * manual_attention).sum(dim=(2, 3))
        manual_logits = model.classifier(pooled)
    assert torch.allclose(logits, manual_logits, atol=1e-5), (
        f"forward与attention加权路径不一致，最大差={abs(logits - manual_logits).max()}"
    )
    images.requires_grad_(True)
    logits_grad, attention_grad = model(images)
    gradient = torch.autograd.grad(logits_grad.sum(), attention_grad)[0]
    assert float(gradient.abs().sum()) > 0, "logits对attention无梯度依赖"


def test_attention_loss_cancer_mask() -> None:
    """癌/非癌掩码：attention KD只对癌图生效，非癌样本梯度恒为0。"""
    torch.manual_seed(1)
    batch = 3
    attention = torch.softmax(torch.randn(batch, 1, G, G), dim=-1)
    attention = attention.requires_grad_(True)
    targets = torch.zeros(batch, G * G)
    targets[0] = torch.softmax(torch.randn(G * G), dim=0)
    targets[1] = torch.softmax(torch.randn(G * G), dim=0)
    has_target = torch.tensor([True, True, False])
    logits = torch.randn(batch, 2)
    labels = torch.tensor([1, 1, 0])
    teacher_logits = torch.randn(batch, 2)
    losses = compute_mg2_losses(
        "C", logits, attention, labels, teacher_logits, targets, has_target,
        beta=1.0, label_smoothing=0.1,
    )
    # 手工均值：仅两张癌图的KL(target||attention)。
    manual = []
    flat = attention.detach().flatten(1).clamp_min(1e-8)
    for index in (0, 1):
        target = targets[index]
        positive = target > 0
        manual.append(float(
            (target[positive] * (target[positive].log() - flat[index][positive].log())).sum()
        ))
    assert abs(float(losses["kd_attention"].detach()) - float(np.mean(manual))) < 1e-6
    losses["total"].backward()
    gradient = attention.grad.flatten(1)
    assert float(gradient[2].abs().sum()) == 0.0, "非癌样本attention获得了梯度"
    assert float(gradient[:2].abs().sum()) > 0, "癌样本attention梯度异常为0"


def _synthetic_cache(shas: list[str]) -> dict:
    """Build one minimal valid cache object for validation tests."""
    entries = {}
    for sha in shas:
        entries[sha] = {
            state: {
                "logits": torch.zeros(2),
                "attention": torch.full((G * G,), 1.0 / (G * G)),
            }
            for state in ("flip0", "flip1")
        }
    return {
        "format": CACHE_FORMAT,
        "teacher_checkpoint_sha256": FROZEN_TEACHER_SHA256,
        "manifest_sha256": "a" * 64,
        "grid_size": G,
        "image_size": IMAGE_SIZE,
        "debug": False,
        "order": list(shas),
        "entries": entries,
    }


def test_cache_sha_binding_and_alignment() -> None:
    """教师缓存SHA绑定与逐图对齐：正常通过，篡改即快速失败。"""
    shas = ["sha_a", "sha_b", "sha_c"]
    frame = pd.DataFrame({"sha256": shas})
    cache = _synthetic_cache(shas)
    validate_teacher_cache(cache, frame, "a" * 64, False)

    def expect_failure(mutate, message: str) -> None:
        broken = _synthetic_cache(shas)
        mutate(broken)
        try:
            validate_teacher_cache(broken, frame, "a" * 64, False)
        except ValueError:
            return
        raise AssertionError(f"未检测到篡改: {message}")

    expect_failure(
        lambda c: c.update(teacher_checkpoint_sha256="0" * 64), "教师SHA篡改"
    )
    expect_failure(lambda c: c.update(manifest_sha256="b" * 64), "manifest SHA篡改")
    expect_failure(lambda c: c.update(order=list(reversed(shas))), "逐图顺序重排")
    expect_failure(lambda c: c["entries"].pop("sha_b"), "缺失单图条目")
    expect_failure(lambda c: c.update(debug=True), "debug标记不一致")
    expect_failure(
        lambda c: c["entries"]["sha_a"]["flip0"].update(
            attention=torch.zeros(G * G)
        ),
        "注意力总和不为1",
    )
    try:
        validate_teacher_cache(_synthetic_cache(shas), frame, "c" * 64, False)
    except ValueError:
        pass
    else:
        raise AssertionError("未检测到manifest实参不一致")


def run_all() -> None:
    """Execute every ``test_*`` function in definition order and report."""
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"MG2单元测试全部通过: {len(tests)}项")


if __name__ == "__main__":
    run_all()
