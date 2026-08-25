#!/usr/bin/env python3
"""CD1 Gray训练入口的参数差异和产品核验测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import yaml

from train_cd1_gray import (
    CD1_GRAY_TRAIN_ARGS,
    FORMAL_EPOCHS,
    FORMAL_PATIENCE,
    FROZEN_TRAIN_ARGS,
    IMAGE_SIZE,
    expected_run_name,
    verify_args_yaml,
    verify_products,
)


def main() -> None:
    """验证只允许颜色参数变化以及debug/formal产品边界。"""
    changed = {
        key for key in CD1_GRAY_TRAIN_ARGS
        if CD1_GRAY_TRAIN_ARGS[key] != FROZEN_TRAIN_ARGS[key]
    }
    assert changed == {"hsv_h", "hsv_s"}
    assert CD1_GRAY_TRAIN_ARGS["hsv_h"] == 0.0
    assert CD1_GRAY_TRAIN_ARGS["hsv_s"] == 0.0
    assert CD1_GRAY_TRAIN_ARGS["hsv_v"] == FROZEN_TRAIN_ARGS["hsv_v"] == 0.4
    assert expected_run_name(42, False) == "cd1_gray_yolo26s_640_seed42"
    assert expected_run_name(42, True).endswith("_debug")

    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary)
        expected = {
            **CD1_GRAY_TRAIN_ARGS,
            "imgsz": IMAGE_SIZE,
            "seed": 42,
            "epochs": 1,
            "patience": FORMAL_PATIENCE,
            "fraction": 1.0,
            "nms": False,
            "max_det": 300,
            "data": str((Path(__file__).resolve().parents[3] / "数据整理记录/CADe_CD1_Gray_20260825/data.yaml").resolve()),
        }
        (run / "args.yaml").write_text(yaml.safe_dump(expected), encoding="utf-8")
        values = verify_args_yaml(run / "args.yaml", 42, True)
        assert values["epochs"] == 1

        changed_args = dict(expected)
        changed_args["optimizer"] = "auto"
        (run / "args.yaml").write_text(yaml.safe_dump(changed_args), encoding="utf-8")
        try:
            verify_args_yaml(run / "args.yaml", 42, True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("训练参数漂移没有被拒绝")

        (run / "weights").mkdir()
        (run / "weights/best.pt").write_bytes(b"best")
        (run / "weights/last.pt").write_bytes(b"last")
        pd.DataFrame({"metrics/mAP50-95(B)": [0.12]}).to_csv(run / "results.csv", index=False)
        products = verify_products(run, True)
        assert products["epochs_completed"] == 1
        assert products["best_validation_map50_95"] == 0.12
        assert products["checkpoint_selection"] == "ultralytics_validation_fitness_map50_95"

    assert FORMAL_EPOCHS == 100
    print("CD1 train entry tests: 12/12 passed")


if __name__ == "__main__":
    main()
