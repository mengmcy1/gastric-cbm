#!/usr/bin/env python3
"""冻结并复验一次RP-A development run实际执行的代码指纹。"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from clong_rpa_artifacts import file_sha256


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PROTOCOL_BUNDLE_SHA = "768da344bfd3d49ca518528bc043a4eb2ef5b1b4f76b449c42e00c7223093280"
_SAE_CODE_FILES = (
    "run_clong_rpa_development.sh", "run_clong_rpa_development_training.sh",
    "clong_rpa_train_development.py", "clong_rpa_prepare_seed.py",
    "clong_rpa_match_development.py", "clong_rpa_development_core.py",
    "clong_rpa_null_fdr.py", "clong_rpa_bootstrap.py",
    "clong_rpa_bootstrap_worker.py", "clong_rpa_bootstrap_coordinator.py",
    "clong_rpa_validate_development.py", "clong_rpa_finalize_development.py",
    "clong_rpa_record_implementation_failure.py", "clong_rpa_artifacts.py",
    "clong_rpa_provenance.py", "benchmark_clong_rpa_matching.py",
    "clong_s2c_core.py", "clong_s2c_matryoshka.py", "clong_sae_discovery.py",
    "clong_s2b_discovery.py", "clong_s2b_core.py",
)
CODE_PATHS = {
    **{name: SCRIPT_DIR / name for name in _SAE_CODE_FILES},
    "train_utils.py": PROJECT_ROOT / "程序/模型训练/正式代码/train_utils.py",
}
CODE_FILES = tuple(CODE_PATHS)


def current_code_sha256() -> dict[str, str]:
    """按显式正式执行清单计算逐文件SHA256。"""
    return {name: file_sha256(CODE_PATHS[name]) for name in CODE_FILES}


def current_git_commit() -> str:
    """读取当前仓库提交，和逐文件SHA分开记录。"""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def build_snapshot() -> dict:
    """构造不含统计结果的运行起点代码快照。"""
    return {
        "schema": "rpa_development_code_snapshot_v1",
        "git_commit": current_git_commit(),
        "protocol_bundle_sha256": PROTOCOL_BUNDLE_SHA,
        "code_file_sha256": current_code_sha256(),
    }


def validate_snapshot(snapshot: dict) -> None:
    """要求Git、bundle和每个执行文件均与启动时快照逐位一致。"""
    required = {"git_commit", "protocol_bundle_sha256", "code_file_sha256"}
    if not required.issubset(snapshot):
        raise RuntimeError("代码快照缺少Git、protocol bundle或code SHA")
    if snapshot["protocol_bundle_sha256"] != PROTOCOL_BUNDLE_SHA:
        raise RuntimeError("代码快照绑定的protocol bundle不一致")
    if snapshot["git_commit"] != current_git_commit():
        raise RuntimeError("运行期间Git commit发生变化")
    observed = snapshot["code_file_sha256"]
    expected = current_code_sha256()
    if set(observed) != set(expected):
        raise RuntimeError("运行代码快照文件集合发生变化")
    mismatched = [name for name in CODE_FILES if observed[name] != expected[name]]
    if mismatched:
        raise RuntimeError(f"运行期间正式代码发生变化: {mismatched}")


def load_and_validate_snapshot(path: Path) -> dict:
    """读取runner启动快照并对当前代码执行完整复验。"""
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    validate_snapshot(snapshot)
    return snapshot


def main() -> None:
    """首次以独占方式创建快照；续跑只允许验证既有快照。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "create":
        if args.output.exists():
            raise FileExistsError(f"代码快照已存在，禁止覆盖: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(build_snapshot(), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"RP-A代码快照已冻结: {args.output}")
    else:
        load_and_validate_snapshot(args.output)
        print(f"RP-A代码快照复验通过: {args.output}")


if __name__ == "__main__":
    main()
