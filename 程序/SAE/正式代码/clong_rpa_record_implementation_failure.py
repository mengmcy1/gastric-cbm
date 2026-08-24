#!/usr/bin/env python3
"""把RP-A runner的实现失败现场保存到独立不可覆盖目录。"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from clong_rpa_artifacts import validate_failure_record
from clong_rpa_train_development import OUTPUT_ROOT


PROTOCOL_BUNDLE_SHA = "768da344bfd3d49ca518528bc043a4eb2ef5b1b4f76b449c42e00c7223093280"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    """记录失败阶段、命令、日志尾部、Git与协议血缘。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = OUTPUT_ROOT / "implementation_failures" / stamp
    target.mkdir(parents=True)
    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines() \
        if args.log.is_file() else []
    record = {
        "failure_type": "implementation_failure",
        "is_formal_result": False,
        "failure_stage": args.stage,
        "exception_type": "SubprocessFailure",
        "message": f"exit_code={args.exit_code}; command={args.command}",
        "traceback": "\n".join(lines[-200:]),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "protocol_bundle_sha256": PROTOCOL_BUNDLE_SHA,
        "log_path": str(args.log),
    }
    validate_failure_record(record)
    (target / "implementation_failure.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"实现失败现场: {target}")


if __name__ == "__main__":
    main()
