#!/usr/bin/env python3
"""Resume and verify the one frozen ViewCrafter checkpoint without extra assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download


REPO_ID = "Drexubery/ViewCrafter_25_512"
REVISION = "8db619bda8cd745ac8df3918095f959517ecd318"
FILENAME = "model.ckpt"
EXPECTED_BYTES = 10_437_386_907
EXPECTED_SHA256 = "e15528ca4fade935222ba159356708294bf788c38e22b366597701372ed7f8df"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--retry-delay-seconds", type=float, default=15.0)
    args = parser.parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, object]] = []
    for attempt in range(1, args.max_attempts + 1):
        try:
            result = Path(hf_hub_download(
                repo_id=REPO_ID,
                filename=FILENAME,
                revision=REVISION,
                local_dir=args.local_dir,
            ))
            actual_bytes = result.stat().st_size
            if actual_bytes != EXPECTED_BYTES:
                raise RuntimeError(f"byte count mismatch: {actual_bytes} != {EXPECTED_BYTES}")
            actual_sha256 = digest(result)
            if actual_sha256 != EXPECTED_SHA256:
                raise RuntimeError(f"SHA256 mismatch: {actual_sha256} != {EXPECTED_SHA256}")
            print(json.dumps({
                "status": "pass",
                "repo_id": REPO_ID,
                "revision": REVISION,
                "path": str(result.resolve()),
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "attempts_this_run": attempt,
                "recoverable_failures": failures,
            }, indent=2))
            return
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            failure = {
                "attempt": attempt,
                "type": type(error).__name__,
                "message": str(error),
            }
            failures.append(failure)
            print(json.dumps({"status": "retrying", **failure}), flush=True)
            if attempt == args.max_attempts:
                raise
            time.sleep(args.retry_delay_seconds)


if __name__ == "__main__":
    main()
