"""Create an expanded, non-overwriting summary from a completed performance run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_performance_benchmark import aggregate, flatten_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-stem", default="performance_results_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.run_root / "performance_results.json"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    formal_results = source.get("formal_results")
    if not isinstance(formal_results, list) or not formal_results:
        raise ValueError("performance_results.json 不含正式结果")

    csv_path = args.run_root / f"{args.output_stem}.csv"
    json_path = args.run_root / f"{args.output_stem}.json"
    for path in (csv_path, json_path):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有汇总：{path}")

    rows = [flatten_result(result) for result in formal_results]
    with csv_path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    expanded = {
        "schema_version": "1.1",
        "source": source_path.name,
        "profile": source.get("profile"),
        "comparison_policy": source.get("comparison_policy"),
        "gpu_preflight": source.get("gpu_preflight"),
        "formal_rows": rows,
        "formal_aggregate": aggregate(rows),
    }
    json_path.write_text(
        json.dumps(expanded, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"csv": str(csv_path), "json": str(json_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
