"""Aggregate independent load-generator runs without discarding raw records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pd_disagg.loadgen import metric_summary, utc_now


def raw_path_for_summary(summary_path: Path) -> Path:
    suffix = ".summary.json"
    if not summary_path.name.endswith(suffix):
        raise ValueError(f"not a loadgen summary path: {summary_path}")
    return summary_path.with_name(summary_path.name[: -len(suffix)] + ".jsonl")


def read_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def aggregate(summary_paths: list[Path]) -> dict[str, Any]:
    if len(summary_paths) < 5:
        raise ValueError("reported aggregates require at least five independent runs")

    summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in summary_paths
    ]
    records = [
        record
        for path in summary_paths
        for record in read_records(raw_path_for_summary(path))
    ]
    included = [record for record in records if record["included_in_summary"]]
    successful = [record for record in included if record["status"] == "ok"]

    metric_runs: dict[str, Any] = {}
    for metric in ("ttft_s", "total_s", "itl_s"):
        metric_runs[metric] = {
            "run_medians": metric_summary(
                summary["summary"][metric]["median"]
                for summary in summaries
                if summary["summary"][metric] is not None
            ),
            "run_p90s": metric_summary(
                summary["summary"][metric]["p90"]
                for summary in summaries
                if summary["summary"][metric] is not None
            ),
        }

    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "run_count": len(summaries),
        "run_ids": [summary["run_id"] for summary in summaries],
        "request_count": len(included),
        "success_count": len(successful),
        "error_count": len(included) - len(successful),
        "validation": {
            "all_input_lengths_exact": all(
                record["input_length_requested"] == record["prompt_tokens_server"]
                for record in successful
            ),
            "all_output_lengths_exact": all(
                record["output_length_requested"]
                == record["completion_tokens_server"]
                for record in successful
            ),
            "all_itl_counts_valid": all(
                record["itl_valid"] for record in successful
            ),
        },
        "metrics_across_runs": metric_runs,
        "source_summaries": [str(path) for path in summary_paths],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--summaries", nargs="+", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    result = aggregate(args.summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

