from pathlib import Path

import pytest

from pd_disagg.aggregate import raw_path_for_summary


def test_raw_path_for_summary() -> None:
    assert raw_path_for_summary(Path("run-1.summary.json")) == Path("run-1.jsonl")


def test_raw_path_rejects_unknown_suffix() -> None:
    with pytest.raises(ValueError):
        raw_path_for_summary(Path("run-1.json"))

