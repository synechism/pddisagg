from __future__ import annotations

import pytest

from pd_disagg.correctness import output_token_ids


def test_output_token_ids_requires_one_tokenized_choice() -> None:
    assert output_token_ids({"choices": [{"token_ids": [3, 4]}]}) == [3, 4]
    with pytest.raises(RuntimeError):
        output_token_ids({"choices": []})
    with pytest.raises(RuntimeError):
        output_token_ids({"choices": [{"text": "missing token IDs"}]})
