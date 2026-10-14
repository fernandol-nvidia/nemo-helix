# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reading an ATIF file one step at a time."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nhx_evals_sdk.values import atif_stream
from nhx_evals_sdk.values.atif import Trajectory
from nhx_evals_sdk.values.atif_stream import read_atif_lazily
from nhx_evals_sdk.values.evidence import ATIFTraceHandle, EvidenceDescriptor

DOCUMENT = {
    "schema_version": "ATIF-v1.7",
    "session_id": "s",
    "agent": {"name": "codex", "version": "1.0"},
    "extra": {
        "observed_events": [{"kind": "scope", "data": {"n": [1, 2.5, None]}}] * 3,
        "nemo_relay": {},
        "counts": [1234567, 89.125, -42],
    },
    "steps": [
        {"step_id": 1, "source": "user", "message": "q 🙂", "timestamp": "2026-01-01T00:00:00+00:00"},
        {
            "step_id": 2,
            "source": "agent",
            "message": "a",
            "extra": {"llm_request": {"messages": [{"content": "x" * 50}]}, "count": 12345},
        },
    ],
    "final_metrics": {"total_prompt_tokens": 10},
}


@pytest.mark.parametrize("indent", [None, 2])
@pytest.mark.parametrize("chunk", [1, 7, 1 << 20])
def test_lazy_read_matches_a_full_read_without_the_top_level_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, indent: int | None, chunk: int
) -> None:
    # Small chunks split strings, numbers and multi-byte characters across reads.
    monkeypatch.setattr(atif_stream, "_CHUNK_CHARS", chunk)
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(DOCUMENT, indent=indent, ensure_ascii=False), encoding="utf-8")

    lazy = read_atif_lazily(path)
    full = Trajectory.model_validate(DOCUMENT)

    assert list(lazy.steps) == full.steps
    assert lazy.steps[-1] == full.steps[-1]
    assert lazy.model_dump(exclude={"steps", "extra"}) == full.model_dump(exclude={"steps", "extra"})
    assert lazy.extra is None


def test_lazy_read_never_decodes_the_event_log_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events = [{"kind": "scope", "data": "x" * 100}] * 50
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps({**DOCUMENT, "extra": {"observed_events": events}}), encoding="utf-8")
    decoded: list[int] = []
    value = atif_stream._Scanner.value

    def spy(self: atif_stream._Scanner) -> tuple[object, str]:
        result = value(self)
        decoded.append(len(result[1]))
        return result

    monkeypatch.setattr(atif_stream._Scanner, "value", spy)

    read_atif_lazily(path)

    assert max(decoded) < len(json.dumps(events)) / 10


def test_lazy_read_rejects_a_truncated_document(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(DOCUMENT)[:-40], encoding="utf-8")

    with pytest.raises(ValueError):
        read_atif_lazily(path)


async def test_lazy_trace_reads_a_file_backed_trace_without_caching(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(DOCUMENT), encoding="utf-8")
    handle = ATIFTraceHandle(EvidenceDescriptor(kind="trace", format="atif", ref=str(path)))

    first, second = await handle.lazy_trace(), await handle.lazy_trace()

    assert len(first.steps) == 2
    assert first.extra is None
    assert first is not second
