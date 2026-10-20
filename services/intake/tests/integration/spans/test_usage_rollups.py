# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every usage view counts each model call's usage once."""

from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from typing import Any, NamedTuple

import pytest
from fastapi.testclient import TestClient

WORKSPACE = "/apis/intake/v2/workspaces/default"
EVALUATION = "usage-rollups"


class Usage(NamedTuple):
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: Decimal | None = None
    total_tokens: int | None = None


# Short names for the attributes the tests set, a mix of OpenTelemetry GenAI and OpenInference names.
_KEYS = {
    "kind": "openinference.span.kind",
    "operation": "gen_ai.operation.name",
    "input": "gen_ai.usage.input_tokens",
    "output": "gen_ai.usage.output_tokens",
    "cached": "llm.token_count.prompt_details.cache_read",
    "total": "llm.token_count.total",
    "cost": "llm.cost.total",
    "model": "gen_ai.request.model",
    "agent": "gen_ai.agent.name",
}


def _span(span_id: int, *, parent: int | None = None, **fields: Any) -> dict[str, Any]:
    attributes = {_KEYS.get(name, name): value for name, value in fields.items()}
    return {"span_id": f"{span_id:016x}", "parent_span_id": parent and f"{parent:016x}", "attributes": attributes}


@pytest.fixture(autouse=True)
def _evaluation(client: TestClient) -> None:
    """Ingest rejects traces that name an evaluation before it exists."""
    group = client.post(f"{WORKSPACE}/experiments", json={"name": EVALUATION})
    group = group if group.status_code != 409 else client.get(f"{WORKSPACE}/experiments/{EVALUATION}")
    evaluation = {"name": EVALUATION, "experiment_group_id": group.json()["id"], "dataset_name": "d"}
    assert client.post(f"{WORKSPACE}/evaluations", json=evaluation).status_code in {201, 409}


def test_every_usage_view_counts_each_call_once(client: TestClient, make_otlp_request) -> None:
    traces = [
        # Calls 2-5 (2 is a root call) beat unlabeled root 1's totals. Root 1 covers the cached tokens and cost
        # that no call reports, ahead of agent span 6. Tools, agents and chains are not calls, and an
        # OpenInference kind wins over the operation (span 7).
        [
            _span(1, input=999, output=999, cached=40, cost=0.1, total=9999, model="a", agent="a"),
            _span(2, operation="chat", input=100, output=10, total=110),
            _span(3, parent=1, kind="LLM", input=200, output=20, total=220),
            _span(4, parent=1, operation="embeddings", input=300),
            _span(5, parent=1, input=50, output=5),
            _span(6, parent=1, operation="invoke_agent", input=999, output=999, cost=9.9),
            _span(7, parent=1, kind="CHAIN", operation="chat", input=999),
            _span(8, parent=1, kind="TOOL", input=999),
        ],
        # With nothing on the calls or the root, the spans that report usage still count.
        [
            _span(9),
            _span(10, parent=9, operation="invoke_agent", input=40, output=4),
            _span(11, parent=9, operation="invoke_agent", input=60),
        ],
        # The root stands in for its calls, except where a call reports a field, even as zero.
        [
            _span(12, kind="AGENT", input=50, output=5, cached=9, cost=0.2, model="b", agent="b"),
            _span(13, parent=12, kind="LLM", cached=0),
        ],
    ]
    expected = [
        Usage(input_tokens=650, output_tokens=35, cached_tokens=40, cost_usd=Decimal("0.1"), total_tokens=330),
        Usage(input_tokens=100, output_tokens=4),
        Usage(input_tokens=50, output_tokens=5, cached_tokens=0, cost_usd=Decimal("0.2")),
    ]

    trace_ids = _ingest(client, make_otlp_request, "usage", traces)
    _assert_usage(client, "usage", dict(zip(trace_ids, expected, strict=True)))
    rollup = _get(client, f"evaluations/{EVALUATION}")
    assert (rollup["model_names"], rollup["agent_names"]) == (["a", "b"], ["a", "b"])


def test_republishing_a_trace_counts_its_latest_spans_once(client: TestClient, make_otlp_request) -> None:
    # Span 2 becomes a chain over a new call, unlabeled span 3 gains its parent, and span 4 drops its usage.
    first = [
        _span(1, kind="AGENT", input=999),
        _span(2, parent=1, kind="LLM", input=100),
        _span(3, input=50),
        _span(4, parent=1, kind="LLM", input=70),
    ]
    revised = [
        _span(1, kind="AGENT", input=999),
        _span(2, parent=1, kind="CHAIN", input=150),
        _span(5, parent=2, kind="LLM", input=150),
        _span(3, parent=1, input=50),
        _span(4, parent=1, kind="LLM"),
    ]
    for spans in (first, first, revised):
        [trace_id] = _ingest(client, make_otlp_request, "usage", [spans])

    _assert_usage(client, "usage", {trace_id: Usage(input_tokens=200)})


def test_evaluation_sessions_sort_by_counted_usage(client: TestClient, make_otlp_request) -> None:
    # Summing every span would sort usage-totals last by tokens and by cost.
    for session_id, spans in {
        "usage-totals": [
            _span(1, kind="AGENT", input=900, output=900, cost=0.9),
            _span(2, parent=1, kind="LLM", input=10, output=1, cost=0.005),
        ],
        "usage-calls": [_span(1, kind="AGENT"), _span(2, parent=1, kind="LLM", input=100, output=1000, cost=0.001)],
        "usage-root": [_span(1, kind="AGENT", input=550, cost=0.02)],
    }.items():
        _ingest(client, make_otlp_request, session_id, [spans])

    for sort, expected in (("tokens", ["totals", "root", "calls"]), ("cost_total_usd", ["calls", "totals", "root"])):
        rows = _get(client, f"evaluations/{EVALUATION}/sessions", params={"sort": sort})["data"]
        assert [row["session_id"] for row in rows] == [f"usage-{name}" for name in expected]


def test_atif_usage_counts_as_ingested(client: TestClient) -> None:
    # ATIF ingest puts a subagent's final_metrics on its agent span, beside a step that reports usage.
    step = {"step_id": 1, "timestamp": datetime.now(timezone.utc).isoformat(), "source": "agent", "message": "done"}
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "session_id": "usage",
        "evaluation_context": {"evaluation_name": EVALUATION, "test_case_name": "usage"},
        "agent": {"name": "orchestrator", "version": "1"},
        "steps": [{**step, "metrics": {"prompt_tokens": 100, "completion_tokens": 10}}],
        "subagent_trajectories": [
            {
                "schema_version": "ATIF-v1.7",
                "trajectory_id": "worker",
                "agent": {"name": "worker", "version": "1"},
                "final_metrics": {"total_prompt_tokens": 50, "total_completion_tokens": 5},
                "steps": [step],
            }
        ],
    }
    response = client.post(f"{WORKSPACE}/ingest/atif", json=trajectory)
    assert response.status_code == 201, response.text

    _assert_usage(client, "usage", {"usage": Usage(input_tokens=150, output_tokens=15, total_tokens=165)})


def _ingest(client: TestClient, make_otlp_request, session_id: str, traces: list[list[dict[str, Any]]]) -> list[str]:
    """Publish each trace as OTLP; a span's start time follows its id, so republished spans keep their identity."""
    base_ns = int(datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1e9)
    trace_ids = [sha256(f"{session_id}/{index}".encode()).hexdigest()[:32] for index in range(len(traces))]
    for trace_id, spans in zip(trace_ids, traces):
        for span in spans:
            span["attributes"]["gen_ai.conversation.id"] = session_id
            if not span["parent_span_id"]:
                span["attributes"] |= {"nemo.evaluation.name": EVALUATION, "nemo.test_case.name": session_id}
            span["start_time_unix_nano"] = base_ns + int(span["span_id"], 16) * 1_000_000
        response = client.post(
            f"{WORKSPACE}/ingest/otlp/v1/traces",
            content=make_otlp_request(spans, trace_id=trace_id),
            headers={"Content-Type": "application/x-protobuf"},
        )
        assert response.status_code == 200 and response.json() == {"errors": []}, response.text
    return trace_ids


def _assert_usage(client: TestClient, session_id: str, expected: dict[str, Usage]) -> None:
    """The trace list, session detail, evaluation sessions and evaluation rollup agree; sessions sum traces."""
    session = Usage(*(_total(values) for values in zip(*expected.values())))
    traces = _get(client, "traces", params={"filter[session_id]": session_id})["data"]
    assert {trace["id"]: _usage(trace) for trace in traces} == expected
    assert _usage(_get(client, f"sessions/{session_id}")) == session
    row = {row["session_id"]: row for row in _get(client, f"evaluations/{EVALUATION}/sessions")["data"]}[session_id]
    assert _usage({**row, "cost_usd": row["cost_total_usd"]}) == session._replace(total_tokens=None)
    rollup = _get(client, f"evaluations/{EVALUATION}")
    assert (rollup.get("tokens") or {}).get("sum") == _total([session.input_tokens, session.output_tokens])
    assert _decimal((rollup.get("cost_usd") or {}).get("sum")) == session.cost_usd


def _get(client: TestClient, path: str, **kwargs: Any) -> dict[str, Any]:
    response = client.get(f"{WORKSPACE}/{path}", **kwargs)
    assert response.status_code == 200, response.text
    return response.json()


def _usage(payload: dict[str, Any]) -> Usage:
    usage = Usage(*(payload.get(field) for field in Usage._fields))
    return usage._replace(cost_usd=_decimal(payload.get("cost_usd")))


def _decimal(value: float | None) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _total(values) -> Any:
    reported = [value for value in values if value is not None]
    return sum(reported) if reported else None
