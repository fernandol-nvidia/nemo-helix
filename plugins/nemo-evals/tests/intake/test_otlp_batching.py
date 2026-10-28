# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_evals.intake.mapping import bounded_otlp_requests
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest


def _request() -> ExportTraceServiceRequest:
    request = ExportTraceServiceRequest()
    for resource_index in range(2):
        resource = request.resource_spans.add(schema_url=f"resource-{resource_index}")
        resource.resource.attributes.add(key="service.name").value.string_value = "producer λ"
        resource.resource.dropped_attributes_count = 3
        for scope_index in range(2):
            scope = resource.scope_spans.add(schema_url=f"scope-{scope_index}")
            scope.scope.name = "instrumentation"
            scope.scope.version = "1.0"
            scope.scope.attributes.add(key="scope-key").value.int_value = scope_index
            for span_index in range(3):
                index = resource_index * 6 + scope_index * 3 + span_index
                span = scope.spans.add(
                    trace_id=b"t" * 16,
                    span_id=(index + 1).to_bytes(8),
                    parent_span_id=b"" if index == 0 else (1).to_bytes(8),
                    name="λ" * 80,
                    start_time_unix_nano=100 + index,
                    end_time_unix_nano=200 + index,
                )
                span.attributes.add(key="input.value").value.string_value = "input"
                span.events.add(name="event")
                span.links.add(trace_id=b"l" * 16, span_id=b"s" * 8)
    return request


def _entries(payload: bytes) -> list[tuple[bytes, str, bytes, str, bytes]]:
    request = ExportTraceServiceRequest.FromString(payload)
    return [
        (
            resource.resource.SerializeToString(),
            resource.schema_url,
            scope.scope.SerializeToString(),
            scope.schema_url,
            span.SerializeToString(),
        )
        for resource in request.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    ]


def test_a_request_at_the_budget_is_unchanged() -> None:
    payload = _request().SerializeToString()
    assert bounded_otlp_requests(payload, max_bytes=len(payload)) == [payload]


def test_split_requests_preserve_every_span_in_order_and_its_resource_and_scope() -> None:
    payload = _request().SerializeToString()
    requests = bounded_otlp_requests(payload, max_bytes=800)
    assert len(requests) > 1
    assert all(len(request) <= 800 for request in requests)
    assert [entry for request in requests for entry in _entries(request)] == _entries(payload)


def test_a_late_oversized_span_is_rejected() -> None:
    request = _request()
    request.resource_spans[-1].scope_spans[-1].spans[-1].name = "x" * 1000
    with pytest.raises(ValueError, match="One OTLP span exceeds"):
        bounded_otlp_requests(request.SerializeToString(), max_bytes=800)


@pytest.mark.parametrize("span_bytes", [1, 127, 128, 16383, 16384])
def test_split_requests_fit_exactly_including_the_protobuf_envelope(span_bytes) -> None:
    request = ExportTraceServiceRequest()
    resource = request.resource_spans.add(schema_url="resource")
    scope = resource.scope_spans.add(schema_url="scope")
    span = scope.spans.add(name="x" * span_bytes)
    resource.resource.attributes.add(key="service.name").value.string_value = "producer"
    scope.scope.name = "instrumentation"
    single = request.SerializeToString()
    scope.spans.append(span)
    assert bounded_otlp_requests(request.SerializeToString(), max_bytes=len(single)) == [single, single]
    with pytest.raises(ValueError, match="One OTLP span exceeds"):
        bounded_otlp_requests(single, max_bytes=len(single) - 1)


@pytest.mark.parametrize("budget", [0, -1])
def test_invalid_otlp_budget_is_rejected(budget) -> None:
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        bounded_otlp_requests(b"", max_bytes=budget)
