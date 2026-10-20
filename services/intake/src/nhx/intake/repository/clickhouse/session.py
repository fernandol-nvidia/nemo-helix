# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse implementation of Intake session reads."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from nhx.intake.repository.clickhouse.executor import ClickHouseExecutor, ClickHouseQuery
from nhx.intake.repository.clickhouse.tables import ClickHouseTable
from nhx.intake.repository.session import SessionRepository
from nhx.intake.spans.domain import IntakeSession
from nhx.intake.spans.span_rollups import summed_metric_columns, usage_sql
from nhx.intake.spans.storage import float_or_none, int_or_none, normalize_span_status


class ClickHouseSessionRepository(SessionRepository):
    def __init__(self, executor: ClickHouseExecutor) -> None:
        self._executor = executor

    async def get_session(self, *, workspace: str, session_id: str) -> IntakeSession | None:
        query = _session_detail_query(self._executor.table(ClickHouseTable.SPANS)).bind(
            workspace=workspace,
            session_id=session_id,
        )
        rows = await self._executor.fetch_all(query)
        return _row_to_session(rows[0]) if rows else None


def _session_detail_query(table: str) -> ClickHouseQuery:
    """Build a primary-key-pruned aggregate over the current rows of one session."""

    usage = usage_sql(scaled=False)
    statement = f"""
        WITH
        -- The session's current spans: the columns the session reads, and the inputs of the usage rule.
        session_spans AS (
            SELECT
                source_format,
                trace_id,
                start_time,
                end_time,
                status,
                {usage.span_columns}
            FROM {table} FINAL
            PREWHERE workspace = %(workspace)s AND session_id = %(session_id)s
            WHERE is_deleted = 0
        ),
        -- One row per trace: its usage (see span_rollups) and its span times and counts.
        session_traces AS (
            SELECT
                min(session_spans.start_time) AS started_at,
                max(session_spans.end_time) AS ended_at,
                countIf(session_spans.end_time = toDateTime64(0, 6)) AS open_span_count,
                countIf(session_spans.status = 'error') AS error_span_count,
                countIf(session_spans.status = 'cancelled') AS cancelled_span_count,
                countIf(session_spans.status = 'unknown') AS unknown_span_count,
                {usage.trace_columns},
                count() AS span_count
            FROM session_spans
            GROUP BY session_spans.source_format, session_spans.trace_id
        )
        SELECT
            %(session_id)s AS id,
            %(workspace)s AS workspace,
            min(session_traces.started_at) AS started_at,
            if(sum(session_traces.open_span_count) > 0, NULL, max(session_traces.ended_at)) AS ended_at,
            multiIf(
                sum(session_traces.error_span_count) > 0, 'error',
                sum(session_traces.cancelled_span_count) > 0, 'cancelled',
                sum(session_traces.unknown_span_count) > 0, 'unknown',
                'success'
            ) AS status,
            {summed_metric_columns("session_traces")},
            count() AS trace_count,
            sum(session_traces.span_count) AS span_count
        FROM session_traces
        HAVING span_count > 0
    """
    return ClickHouseQuery(
        name="sessions.get",
        statement=statement,
        parameters=usage.parameters,
    )


def _row_to_session(row: dict[str, Any]) -> IntakeSession:
    ended_at = row.get("ended_at")
    return IntakeSession(
        id=row["id"],
        workspace=row["workspace"],
        started_at=row["started_at"],
        ended_at=ended_at,
        duration_ms=_duration_ms(row["started_at"], ended_at),
        status=normalize_span_status(row.get("status")),
        input_tokens=int_or_none(row.get("input_tokens")),
        output_tokens=int_or_none(row.get("output_tokens")),
        cached_tokens=int_or_none(row.get("cached_tokens")),
        total_tokens=int_or_none(row.get("total_tokens")),
        cost_usd=float_or_none(row.get("cost_usd")),
        cost_input_usd=float_or_none(row.get("cost_input_usd")),
        cost_output_usd=float_or_none(row.get("cost_output_usd")),
        trace_count=int(row["trace_count"]),
        span_count=int(row["span_count"]),
    )


def _duration_ms(started_at: datetime, ended_at: datetime | None) -> float | None:
    if ended_at is None:
        return None
    return (ended_at - started_at).total_seconds() * 1000
