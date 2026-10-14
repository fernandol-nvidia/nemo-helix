# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ATIF trajectory ingest for ClickHouse-backed Intake spans."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from nemo_helix_plugin.intake.atif_domain import (
    AtifIngestRequest,
    AtifTrajectory,
)
from nemo_helix_plugin.intake.atif_mapping import (
    ATIF_SOURCE,
    DEFAULT_ATIF_MAX_SUBAGENT_DEPTH,
    AtifTrajectoryDepthError,
    evaluator_rewards,
    trajectory_to_spans,
)
from nemo_helix_plugin.intake.types import DirectSpanInput
from nhx.common.entities.client import EntityClient
from nhx.common.service.dependencies import get_entity_client
from nhx.intake.config import IntakeConfig
from nhx.intake.spans.api.dependencies import DenormalizerDep, SpansServiceDep, require_workspace_access
from nhx.intake.spans.domain import EvaluatorResult, EvaluatorResultDataType, TraceBatch
from nhx.intake.spans.ingest.evaluation_context_validation import validate_evaluation_context
from nhx.intake.spans.ingest.spans import direct_span_to_domain
from nhx.intake.spans.storage import stable_id, utc_now

router = APIRouter(dependencies=[Depends(require_workspace_access)])
API_TAG = "Ingest"
EntityClientDep = Annotated[EntityClient, Depends(get_entity_client)]


def _atif_max_subagent_depth(request: Request) -> int:
    """Read the configured ATIF subagent depth cap from the Intake service."""
    service = getattr(request.app.state, "intake_service", None) or getattr(request.app.state, "service", None)
    cfg: IntakeConfig | None = getattr(service, "service_config", None) if service is not None else None
    if cfg is None:
        cfg = IntakeConfig()
    return cfg.atif_max_subagent_depth


@router.post(
    "/v2/workspaces/{workspace}/ingest/atif",
    tags=[API_TAG],
    status_code=status.HTTP_201_CREATED,
    response_class=Response,
)
async def ingest_atif(
    workspace: str,
    body: AtifIngestRequest,
    request: Request,
    service: SpansServiceDep,
    entity_client: EntityClientDep,
    denormalizer: DenormalizerDep,
) -> Response:
    await validate_evaluation_context(
        workspace=workspace,
        context=body.evaluation_context,
        entity_client=entity_client,
    )
    ingested_at = utc_now()
    trajectory = body.to_trajectory()
    max_subagent_depth = _atif_max_subagent_depth(request)
    try:
        batch = atif_trace_batch(
            workspace=workspace,
            trajectory=trajectory,
            ingested_at=ingested_at,
            max_subagent_depth=max_subagent_depth,
        )
    except AtifTrajectoryDepthError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
    await service.ingest_batch(batch)
    context = body.evaluation_context
    if denormalizer is not None and context is not None and context.evaluation_name:
        denormalizer.mark_dirty(workspace=workspace, evaluation_name=context.evaluation_name)
    return Response(status_code=status.HTTP_201_CREATED)


def atif_trace_batch(
    *,
    workspace: str,
    trajectory: AtifTrajectory,
    ingested_at: datetime,
    max_subagent_depth: int = DEFAULT_ATIF_MAX_SUBAGENT_DEPTH,
) -> TraceBatch:
    """Map a trajectory with the shared SDK mapping and store it as /ingest/spans does, with its reward rows."""
    spans = trajectory_to_spans(
        workspace=workspace,
        trajectory=trajectory,
        ingested_at=ingested_at,
        max_subagent_depth=max_subagent_depth,
    )
    return TraceBatch(
        spans=[
            direct_span_to_domain(workspace=workspace, source=ATIF_SOURCE, span=span, ingested_at=ingested_at)[0]
            for span in spans
        ],
        evaluator_results=_evaluator_results(
            workspace=workspace,
            session_id=trajectory.session_id,
            spans=spans,
            ingested_at=ingested_at,
        ),
    )


def _evaluator_results(
    *,
    workspace: str,
    session_id: str,
    spans: list[DirectSpanInput],
    ingested_at: datetime,
) -> list[EvaluatorResult]:
    """Emit one evaluator_results row per Harbor reward key, targeting its EVALUATOR span.

    The span preserves the verifier payload; these rows make each score queryable by name and value without parsing
    the span payload.
    """
    results: list[EvaluatorResult] = []
    for span in spans:
        for name, raw_value in evaluator_rewards(span):
            data_type, value, string_value = _coerce_evaluator_value(raw_value)
            results.append(
                EvaluatorResult(
                    # Per-key id: the reward name keeps each criterion's row distinct on the
                    # same span, and an identical re-ingest hashes to the same id (dedupe).
                    evaluator_result_id=stable_id(span.span_id, name, prefix="eval"),
                    span_id=span.span_id,
                    session_id=session_id,
                    workspace=workspace,
                    name=name,
                    value=value,
                    string_value=string_value,
                    data_type=data_type,
                    comment=None,
                    created_by="intake:atif_importer",
                    created_at=ingested_at,
                    ingested_at=ingested_at,
                )
            )
    return results


def _coerce_evaluator_value(
    score: bool | int | float | str,
) -> tuple[EvaluatorResultDataType, float | None, str | None]:
    """Coerce an ATIF reward into evaluator result storage columns."""
    if isinstance(score, bool):
        return EvaluatorResultDataType.BOOLEAN, 1.0 if score else 0.0, None
    if isinstance(score, (int, float)):
        return EvaluatorResultDataType.NUMERIC, float(score), None
    return EvaluatorResultDataType.TEXT, None, str(score)
