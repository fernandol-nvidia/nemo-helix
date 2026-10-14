# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path

import pytest
from nhx_evals_sdk.agent_eval.evaluator import AgentEvaluator
from nhx_evals_sdk.agent_eval.persistence import persist_run, read_trials
from nhx_evals_sdk.agent_eval.runtimes.gym import GymRewardMetric
from nhx_evals_sdk.agent_eval.runtimes.gym.results import trials_from_rollouts
from nhx_evals_sdk.agent_eval.tasks import AgentEvalRunConfig, AgentEvalTask
from nhx_evals_sdk.agent_eval.trials import TrialError

TASKS = [AgentEvalTask(id="task", intent="question", inputs={}, metrics=[GymRewardMetric()])]


def _trials(tmp_path: Path, exports: int, *, inside: bool = True) -> list:
    trace = tmp_path / ("bundle/relay" if inside else "export") / "trajectory.json"
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps({"schema_version": "ATIF-v1.7", "steps": [{"source": "agent", "message": "done"}]}))
    artifacts = [{"name": f"trace-{i}", "kind": "atif", "path": str(trace)} for i in range(exports)]
    record = {"_ng_task_index": 0, "reward": 1.0, "fabric_result": {"artifacts": {"artifacts": artifacts}}}
    rollouts = tmp_path / "bundle" / "rollouts.jsonl"
    rollouts.parent.mkdir(exist_ok=True)
    rollouts.write_text(json.dumps(record) + "\n")
    return trials_from_rollouts(rollouts, TASKS, {0: "task"})


@pytest.mark.asyncio
async def test_declared_atif_is_the_trace_of_a_relocated_bundle(tmp_path: Path) -> None:
    trials = _trials(tmp_path, 1)
    bundle = tmp_path / "bundle"
    result = await AgentEvaluator().run(tasks=TASKS, trials=trials, config=AgentEvalRunConfig(work_dir=bundle))
    persist_run(result, bundle, write_html_dashboard=False)
    shutil.move(bundle, tmp_path / "moved")
    (restored,) = read_trials(tmp_path / "moved")
    assert restored.evidence is not None
    assert [step.message for step in await (await restored.evidence.trace(format="atif")).steps()] == ["done"]


@pytest.mark.parametrize(
    ("exports", "inside", "trace"), [(0, True, "otlp"), (1, False, "otlp"), (1, True, "atif"), (2, True, "otlp")]
)
def test_only_a_single_local_atif_replaces_the_gym_projection(
    tmp_path: Path, exports: int, inside: bool, trace: str
) -> None:
    (trial,) = _trials(tmp_path, exports, inside=inside)
    assert trial.evidence is not None
    traces = {key: d.format for key, d in trial.evidence.descriptors.items() if key.startswith("trace")}
    assert traces == {"trace": trace, f"trace:{trace}": trace}


def test_failed_rollout_keeps_its_diagnostic_and_model_call_trace(tmp_path: Path) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text("")
    failure = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "_ng_failure_class": "agent_run_error",
        "_ng_failure_type": "FabricInvocationError",
        "_ng_failure_message": "NeMo Fabric invocation failed: provider unavailable",
    }
    (tmp_path / "rollouts_failures.jsonl").write_text(json.dumps(failure) + "\n")
    capture = tmp_path / "model_calls"
    capture.mkdir()
    (capture / "0-0.capture.jsonl").write_text('{"model_call_id": "recorded-call"}\n')

    (trial,) = trials_from_rollouts(rollouts, TASKS, {0: "task"}, capture_dir=capture)

    assert trial.status.value == "failed"
    assert trial.error == TrialError(type="FabricInvocationError", message=failure["_ng_failure_message"])
    assert trial.evidence is not None
    assert {"ng_trajectory", "trace:otlp"} <= trial.evidence.descriptors.keys()
    assert trial.evidence.require("rollouts").ref == str(tmp_path / "rollouts_failures.jsonl")
