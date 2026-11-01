from __future__ import annotations

import argparse
from contextlib import nullcontext
from types import SimpleNamespace
from uuid import uuid4

import pytest

from smartmeter.services import detection_worker
from smartmeter.services.detection_worker import (
    WorkerConfig,
    _build_parser,
    _positive_float,
    _positive_integer,
)


def test_worker_config_requires_safe_lease_timing():
    with pytest.raises(ValueError, match="twice heartbeat_seconds"):
        WorkerConfig(lease_seconds=40, heartbeat_seconds=20)

    assert WorkerConfig(lease_seconds=41, heartbeat_seconds=20).lease_seconds == 41


@pytest.mark.parametrize(
    ("parser", "invalid_value"),
    [
        (_positive_float, "0"),
        (_positive_float, "invalid"),
        (_positive_integer, "-1"),
        (_positive_integer, "invalid"),
    ],
)
def test_worker_configuration_arguments_require_positive_values(parser, invalid_value):
    with pytest.raises(argparse.ArgumentTypeError):
        parser(invalid_value)


def test_worker_cli_defaults_and_one_shot_mode():
    arguments = _build_parser().parse_args(["--once"])

    assert arguments.once is True
    assert arguments.poll_seconds == 2.0
    assert arguments.lease_seconds == 120
    assert arguments.heartbeat_seconds == 20


def test_one_shot_worker_returns_failure_exit_code(monkeypatch):
    monkeypatch.setattr(detection_worker, "configure_logging", lambda: None)
    monkeypatch.setattr(detection_worker, "create_session_factory", lambda: object())
    monkeypatch.setattr(detection_worker, "run_one_detection", lambda **_kwargs: "failed")

    assert detection_worker.main(["--once"]) == 1


def test_worker_persists_safe_failure_status(monkeypatch):
    session = object()
    run = SimpleNamespace(id=uuid4(), lease_token=uuid4())
    failure = {}

    class FakeFactory:
        def begin(self):
            return nullcontext(session)

    class FakeDetectionRunRepository:
        def fail_exhausted(self, *_args, **_kwargs):
            return 0

        def claim_next(self, *_args, **_kwargs):
            return run

        def mark_failed(self, _session, _run_id, _token, **kwargs):
            failure.update(kwargs)
            return True

    def fail_detection(*_args, **_kwargs):
        raise RuntimeError("internal detail")

    monkeypatch.setattr(
        detection_worker,
        "DetectionRunRepository",
        FakeDetectionRunRepository,
    )
    monkeypatch.setattr(
        detection_worker,
        "process_detection_run",
        fail_detection,
    )

    result = detection_worker.run_one_detection(
        session_factory=FakeFactory(),
        config=WorkerConfig(poll_seconds=0.1, lease_seconds=5, heartbeat_seconds=2),
    )

    assert result == "failed"
    assert "RuntimeError" in failure["error_message"]
    assert "internal detail" not in failure["error_message"]
