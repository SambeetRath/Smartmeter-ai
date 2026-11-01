from __future__ import annotations

import argparse
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Sequence
from uuid import UUID

from smartmeter.database.repositories import DetectionRunRepository
from smartmeter.database.session import SessionFactory, create_session_factory
from smartmeter.logging_config import configure_logging
from smartmeter.services.detection_service import process_detection_run

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 2.0
DEFAULT_LEASE_SECONDS = 120
DEFAULT_HEARTBEAT_SECONDS = 20
MAXIMUM_ATTEMPTS = 3


@dataclass(frozen=True)
class WorkerConfig:
    poll_seconds: float = DEFAULT_POLL_SECONDS
    lease_seconds: int = DEFAULT_LEASE_SECONDS
    heartbeat_seconds: int = DEFAULT_HEARTBEAT_SECONDS
    maximum_attempts: int = MAXIMUM_ATTEMPTS

    def __post_init__(self) -> None:
        if not math.isfinite(self.poll_seconds) or self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive.")
        if self.lease_seconds < 3 or self.heartbeat_seconds < 1:
            raise ValueError("lease_seconds and heartbeat_seconds must be positive.")
        if self.lease_seconds <= self.heartbeat_seconds * 2:
            raise ValueError("lease_seconds must exceed twice heartbeat_seconds.")
        if self.maximum_attempts < 1:
            raise ValueError("maximum_attempts must be positive.")


class _LeaseHeartbeat:
    def __init__(
        self,
        run_id: UUID,
        lease_token: UUID,
        *,
        session_factory: SessionFactory,
        config: WorkerConfig,
    ) -> None:
        self.run_id = run_id
        self.lease_token = lease_token
        self.session_factory = session_factory
        self.config = config
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name=f"detection-lease-{run_id}",
            daemon=True,
        )

    def _run(self) -> None:
        repository = DetectionRunRepository()
        while not self.stop_event.wait(self.config.heartbeat_seconds):
            try:
                with self.session_factory.begin() as session:
                    alive = repository.heartbeat(
                        session,
                        self.run_id,
                        self.lease_token,
                        lease_seconds=self.config.lease_seconds,
                        now=datetime.now(timezone.utc),
                    )
            except Exception:
                logger.exception("Could not renew detection lease for run %s.", self.run_id)
                continue
            if not alive:
                logger.warning(
                    "Detection lease for run %s is no longer owned by this worker.",
                    self.run_id,
                )
                return

    def __enter__(self) -> _LeaseHeartbeat:
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.config.heartbeat_seconds + 1)


def _claim_run(
    session_factory: SessionFactory,
    config: WorkerConfig,
) -> tuple[UUID, UUID] | None:
    with session_factory.begin() as session:
        repository = DetectionRunRepository()
        repository.fail_exhausted(
            session,
            maximum_attempts=config.maximum_attempts,
            now=datetime.now(timezone.utc),
        )
        run = repository.claim_next(
            session,
            lease_seconds=config.lease_seconds,
            now=datetime.now(timezone.utc),
            maximum_attempts=config.maximum_attempts,
        )
        if run is None:
            return None
        if run.lease_token is None:
            raise RuntimeError("Claimed detection run has no lease token.")
        return run.id, run.lease_token


def run_one_detection(
    *,
    session_factory: SessionFactory | None = None,
    config: WorkerConfig | None = None,
) -> Literal["idle", "completed", "failed"]:
    """Claim at most one job and report whether it was idle, completed, or failed."""
    factory = session_factory or create_session_factory()
    worker_config = config or WorkerConfig()
    claimed = _claim_run(factory, worker_config)
    if claimed is None:
        return "idle"

    run_id, lease_token = claimed
    heartbeat = _LeaseHeartbeat(
        run_id,
        lease_token,
        session_factory=factory,
        config=worker_config,
    )
    try:
        with heartbeat:
            process_detection_run(run_id, lease_token, session_factory=factory)
        return "completed"
    except Exception as exc:
        logger.exception("Detection run %s failed.", run_id)
        try:
            with factory.begin() as session:
                failed = DetectionRunRepository().mark_failed(
                    session,
                    run_id,
                    lease_token,
                    error_message=f"Detection failed ({type(exc).__name__}); see worker logs.",
                    now=datetime.now(timezone.utc),
                )
            if not failed:
                logger.warning(
                    "Failed run %s could not be updated because its lease changed.",
                    run_id,
                )
        except Exception:
            logger.exception("Could not persist failure state for detection run %s.", run_id)
        return "failed"


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number greater than zero.") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return result


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer.") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero.")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SmartMeter AI detection queue worker.")
    parser.add_argument(
        "--poll-seconds",
        type=_positive_float,
        default=os.getenv("SMARTMETER_DETECTION_POLL_SECONDS", str(DEFAULT_POLL_SECONDS)),
    )
    parser.add_argument(
        "--lease-seconds",
        type=_positive_integer,
        default=os.getenv("SMARTMETER_DETECTION_LEASE_SECONDS", str(DEFAULT_LEASE_SECONDS)),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=_positive_integer,
        default=os.getenv(
            "SMARTMETER_DETECTION_HEARTBEAT_SECONDS",
            str(DEFAULT_HEARTBEAT_SECONDS),
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Claim at most one queued run and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        config = WorkerConfig(
            poll_seconds=arguments.poll_seconds,
            lease_seconds=arguments.lease_seconds,
            heartbeat_seconds=arguments.heartbeat_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))
    configure_logging()
    factory = create_session_factory()
    logger.info(
        "Detection worker started: poll=%.1fs lease=%ds heartbeat=%ds.",
        config.poll_seconds,
        config.lease_seconds,
        config.heartbeat_seconds,
    )

    if arguments.once:
        return int(run_one_detection(session_factory=factory, config=config) == "failed")

    try:
        while True:
            outcome = run_one_detection(session_factory=factory, config=config)
            if outcome == "idle":
                time.sleep(config.poll_seconds)
    except KeyboardInterrupt:
        logger.info("Detection worker stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
