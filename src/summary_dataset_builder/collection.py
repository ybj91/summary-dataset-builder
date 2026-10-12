"""Checkpoint-bounded collection; operational gates never grant human approval."""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
import time
from uuid import uuid4

from .files import atomic_json
from .generation import generate
from .store import RunStore


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _collector_lock(run_dir):
    path = run_dir / ".collection.lock"
    owner = {"pid": os.getpid(), "token": uuid4().hex, "created_at": _timestamp()}
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(
            "Collection lock exists; explicitly stop the worker and verify it is stopped "
            "before manually removing .collection.lock"
        ) from None
    identity = os.fstat(descriptor)
    ready = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(owner, stream)
            stream.flush()
            os.fsync(stream.fileno())
        ready = True
        yield
    finally:
        try:
            with path.open(encoding="utf-8") as stream:
                current_owner = json.load(stream)
        except (FileNotFoundError, ValueError):
            current_owner = None
        owned = isinstance(current_owner, dict) and current_owner.get("token") == owner["token"]
        if not ready and current_owner is None:
            try:
                present = path.stat()
                owned = (present.st_dev, present.st_ino) == (identity.st_dev, identity.st_ino)
            except FileNotFoundError:
                pass
        if owned:
            path.unlink()


def _blocked_reason(status: dict) -> str | None:
    if not isinstance(status.get("collection_policy"), dict) or not status["collection_policy"]:
        return "collection_not_authorized"
    if status.get("failed_gate"):
        return "stage_failed"
    if status.get("inflight_attempts"):
        return "inflight_requests"
    if status["accounting"].get(
        "blocking_uncertain_attempts", status["accounting"].get("uncertain_attempts")
    ) or status.get("uncertain_attempt_ids"):
        return "uncertain_billing"
    if status.get("reservation_overruns"):
        return "provider_cost_exceeded_reservation"
    if status["accounting"].get("budget_exceeded"):
        return "budget_exceeded"
    return None


def _validation_reason(error: ValueError) -> str:
    # Classify local validation failures without retaining potentially sensitive text.
    message = str(error).lower()
    if "uncertain" in message:
        return "uncertain_billing"
    if "in-flight" in message or "inflight" in message:
        return "inflight_requests"
    if "reservation" in message and ("exceeded" in message or "overrun" in message):
        return "provider_cost_exceeded_reservation"
    if "budget" in message:
        return "budget_exhausted"
    if "gate" in message:
        return "stage_failed"
    if "request limit" in message:
        return "total_request_limit"
    if "authoriz" in message or "approv" in message:
        return "collection_not_authorized"
    return "validation_failed"


def collect(
    store: RunStore, *, approve_paid: bool, max_requests: int = 30000,
    batch_size: int = 25, client=None, sleep=time.sleep, progress=None,
) -> dict:
    """Resume an explicitly authorized collection, persisting every batch and gate.

    ``max_requests`` limits this invocation; the store enforces the shared budget
    and lifetime authorization. A surviving lock requires explicit manual recovery,
    never automatic PID-based removal. No export or human approval is performed.
    Progress callbacks receive compact event records, not the complete gate history.
    """
    if approve_paid is not True:
        raise ValueError("Paid collection requires --approve-paid on each invocation")
    if type(max_requests) is not int or max_requests <= 0:
        raise ValueError("max_requests must be a positive integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    with _collector_lock(store.run_dir):
        started_at = _timestamp()
        status = {}
        requests = 0
        batch_start = None
        stage = None
        operation = None

        def emit(event, *, state="running", stop_reason=None):
            snapshot = {
                key: status[key]
                for key in (
                    "model", "providers", "dataset_mode", "targets", "counts", "accounting",
                    "generation_targets_met", "approvals", "inflight_attempts",
                    "uncertain_attempt_ids", "reservation_overruns", "collection_policy",
                    "usable_counts", "quarantined_counts", "required_checkpoint",
                    "next_checkpoint", "failed_gate", "human_review_required_before_release",
                    "unconfirmed_billing_attempt_ids", "retry_policy",
                )
                if key in status
            }
            snapshot.update({
                "event": event, "state": state, "stop_reason": stop_reason,
                "started_at": started_at, "updated_at": _timestamp(), "pid": os.getpid(),
                "requests_this_invocation": requests, "max_requests": max_requests,
                "batch_size": batch_size,
                "generation_through": status.get("accounting", {}).get("generation_attempts", 0),
                "stage": stage, "stage_count": len(status.get("stages", [])),
            })
            atomic_json(store.run_dir / "collection-progress.json", snapshot)
            event_record = {
                key: snapshot[key]
                for key in (
                    "event", "state", "stop_reason", "updated_at", "pid",
                    "requests_this_invocation", "generation_through", "stage_count",
                )
            }
            event_record["usable_counts"] = status.get("usable_counts", {})
            event_record["accounting"] = status.get("accounting", {})
            event_record["stage"] = {
                key: stage[key]
                for key in (
                    "passed", "generation_attempts", "window_start", "attempted",
                    "structurally_valid", "usable", "final",
                    "model_responses", "transport_failures", "structural_denominator",
                )
                if stage is not None and key in stage
            }
            with (store.run_dir / "collection-events.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as stream:
                stream.write(json.dumps(event_record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            if progress is not None:
                progress(deepcopy(event_record))
            return snapshot

        def stop(reason):
            return emit(
                "stop", state="completed" if reason == "targets_met" else "stopped",
                stop_reason=reason,
            )

        def refresh_after_failure():
            nonlocal status, requests, batch_start
            status = store.collection_status()
            if batch_start is not None:
                # generate can raise after earlier requests in its batch were persisted.
                requests += max(0, status["accounting"]["generation_attempts"] - batch_start)
                batch_start = None

        try:
            status = store.collection_status()
            stage = next(iter(reversed(status.get("stages", []))), None)
            emit("start")
            while True:
                status = store.collection_status()
                blocked = _blocked_reason(status)
                if blocked is not None:
                    return stop(blocked)
                if status["generation_targets_met"]:
                    operation = "gate"
                    stage = store.evaluate_collection_gate(final=True)
                    operation = None
                    status = store.collection_status()
                    emit("gate")
                    if not stage["passed"]:
                        return stop("stage_failed")
                    if status["generation_targets_met"]:
                        return stop(_blocked_reason(status) or "targets_met")
                    continue
                checkpoint = status.get("required_checkpoint")
                if checkpoint is not None:
                    operation = "gate"
                    stage = store.evaluate_collection_gate()
                    operation = None
                    status = store.collection_status()
                    emit("gate")
                    if not stage["passed"]:
                        return stop("stage_failed")
                    if status.get("required_checkpoint") == checkpoint:
                        raise ValueError("Collection gate did not advance its checkpoint")
                    continue
                if requests >= max_requests:
                    return stop("request_limit")
                accounting = status["accounting"]
                if Decimal(accounting["remaining_usd"]) <= 0:
                    return stop("budget_exhausted")
                remaining_total = (
                    status["collection_policy"]["max_total_requests"] - accounting["attempts"]
                )
                if remaining_total <= 0:
                    return stop("total_request_limit")
                remaining_stage = status["next_checkpoint"] - accounting["generation_attempts"]
                if remaining_stage <= 0:
                    raise ValueError("Collection gate checkpoint is not ahead of generation")
                size = min(batch_size, max_requests - requests, remaining_stage, remaining_total, 100000)
                options = {"approve_paid": True, "max_requests": size, "sleep": sleep}
                if client is not None:
                    options["client"] = client
                batch_start = accounting["generation_attempts"]
                operation = "generation"
                output = generate(store, **options)
                operation = None
                actual = output["requests_this_invocation"]
                if type(actual) is not int or not 0 <= actual <= size:
                    raise RuntimeError("Generation returned an invalid request count")
                requests += actual
                batch_start = None
                status = store.collection_status()
                emit("batch")
                reason = output["stop_reason"]
                if reason not in ("request_limit", "targets_met"):
                    return stop(reason)
                if actual == 0 and not status["generation_targets_met"]:
                    return stop("no_progress")
        except BaseException as error:
            try:
                refresh_after_failure()
            except Exception:
                pass
            if isinstance(error, ValueError) and operation is not None:
                return stop(_validation_reason(error))
            try:
                emit(
                    "stop", state="failed",
                    stop_reason="interrupted" if isinstance(error, KeyboardInterrupt) else "unexpected_error",
                )
            except Exception:
                # Preserve the original failure, including failures of durable progress I/O.
                pass
            raise
