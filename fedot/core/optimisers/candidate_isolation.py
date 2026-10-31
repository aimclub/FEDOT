"""Spawn-safe, killable evaluation of independent optimizer candidates."""

from __future__ import annotations

import math
import multiprocessing
import os
import select
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import cloudpickle


@dataclass(frozen=True)
class CandidateOutcome:
    uid: str
    status: str
    value: Any = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    pid: Optional[int] = None
    payload_bytes: int = 0
    serialization_seconds: float = 0.0
    startup_seconds: float = 0.0
    cleanup_confirmed: bool = True


class CandidateCleanupError(RuntimeError):
    """Raised when an evaluation process cannot be confirmed stopped."""

    def __init__(self, message, *, outcomes=(), unresolved_candidate_pids=()):
        super().__init__(message)
        self.outcomes = tuple(outcomes)
        self.unresolved_candidate_pids = tuple(unresolved_candidate_pids)


_CANDIDATE_GUARD_TIMEOUT = 2.0


def _start_candidate_lifetime_guard(candidate_pgid: int) -> tuple[subprocess.Popen, int]:
    """Start a fixed guardian before allowing the candidate to evaluate."""
    control_read = control_write = ready_read = ready_write = -1
    guardian = None
    try:
        control_read, control_write = os.pipe()
        ready_read, ready_write = os.pipe()
        command = [
            sys.executable,
            "-I",
            os.path.join(os.path.dirname(__file__), "candidate_lifetime_guard.py"),
            "--candidate-pgid",
            str(candidate_pgid),
            "--control-fd",
            str(control_read),
            "--ready-fd",
            str(ready_write),
        ]
        guardian = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            pass_fds=(control_read, ready_write),
            start_new_session=True,
        )
        os.close(control_read)
        control_read = -1
        os.close(ready_write)
        ready_write = -1
        if not select.select([ready_read], [], [], _CANDIDATE_GUARD_TIMEOUT)[0]:
            raise RuntimeError("candidate lifetime guardian did not become ready")
        if os.read(ready_read, 1) != b"R":
            raise RuntimeError("candidate lifetime guardian exited before readiness")
        return guardian, control_write
    except BaseException as original:
        if control_write != -1:
            os.close(control_write)
            control_write = -1
        if guardian is not None:
            try:
                _finish_candidate_lifetime_guard(guardian, None, completed=False)
            except Exception as cleanup_error:
                unresolved_pids = [candidate_pgid, guardian.pid]
                if isinstance(cleanup_error, CandidateCleanupError):
                    unresolved_pids.extend(cleanup_error.unresolved_candidate_pids)
                raise CandidateCleanupError(
                    "candidate guardian cleanup was not confirmed after startup failure: "
                    f"{cleanup_error}",
                    unresolved_candidate_pids=tuple(dict.fromkeys(unresolved_pids)),
                ) from original
        raise
    finally:
        for descriptor in (control_read, ready_read, ready_write):
            if descriptor != -1:
                os.close(descriptor)


def _finish_candidate_lifetime_guard(
    guardian: subprocess.Popen, control_write: int | None, *, completed: bool,
) -> bool:
    """Disarm after confirmed cleanup, or let EOF make the guard kill the group."""
    if control_write is not None:
        try:
            if completed:
                try:
                    os.write(control_write, b"complete\n")
                except BrokenPipeError:
                    pass
        finally:
            os.close(control_write)
    try:
        _stdout, stderr = guardian.communicate(timeout=_CANDIDATE_GUARD_TIMEOUT)
    except subprocess.TimeoutExpired:
        guardian.kill()
        try:
            _stdout, stderr = guardian.communicate(timeout=_CANDIDATE_GUARD_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise CandidateCleanupError(
                f"could not reap candidate lifetime guardian {guardian.pid}"
            )
    if guardian.returncode != 0:
        detail = (stderr or b"").decode("utf-8", errors="replace").strip()
        raise CandidateCleanupError(
            f"candidate lifetime guardian {guardian.pid} failed with exit code "
            f"{guardian.returncode}" + (f": {detail}" if detail else "")
        )
    return True


def _candidate_completion_status(
    *, completed_at: float, candidate_deadline: float, shared_deadline: Optional[float]
) -> str:
    if completed_at < candidate_deadline:
        return "ok"
    if shared_deadline is not None and shared_deadline <= candidate_deadline:
        return "search_budget_timeout"
    return "timeout"


def run_candidates_isolated(
    *,
    evaluator: Callable[..., Any],
    requests: Sequence[tuple[str, tuple[Any, ...], Mapping[str, Any]]],
    n_jobs: int,
    timeout_seconds: float,
    shared_deadline_monotonic: Optional[float] = None,
    startup_timeout_seconds: float = 60.0,
    event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> list[CandidateOutcome]:
    """Run each request once in a separate process and keep sibling outcomes.

    ``timeout_seconds`` starts when the child has entered its worker target, so
    Python spawn/import time does not consume the estimator evaluation budget.
    Startup itself has a separate bounded deadline. A timed-out worker's POSIX
    process group is terminated before a slot is reused.
    """
    _validate_requests(requests, n_jobs, timeout_seconds, startup_timeout_seconds)
    if shared_deadline_monotonic is not None and (
        not isinstance(shared_deadline_monotonic, (int, float))
        or isinstance(shared_deadline_monotonic, bool)
        or not math.isfinite(shared_deadline_monotonic)
    ):
        raise ValueError("shared_deadline_monotonic must be finite when provided")
    if not requests:
        return []

    context = multiprocessing.get_context("spawn")
    pending_requests = list(requests)
    active: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, CandidateOutcome] = {}

    def emit(event: dict[str, Any]) -> None:
        if event_callback is None:
            return
        try:
            event_callback(event)
        except Exception:
            # Diagnostics must not stop a model evaluation from starting/finishing.
            pass

    def budget_exhausted(uid: str, *, payload_bytes: int = 0,
                         serialization_seconds: float = 0.0) -> None:
        outcome = CandidateOutcome(
            uid=uid,
            status="search_budget_exhausted",
            payload_bytes=payload_bytes,
            serialization_seconds=serialization_seconds,
        )
        outcomes[uid] = outcome
        emit({
            "event": "search_budget_exhausted",
            "uid": uid,
            "payload_bytes": payload_bytes,
            "serialization_seconds": serialization_seconds,
            "timestamp_monotonic": time.monotonic(),
        })

    def start_one(request: tuple[str, tuple[Any, ...], Mapping[str, Any]]) -> None:
        uid, args, kwargs = request
        if shared_deadline_monotonic is not None and time.monotonic() >= shared_deadline_monotonic:
            budget_exhausted(uid)
            return
        serialization_started = time.monotonic()
        payload = cloudpickle.dumps((evaluator, args, dict(kwargs)))
        serialization_seconds = time.monotonic() - serialization_started
        if shared_deadline_monotonic is not None and time.monotonic() >= shared_deadline_monotonic:
            budget_exhausted(
                uid, payload_bytes=len(payload), serialization_seconds=serialization_seconds
            )
            return
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_candidate_process_entry,
            args=(child_connection, payload, uid),
            name=f"fedot-candidate-{uid[:48]}",
        )
        process.daemon = False
        requested_at = time.monotonic()
        try:
            process.start()
        except Exception:
            parent_connection.close()
            child_connection.close()
            raise
        child_connection.close()
        guardian = None
        guardian_control = None
        try:
            if os.name == "posix":
                guardian, guardian_control = _start_candidate_lifetime_guard(process.pid)
        except BaseException as start_error:
            candidate_pid = process.pid
            cleanup_confirmed = _terminate_process_group(process)
            process.join(timeout=0.2)
            parent_connection.close()
            if process.pid is not None and not process.is_alive():
                process.close()
            if not cleanup_confirmed:
                unresolved_pids = [candidate_pid]
                if isinstance(start_error, CandidateCleanupError):
                    unresolved_pids.extend(
                        pid for pid in start_error.unresolved_candidate_pids
                        if pid != candidate_pid
                    )
                raise CandidateCleanupError(
                    f"candidate {uid} could not be cleaned after guardian startup failed",
                    unresolved_candidate_pids=tuple(dict.fromkeys(unresolved_pids)),
                ) from start_error
            if isinstance(start_error, CandidateCleanupError):
                unresolved_pids = tuple(
                    pid for pid in start_error.unresolved_candidate_pids if pid != candidate_pid
                )
                if unresolved_pids:
                    raise CandidateCleanupError(
                        f"candidate {uid} stopped but guardian cleanup remains unresolved",
                        unresolved_candidate_pids=unresolved_pids,
                    ) from start_error
            raise
        active[uid] = {
            "connection": parent_connection,
            "process": process,
            "guardian": guardian,
            "guardian_control": guardian_control,
            "requested_at": requested_at,
            "started_at": None,
            "payload_bytes": len(payload),
            "serialization_seconds": serialization_seconds,
            "deadline_monotonic": None,
            "events": [],
        }
        emit({
            "event": "spawned",
            "uid": uid,
            "pid": process.pid,
            "payload_bytes": len(payload),
            "serialization_seconds": serialization_seconds,
            "timestamp_monotonic": requested_at,
        })
        try:
            # The child waits here until its independent group guardian is
            # ready. If this parent dies first, the closed IPC pipe makes the
            # child exit before user evaluation code can start.
            parent_connection.send_bytes(b"run\n")
        except (BrokenPipeError, EOFError, OSError):
            finish(uid, "start_error", error="candidate parent handshake failed")

    def close_item(item: dict[str, Any]) -> None:
        process = item["process"]
        connection = item["connection"]
        connection.close()
        if process.pid is not None and not process.is_alive():
            process.close()

    def finish(uid: str, status: str, *, value: Any = None, error: Optional[str] = None) -> None:
        item = active[uid]
        process = item["process"]
        worker_pid = process.pid
        started_at = item["started_at"]
        elapsed = max(0.0, time.monotonic() - (started_at or item["requested_at"]))
        process.join(timeout=0.2)
        cleanup_confirmed = _terminate_process_group(process)
        if item["guardian"] is not None:
            guardian_control = item["guardian_control"]
            item["guardian_control"] = None
            try:
                _finish_candidate_lifetime_guard(
                    item["guardian"],
                    guardian_control,
                    completed=cleanup_confirmed,
                )
            except CandidateCleanupError as guard_error:
                cleanup_confirmed = False
                error = f"{error}; {guard_error}" if error else str(guard_error)
        if not cleanup_confirmed:
            outcomes[uid] = CandidateOutcome(
                uid=uid,
                status="cleanup_error",
                error=error,
                elapsed_seconds=elapsed,
                pid=worker_pid,
                payload_bytes=item["payload_bytes"],
                serialization_seconds=item["serialization_seconds"],
                startup_seconds=max(0.0, (started_at or time.monotonic()) - item["requested_at"]),
                cleanup_confirmed=False,
            )
            emit({
                "event": "cleanup_error",
                "uid": uid,
                "pid": worker_pid,
                "elapsed_seconds": elapsed,
                "payload_bytes": item["payload_bytes"],
                "serialization_seconds": item["serialization_seconds"],
                "startup_seconds": max(0.0, (started_at or time.monotonic()) - item["requested_at"]),
                "cleanup_confirmed": False,
                "error": error,
                "timestamp_monotonic": time.monotonic(),
            })
            raise CandidateCleanupError(
                f"could not confirm candidate process group stopped for {uid}; refusing to reuse its slot",
                outcomes=tuple(outcomes.values()),
                unresolved_candidate_pids=(worker_pid,),
            )
        active.pop(uid)
        close_item(item)
        outcome = CandidateOutcome(
            uid=uid,
            status=status,
            value=value,
            error=error,
            elapsed_seconds=elapsed,
            pid=worker_pid,
            payload_bytes=item["payload_bytes"],
            serialization_seconds=item["serialization_seconds"],
            startup_seconds=max(0.0, (started_at or time.monotonic()) - item["requested_at"]),
            cleanup_confirmed=cleanup_confirmed,
        )
        outcomes[uid] = outcome
        fitness = getattr(value, "fitness", None)
        fitness_values = getattr(fitness, "values", None)
        emit({
            "event": "finished" if status == "ok" else status,
            "uid": uid,
            "pid": outcome.pid,
            "elapsed_seconds": elapsed,
            "payload_bytes": outcome.payload_bytes,
            "serialization_seconds": outcome.serialization_seconds,
            "startup_seconds": outcome.startup_seconds,
            "cleanup_confirmed": cleanup_confirmed,
            "fitness_values": list(fitness_values) if fitness_values is not None else None,
            "error": error,
            "timestamp_monotonic": time.monotonic(),
        })

    try:
        while pending_requests or active:
            while pending_requests and len(active) < n_jobs:
                request = pending_requests.pop(0)
                try:
                    start_one(request)
                except CandidateCleanupError:
                    raise
                except Exception:
                    uid = request[0]
                    outcomes[uid] = CandidateOutcome(
                        uid=uid,
                        status="start_error",
                        error=traceback.format_exc(),
                    )
                    emit({
                        "event": "start_error",
                        "uid": uid,
                        "error": outcomes[uid].error,
                        "timestamp_monotonic": time.monotonic(),
                    })

            for uid, item in list(active.items()):
                connection = item["connection"]
                process = item["process"]
                while connection.poll():
                    try:
                        message = cloudpickle.loads(connection.recv_bytes())
                    except Exception:
                        finish(uid, "error", error=traceback.format_exc())
                        break
                    event_type = message.get("event")
                    if event_type == "started":
                        item["started_at"] = float(message.get("timestamp_monotonic", time.monotonic()))
                        if item["started_at"] >= item["requested_at"] + startup_timeout_seconds:
                            finish(
                                uid,
                                "start_timeout",
                                error="candidate worker entered its target after the startup deadline",
                            )
                            break
                        item["deadline_monotonic"] = item["started_at"] + timeout_seconds
                        if shared_deadline_monotonic is not None:
                            item["deadline_monotonic"] = min(
                                item["deadline_monotonic"], shared_deadline_monotonic
                            )
                        item["events"].append(event_type)
                        emit({
                            "event": "started",
                            "uid": uid,
                            "pid": process.pid,
                            "timestamp_monotonic": item["started_at"],
                            "startup_seconds": max(
                                0.0, item["started_at"] - item["requested_at"]
                            ),
                        })
                    elif event_type == "result":
                        completed_at = float(message.get("completed_at", time.monotonic()))
                        status = _candidate_completion_status(
                            completed_at=completed_at,
                            candidate_deadline=item["deadline_monotonic"],
                            shared_deadline=shared_deadline_monotonic,
                        )
                        if status == "ok":
                            finish(uid, "ok", value=message.get("value"))
                        else:
                            finish(uid, status, error="candidate result arrived after its hard deadline")
                        break
                    elif event_type == "error":
                        finish(uid, "error", error=message.get("error"))
                        break

                if uid not in active:
                    continue

                item = active[uid]
                if not process.is_alive() and not item["connection"].poll():
                    finish(uid, "error", error=f"candidate worker exited with code {process.exitcode} before returning an outcome")
                    continue

                now = time.monotonic()
                started_at = item["started_at"]
                if started_at is None:
                    expired = now - item["requested_at"] >= startup_timeout_seconds
                    if expired:
                        finish(uid, "start_timeout", error="candidate worker did not enter its target before startup deadline")
                elif shared_deadline_monotonic is not None and now >= shared_deadline_monotonic:
                    finish(uid, "search_budget_timeout", error="shared FEDOT search budget expired")
                elif now >= item["deadline_monotonic"]:
                    finish(uid, "timeout", error=f"candidate exceeded {timeout_seconds:g}s evaluation deadline")

            if active:
                time.sleep(0.01)
    except CandidateCleanupError as cleanup_error:
        for request in pending_requests:
            uid = request[0]
            outcomes[uid] = CandidateOutcome(
                uid=uid,
                status="not_started_cleanup_failure",
                error="not started because a prior candidate process group could not be confirmed stopped",
            )
            emit({
                "event": "not_started_cleanup_failure",
                "uid": uid,
                "error": outcomes[uid].error,
                "timestamp_monotonic": time.monotonic(),
            })
        unresolved_pids = list(cleanup_error.unresolved_candidate_pids)
        for uid, item in list(active.items()):
            process = item["process"]
            cleanup_confirmed = _terminate_process_group(process)
            if item["guardian"] is not None:
                guardian_control = item["guardian_control"]
                item["guardian_control"] = None
                try:
                    _finish_candidate_lifetime_guard(
                        item["guardian"],
                        guardian_control,
                        completed=cleanup_confirmed,
                    )
                except CandidateCleanupError:
                    cleanup_confirmed = False
            if cleanup_confirmed:
                process.join(timeout=0)
                close_item(item)
                active.pop(uid, None)
            else:
                if process.pid is not None and process.pid not in unresolved_pids:
                    unresolved_pids.append(process.pid)
                outcomes[uid] = CandidateOutcome(
                    uid=uid,
                    status="cleanup_error",
                    error="process group still alive after dispatcher abort cleanup",
                    pid=process.pid,
                    payload_bytes=item["payload_bytes"],
                    serialization_seconds=item["serialization_seconds"],
                    cleanup_confirmed=False,
                )
        raise CandidateCleanupError(
            str(cleanup_error),
            outcomes=tuple(outcomes.values()),
            unresolved_candidate_pids=tuple(unresolved_pids),
        ) from cleanup_error

    return [outcomes[uid] for uid, _args, _kwargs in requests]


def _validate_requests(requests, n_jobs, timeout_seconds, startup_timeout_seconds):
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    for name, value in (("timeout_seconds", timeout_seconds),
                        ("startup_timeout_seconds", startup_timeout_seconds)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    uids = [request[0] for request in requests]
    if len(uids) != len(set(uids)):
        raise ValueError("candidate UIDs must be unique within a population")
    for request in requests:
        if not isinstance(request, tuple) or len(request) != 3:
            raise ValueError("each request must be (uid, args, kwargs)")
        uid, args, kwargs = request
        if not isinstance(uid, str) or not uid:
            raise ValueError("candidate UID must be a non-empty string")
        if not isinstance(args, tuple) or not isinstance(kwargs, Mapping):
            raise ValueError("candidate args must be a tuple and kwargs a mapping")


def _candidate_process_entry(connection, payload, candidate_uid):
    try:
        if os.name == "posix":
            os.setsid()
        if connection.recv_bytes() != b"run\n":
            raise RuntimeError("candidate dispatcher sent an invalid start handshake")
        os.environ["FEDOT_CANDIDATE_UID"] = candidate_uid
        evaluator, args, kwargs = cloudpickle.loads(payload)
        connection.send_bytes(cloudpickle.dumps({
            "event": "started", "timestamp_monotonic": time.monotonic()
        }))
        value = evaluator(*args, **kwargs)
        completed_at = time.monotonic()
        connection.send_bytes(cloudpickle.dumps({
            "event": "result", "value": value, "completed_at": completed_at,
        }))
    except BaseException:
        try:
            connection.send_bytes(cloudpickle.dumps({"event": "error", "error": traceback.format_exc()}))
        except BaseException:
            pass
    finally:
        connection.close()


def _terminate_process_group(process, grace_seconds: float = 0.5) -> bool:
    pid = process.pid
    if pid is None:
        return True
    if os.name == "posix":
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            if process.is_alive():
                process.terminate()
        except OSError:
            try:
                process.terminate()
            except OSError:
                pass
        process.join(timeout=grace_seconds)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            if process.is_alive():
                process.kill()
        except OSError:
            if process.is_alive():
                try:
                    process.kill()
                except OSError:
                    pass
        cleanup_deadline = time.monotonic() + max(grace_seconds, 0.5)
        while time.monotonic() < cleanup_deadline:
            process.join(timeout=0.02)
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                return not process.is_alive()
            except OSError:
                # Permission errors and other failures do not prove that the
                # process group has disappeared.
                return False
        return False
    elif process.is_alive():
        process.terminate()
        process.join(timeout=grace_seconds)
        if process.is_alive():
            process.kill()
            process.join(timeout=grace_seconds)
    return not process.is_alive()
