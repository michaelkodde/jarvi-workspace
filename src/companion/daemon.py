# src/companion/daemon.py
"""
Event-driven daemon loop and MCP-ready query API for the Jarvi Companion.

Watches 00_inbox/ for JSON tickets, drives PipelineFlow execution,
and exposes operational state for HITL approvals.
"""
from __future__ import annotations

import json
import logging
import os
import re
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "ApprovalItem",
    "ApprovalResult",
    "ApprovalsResult",
    "DaemonError",
    "DaemonLoop",
    "DaemonStats",
    "ErrorCode",
    "QueryAPI",
]

logger = logging.getLogger(__name__)

_TICKET_FILENAME_RE = re.compile(r"^ticket_[a-z0-9_]+\.json$")
_MAX_TICKET_SIZE_BYTES = 1024 * 1024  # 1 MB
_MAX_INGEST_PER_CYCLE = 10

SIGBREAK = getattr(signal, "SIGBREAK", signal.SIGINT)


class ErrorCode:
    """Machine-readable error codes for daemon operations."""

    DAEMON_ALREADY_RUNNING = "DAEMON_ALREADY_RUNNING"
    INVALID_TICKET = "INVALID_TICKET"
    TICKET_TOO_LARGE = "TICKET_TOO_LARGE"
    DUPLICATE_TICKET = "DUPLICATE_TICKET"
    INGEST_FAILED = "INGEST_FAILED"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_NOT_PENDING = "JOB_NOT_PENDING"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    IO_ERROR = "IO_ERROR"
    INTERNAL = "INTERNAL"


class DaemonError(Exception):
    """Raised when a daemon operation fails. Chains from the original exception."""

    def __init__(self, message: str, error_code: str = ErrorCode.INTERNAL) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class DaemonStats:
    """Operational statistics for the daemon."""

    is_running: bool
    tickets_ingested: int
    uptime_sec: int
    last_heartbeat: str
    queue_stats: dict[str, int]


@dataclass(frozen=True)
class ApprovalItem:
    """A single job awaiting HITL approval."""

    job_id: str
    job_type: str
    current_step: str
    description: str
    created_at: str


@dataclass(frozen=True)
class ApprovalsResult:
    """Result of a get_pending_approvals query."""

    status: str
    approvals: tuple[ApprovalItem, ...]
    count: int
    error: str | None
    error_code: str | None


@dataclass(frozen=True)
class ApprovalResult:
    """Result of an approve_job or reject_job operation."""

    status: str
    job_id: str | None
    error: str | None
    error_code: str | None


class DaemonLoop:
    """
    Event-driven background service that watches the inbox and drives pipeline execution.

    Args:
        workspace_path: Root of jarvis-workspace.
        pipeline_flow: Existing PipelineFlow instance.
        poll_interval_sec: Seconds between polling cycles (0.5 to 10.0).
        heartbeat_path: Path to write JSON heartbeat.
        lock_file_path: Path to PID lock file.
        queue_depth_threshold: Max queued jobs before backpressure activates.
        install_signals: If True, registers SIGINT/SIGBREAK handlers in start().

    Raises:
        DaemonError: If workspace is invalid, lock is held, or dependencies are invalid.
    """

    def __init__(
        self,
        workspace_path: str | Path,
        pipeline_flow: Any,
        context_injector: Any | None = None,  # <-- ADDED
        poll_interval_sec: float = 2.0,
        heartbeat_path: str | Path | None = None,
        lock_file_path: str | Path | None = None,
        queue_depth_threshold: int = 50,
        install_signals: bool = True,
    ) -> None:
        if not isinstance(workspace_path, (str, Path)):
            raise DaemonError("workspace_path must be a str or Path", ErrorCode.INTERNAL)
        if not (0.1 <= poll_interval_sec <= 10.0):
            raise DaemonError("poll_interval_sec must be between 0.1 and 10.", ErrorCode.INTERNAL)

        self.workspace = Path(workspace_path).resolve()
        self.pipeline_flow = pipeline_flow
        self.context_injector = context_injector  # <-- ADDED
        self.poll_interval_sec = poll_interval_sec
        self.heartbeat_path = Path(heartbeat_path or self.workspace / "logs" / "daemon_heartbeat.json")
        self.lock_file_path = Path(lock_file_path or self.workspace / "logs" / "daemon.lock")
        self.queue_depth_threshold = queue_depth_threshold
        self._install_signals = install_signals

        self.inbox_dir = self.workspace / "00_inbox"
        self.processed_dir = self.inbox_dir / ".processed"
        self.rejected_dir = self.inbox_dir / ".rejected"

        self._validate_and_setup_dirs()

        self._is_running = False
        self._shutdown_requested = False
        self._tickets_ingested = 0
        self._start_time = time.time()
        self._last_heartbeat = ""
        self._lock_file: Any = None

    def _validate_and_setup_dirs(self) -> None:
        """Validate workspace and create required directories."""
        if not self.workspace.is_dir():
            raise DaemonError(f"Workspace does not exist: {self.workspace}", ErrorCode.IO_ERROR)
        
        for dir_path in (self.inbox_dir, self.processed_dir, self.rejected_dir, self.heartbeat_path.parent):
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise DaemonError(f"Failed to create directory {dir_path}: {type(e).__name__}", ErrorCode.IO_ERROR) from e

    def _acquire_lock(self) -> bool:
        """Acquire exclusive lock via OS file locking. Returns True if successful."""
        try:
            self._lock_file = open(self.lock_file_path, "a", encoding="utf-8")
            if os.name == "nt":  # Windows
                import msvcrt
                try:
                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    self._lock_file.close()
                    return False
            else:  # POSIX
                import fcntl
                try:
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    self._lock_file.close()
                    return False
            
            # Write PID for observability only (OS manages the actual lock)
            self._lock_file.seek(0)
            self._lock_file.truncate()
            self._lock_file.write(str(os.getpid()))
            self._lock_file.flush()
            return True
        except OSError:
            return False

    def _release_lock(self) -> None:
        """Release the OS file lock."""
        if hasattr(self, "_lock_file") and self._lock_file:
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()
            except OSError:
                pass

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Signal handler for graceful shutdown."""
        logger.info("Shutdown signal received (%s). Stopping gracefully...", signum)
        self._shutdown_requested = True

    def _write_heartbeat(self) -> None:
        """Write JSON heartbeat to disk."""
        try:
            stats = self.pipeline_flow.get_queue_stats()
            queue_depth = stats.get("queued", 0)
        except Exception:
            queue_depth = -1  # Fallback if stats fail

        self._last_heartbeat = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        heartbeat_data = {
            "timestamp": self._last_heartbeat,
            "uptime_sec": int(time.time() - self._start_time),
            "queue_depth": queue_depth,
            "is_running": self._is_running,
        }
        
        temp_path = self.heartbeat_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(heartbeat_data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.heartbeat_path)
        except OSError as e:
            logger.error("Failed to write heartbeat: %s", type(e).__name__)

    def _check_backpressure(self) -> bool:
        """
        Check if queue depth exceeds threshold.
        Returns True if ingestion should be skipped (backpressure active).
        """
        try:
            stats = self.pipeline_flow.get_queue_stats()
            if stats.get("queued", 0) > self.queue_depth_threshold:
                logger.warning(
                    "Backpressure active: queue depth %d > %d",
                    stats.get("queued", 0),
                    self.queue_depth_threshold,
                )
                return True
        except Exception as e:
            logger.error("Failed to get queue stats for backpressure check: %s", type(e).__name__)
            # Fail open: allow ingestion to proceed if stats cannot be fetched
            
        return False

    def _is_duplicate(self, ticket_id: str) -> bool:
        """Check if ticket_id already exists in JobQueue."""
        try:
            status = self.pipeline_flow.get_job_status(ticket_id)
            return bool(status)  # Returns {} if not found, truthy dict if found
        except Exception:
            return False

    def _move_to_processed(self, file_path: Path) -> None:
        """Atomically move ticket to .processed/."""
        dest = self.processed_dir / file_path.name
        try:
            os.replace(file_path, dest)
        except OSError as e:
            logger.error("Failed to move ticket to processed: %s", type(e).__name__)

    def _move_to_rejected(self, file_path: Path, error_code: str, error_msg: str, ticket_id: str, job_type: str) -> None:
        """Atomically move ticket to .rejected/ with a safe .err sidecar (NO PII)."""
        dest = self.rejected_dir / file_path.name
        err_path = self.rejected_dir / f"{file_path.stem}.err"
        
        err_data = {
            "ticket_id": ticket_id,
            "job_type": job_type,
            "error_code": error_code,
            "error_message": error_msg,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        
        try:
            os.replace(file_path, dest)
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump(err_data, f, indent=2)
        except OSError as e:
            logger.error("Failed to move ticket to rejected: %s", type(e).__name__)

    def _ingest_tickets(self) -> None:
        """Scan inbox, validate, and submit up to MAX_INGEST_PER_CYCLE tickets."""
        if self._check_backpressure():
            return  # Skip ingestion entirely if backpressure is active

        try:
            ticket_files = sorted(self.inbox_dir.glob("ticket_*.json"))
        except OSError as e:
            logger.error("Failed to scan inbox: %s", type(e).__name__)
            return

        ingested_this_cycle = 0
        for file_path in ticket_files:
            if ingested_this_cycle >= _MAX_INGEST_PER_CYCLE:
                break

            # Validate filename
            if not _TICKET_FILENAME_RE.match(file_path.name):
                self._move_to_rejected(file_path, ErrorCode.INVALID_TICKET, "Invalid filename", "unknown", "unknown")
                continue

            # Validate size
            try:
                if file_path.stat().st_size > _MAX_TICKET_SIZE_BYTES:
                    self._move_to_rejected(file_path, ErrorCode.TICKET_TOO_LARGE, "File exceeds 1MB", "unknown", "unknown")
                    continue
            except OSError as e:
                logger.error("Failed to stat file %s: %s", file_path.name, type(e).__name__)
                continue

            # Parse and validate content
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    ticket_data = json.load(f)
                
                ticket_id = str(ticket_data.get("ticket_id", ""))
                job_type = str(ticket_data.get("job_type", ""))
                
                if not ticket_id or not job_type:
                    self._move_to_rejected(file_path, ErrorCode.INVALID_TICKET, "Missing ticket_id or job_type", "unknown", "unknown")
                    continue

            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._move_to_rejected(file_path, ErrorCode.INVALID_TICKET, f"Invalid JSON: {type(e).__name__}", "unknown", "unknown")
                continue
            except OSError as e:
                logger.error("Failed to read file %s: %s", file_path.name, type(e).__name__)
                continue

            # Check for duplicate
            if self._is_duplicate(ticket_id):
                logger.info("Duplicate ticket detected, moving to processed: %s", ticket_id)
                self._move_to_processed(file_path)
                continue

            # Submit to PipelineFlow
            try:
                # 1. Enrich parameters with context (if injector is available)
                final_parameters = ticket_data.get("parameters", {})
                if self.context_injector is not None:
                    try:
                        context_result = self.context_injector.inject_context(job_type, final_parameters)
                        if context_result.status == "success" and context_result.enriched_parameters:
                            final_parameters = context_result.enriched_parameters
                    except Exception as e:
                        # Graceful degradation: log warning, proceed with original parameters
                        logger.warning("Context injection failed for %s: %s. Proceeding without context.", ticket_id, type(e).__name__)

                # 2. Submit the (potentially enriched) job
                self.pipeline_flow.submit_job(
                    job_type=job_type,
                    parameters=final_parameters,
                    priority=ticket_data.get("priority", 5),
                    description=ticket_data.get("description", ""),
                )
                self._move_to_processed(file_path)
                self._tickets_ingested += 1
                ingested_this_cycle += 1
                logger.info("Ticket ingested: %s (type=%s)", ticket_id, job_type)
            except Exception as e:
                logger.error("Failed to submit job %s: %s", ticket_id, type(e).__name__)
                self._move_to_rejected(file_path, ErrorCode.INGEST_FAILED, str(e), ticket_id, job_type)

    def start(self) -> None:
        """Start the blocking daemon event loop."""
        if not self._acquire_lock():
            raise DaemonError("Another daemon instance is already running.", ErrorCode.DAEMON_ALREADY_RUNNING)

        original_sigint = None
        original_sigbreak = None
        
        if self._install_signals:
            original_sigint = signal.signal(signal.SIGINT, self._handle_shutdown)
            original_sigbreak = signal.signal(SIGBREAK, self._handle_shutdown)

        self._is_running = True
        logger.info("Daemon started.")

        try:
            while not self._shutdown_requested:
                self._write_heartbeat()
                self._ingest_tickets()
                
                try:
                    self.pipeline_flow.process_next_job()
                except Exception as e:
                    logger.error("Error in process_next_job: %s", type(e).__name__)

                # Sleep in small chunks to remain responsive to shutdown signals
                sleep_remaining = self.poll_interval_sec
                while sleep_remaining > 0 and not self._shutdown_requested:
                    time.sleep(min(0.1, sleep_remaining))
                    sleep_remaining -= 0.1

        finally:
            self._is_running = False
            self._write_heartbeat()
            self._release_lock()
            
            # Restore original signal handlers if we installed them
            if self._install_signals:
                if original_sigint is not None:
                    signal.signal(signal.SIGINT, original_sigint)
                if original_sigbreak is not None:
                    signal.signal(SIGBREAK, original_sigbreak)
                    
            logger.info("Daemon stopped.")

    def stop(self, timeout_sec: float = 30.0) -> None:
        """Signal graceful shutdown and wait for current cycle to complete."""
        self._shutdown_requested = True
        start_wait = time.time()
        while self._is_running and (time.time() - start_wait) < timeout_sec:
            time.sleep(0.1)
        
        if self._is_running:
            logger.warning("Shutdown timeout reached. Forcing exit.")


class QueryAPI:
    """
    MCP-ready query interface for HITL approvals and operational stats.
    Thread-safe and read-only (except for approve/reject actions).
    """

    def __init__(self, pipeline_flow: Any) -> None:
        if not hasattr(pipeline_flow, "list_jobs"):
            raise DaemonError("pipeline_flow must have list_jobs method", ErrorCode.INTERNAL)
        self.pipeline_flow = pipeline_flow

    def get_pending_approvals(self) -> ApprovalsResult:
        """Retrieve all jobs currently in pending_review status."""
        try:
            jobs = self.pipeline_flow.list_jobs(status="pending_review", limit=100)
            approvals = tuple(
                ApprovalItem(
                    job_id=str(j.get("job_id", "")),
                    job_type=str(j.get("job_type", "")),
                    current_step=str(j.get("current_step", "")),
                    description=str(j.get("description", "")),
                    created_at=str(j.get("created_at", "")),
                )
                for j in jobs
            )
            return ApprovalsResult(
                status="success",
                approvals=approvals,
                count=len(approvals),
                error=None,
                error_code=None,
            )
        except Exception as e:
            logger.error("Failed to get pending approvals: %s", type(e).__name__)
            return ApprovalsResult(
                status="error",
                approvals=(),
                count=0,
                error=str(e),
                error_code=ErrorCode.INTERNAL,
            )

    def approve_job(self, job_id: str, feedback: str | None = None) -> ApprovalResult:
        """Approve a pending job, resuming the pipeline."""
        if not hasattr(self.pipeline_flow, "approve_job"):
            return ApprovalResult(
                status="error",
                job_id=job_id,
                error="PipelineFlow does not support approve_job",
                error_code=ErrorCode.UNSUPPORTED_OPERATION,
            )
        try:
            self.pipeline_flow.approve_job(job_id, feedback=feedback)
            return ApprovalResult(status="success", job_id=job_id, error=None, error_code=None)
        except Exception as e:
            logger.error("Failed to approve job %s: %s", job_id, type(e).__name__)
            return ApprovalResult(
                status="error",
                job_id=job_id,
                error=str(e),
                error_code=ErrorCode.JOB_NOT_PENDING if "pending" in str(e).lower() else ErrorCode.INTERNAL,
            )

    def reject_job(self, job_id: str, reason: str) -> ApprovalResult:
        """Reject a pending job, canceling the pipeline."""
        if not hasattr(self.pipeline_flow, "reject_job"):
            return ApprovalResult(
                status="error",
                job_id=job_id,
                error="PipelineFlow does not support reject_job",
                error_code=ErrorCode.UNSUPPORTED_OPERATION,
            )
        try:
            self.pipeline_flow.reject_job(job_id, reason=reason)
            return ApprovalResult(status="success", job_id=job_id, error=None, error_code=None)
        except Exception as e:
            logger.error("Failed to reject job %s: %s", job_id, type(e).__name__)
            return ApprovalResult(
                status="error",
                job_id=job_id,
                error=str(e),
                error_code=ErrorCode.JOB_NOT_PENDING if "pending" in str(e).lower() else ErrorCode.INTERNAL,
            )

    def get_stats(self) -> DaemonStats:
        """Retrieve current daemon operational statistics."""
        try:
            queue_stats = self.pipeline_flow.get_queue_stats()
            return DaemonStats(
                is_running=False,  # QueryAPI is stateless
                tickets_ingested=0,
                uptime_sec=0,
                last_heartbeat="",
                queue_stats=queue_stats,
            )
        except Exception as e:
            logger.error("Failed to get stats: %s", type(e).__name__)
            raise DaemonError(f"Failed to get stats: {type(e).__name__}", ErrorCode.INTERNAL) from e