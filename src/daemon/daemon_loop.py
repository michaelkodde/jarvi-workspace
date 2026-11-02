"""
Daemon Loop: The central nervous system of Jarvi.
Watches the inbox, validates tickets, drives pipeline execution, and maintains system heartbeat.
"""
import json
import logging
import os
import re
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .exceptions import DaemonError, ErrorCode
from .lock_manager import LockManager
from .schemas import DaemonStats, Ticket

logger = logging.getLogger(__name__)

# Constants
MAX_TICKETS_PER_CYCLE = 10
MAX_TICKET_SIZE_BYTES = 1048576  # 1MB
SHUTDOWN_TIMEOUT_SEC = 30.0
TICKET_FILENAME_REGEX = re.compile(r"^ticket_[a-z0-9_]+\.json$")


class DaemonLoop:
    """
    Background daemon that watches `00_inbox/` for JSON tickets, 
    submits them to PipelineFlow, and drives execution.
    """

    def __init__(
        self,
        workspace_path: Path,
        pipeline_flow: Any,
        poll_interval_sec: float = 2.0,
        queue_depth_threshold: int = 50,
        enabled: bool = True,
        resource_watchdog: Optional[Any] = None,
        notification_callback: Optional[Any] = None,
        context_injector: Optional[Any] = None,  # NEW: Memory enrichment
    ):
        if not (0.5 <= poll_interval_sec <= 10.0):
            raise DaemonError("poll_interval_sec must be between 0.5 and 10.0")

        # Validate workspace exists
        workspace_path = Path(workspace_path).resolve()
        if not workspace_path.exists():
            raise DaemonError(f"Workspace does not exist: {workspace_path}", ErrorCode.IO_ERROR)
            
        self.workspace_path = workspace_path
        self.pipeline_flow = pipeline_flow
        self.poll_interval_sec = poll_interval_sec
        self.queue_depth_threshold = queue_depth_threshold
        self.enabled = enabled
        self.resource_watchdog = resource_watchdog
        self.notification_callback = notification_callback
        self.context_injector = context_injector  # NEW

        # Paths
        self.inbox_path = self.workspace_path / "00_inbox"
        self.processed_path = self.inbox_path / ".processed"
        self.rejected_path = self.inbox_path / ".rejected"
        self.logs_path = self.workspace_path / "logs"
        
        self.heartbeat_path = self.logs_path / "daemon_heartbeat.json"
        self.lock_file_path = self.logs_path / "daemon.lock"

        # State
        self._is_running = False
        self._shutdown_requested = False
        self._in_cycle = False
        self._tickets_ingested = 0
        self._start_time = 0.0
        self._lock_manager = LockManager(self.lock_file_path)

        # Setup directories
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Ensures all required daemon directories exist."""
        for path in [self.inbox_path, self.processed_path, self.rejected_path, self.logs_path]:
            path.mkdir(parents=True, exist_ok=True)

    def _notify_user(self, message: str, priority: str = "info") -> None:
        """Proactively reaches out to the user when something important happens."""
        if self.notification_callback is not None:
            try:
                self.notification_callback(message, priority)
            except Exception as e:
                logger.debug("Notification callback failed: %s", e)

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Signal handler for graceful shutdown."""
        logger.info("Shutdown signal received (%d). Initiating graceful shutdown...", signum)
        self._shutdown_requested = True

    def _write_heartbeat(self) -> None:
        """Atomically writes the current daemon state to the heartbeat JSON file."""
        uptime = int(time.time() - self._start_time) if self._start_time > 0 else 0
        
        queue_stats = {}
        try:
            queue_stats = self.pipeline_flow.get_queue_stats()
        except Exception as e:
            logger.warning("Failed to get queue stats for heartbeat: %s", type(e).__name__)

        heartbeat = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_sec": uptime,
            "queue_depth": queue_stats.get("queued", 0),
            "is_running": self._is_running,
            "tickets_ingested": self._tickets_ingested,
        }

        try:
            temp_path = self.heartbeat_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(heartbeat, f, indent=2)
            temp_path.replace(self.heartbeat_path)
        except Exception as e:
            logger.error("Failed to write heartbeat file: %s", type(e).__name__)

    def _reject_ticket(self, file_path: Path, error_code: ErrorCode, error_message: str, ticket_id: str = "unknown") -> None:
        """Moves a failed ticket to the rejected folder with an .err sidecar."""
        try:
            err_data = {
                "error_code": error_code.value,
                "error_message": error_message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "original_ticket_id": ticket_id,
            }
            err_path = self.rejected_path / f"{file_path.stem}.err"
            
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump(err_data, f, indent=2)
                
            file_path.replace(self.rejected_path / file_path.name)
            logger.warning("Ticket rejected [%s]: %s (ID: %s)", error_code.value, error_message, ticket_id)
        except Exception as e:
            logger.critical("Failed to move ticket to rejected folder: %s", type(e).__name__)

    def _sanitize_parameters(self, params: Dict[str, Any]) -> bool:
        """Recursively checks for path traversal attempts in parameters."""
        if isinstance(params, dict):
            return all(self._sanitize_parameters(v) for v in params.values())
        elif isinstance(params, list):
            return all(self._sanitize_parameters(v) for v in params)
        elif isinstance(params, str):
            if ".." in params or params.startswith(("/", "\\\\")):
                return False
        return True

    def _process_ticket_file(self, file_path: Path) -> None:
        """Validates, submits, and archives a single ticket file."""
        ticket_id = "unknown"
        
        try:
            # 1. Filename Regex Enforcement
            if not TICKET_FILENAME_REGEX.match(file_path.name):
                self._reject_ticket(file_path, ErrorCode.INVALID_TICKET, "Invalid filename format")
                return

            # 2. Filesystem Idempotency Check
            if (self.processed_path / file_path.name).exists() or (self.rejected_path / file_path.name).exists():
                logger.debug("Ticket already processed/rejected, skipping (ID inferred from filename)")
                return

            # 3. Size Check
            if file_path.stat().st_size > MAX_TICKET_SIZE_BYTES:
                self._reject_ticket(file_path, ErrorCode.TICKET_TOO_LARGE, f"File exceeds {MAX_TICKET_SIZE_BYTES} bytes")
                return

            # 4. Parse JSON
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                self._reject_ticket(file_path, ErrorCode.INVALID_TICKET, f"Invalid JSON: {e}")
                return

            ticket_id = data.get("ticket_id", "unknown")

            # 5. Schema Validation
            try:
                ticket = Ticket(
                    ticket_id=data["ticket_id"],
                    job_type=data["job_type"],
                    parameters=data.get("parameters", {}),
                    priority=data.get("priority", 5),
                    description=data.get("description"),
                )
            except (KeyError, ValueError) as e:
                self._reject_ticket(file_path, ErrorCode.INVALID_TICKET, str(e), ticket_id)
                return

            # 6. Parameter Sanitization (Security)
            if not self._sanitize_parameters(ticket.parameters):
                self._reject_ticket(file_path, ErrorCode.INVALID_TICKET, "Parameters contain forbidden patterns (e.g., path traversal)", ticket_id)
                return

            # 7. Submit to PipelineFlow (with optional context enrichment)
            try:
                # Enrich parameters with context (if injector is available)
                final_parameters = ticket.parameters
                if self.context_injector is not None:
                    try:
                        context_result = self.context_injector.inject_context(ticket.job_type, ticket.parameters)
                        if context_result.status == "success" and context_result.enriched_parameters:
                            final_parameters = context_result.enriched_parameters
                            logger.debug("Context injected for %s (%d memories used)", 
                                        ticket.ticket_id, context_result.memories_used)
                    except Exception as e:
                        # Graceful degradation: log warning, proceed with original parameters
                        logger.warning("Context injection failed for %s: %s. Proceeding without context.", 
                                      ticket.ticket_id, type(e).__name__)
                
                trace_desc = ticket.description or f"Inbox ticket: {ticket.ticket_id}"
                
                self.pipeline_flow.submit_job(
                    job_type=ticket.job_type,
                    parameters=final_parameters,
                    priority=ticket.priority,
                    description=trace_desc,
                )
                
                logger.info("Ticket ingested successfully (ID: %s, Type: %s)", ticket.ticket_id, ticket.job_type)
                self._tickets_ingested += 1
                
                # 8. Atomic Move to Processed
                file_path.replace(self.processed_path / file_path.name)
                
            except Exception as e:
                self._reject_ticket(file_path, ErrorCode.INGEST_FAILED, str(e), ticket.ticket_id)

        except Exception as e:
            logger.error("Unexpected error processing ticket file %s: %s", file_path.name, type(e).__name__)
            self._reject_ticket(file_path, ErrorCode.INTERNAL, "Unexpected processing error", ticket_id)

    def _ingest_cycle(self) -> None:
        """Executes a single ingestion and execution cycle."""
        self._in_cycle = True
        try:
            # 1. Verify Lock
            if hasattr(self._lock_manager, "is_locked"):
                is_locked = self._lock_manager.is_locked()
            else:
                is_locked = self.lock_file_path.exists()
                
            if not is_locked:
                logger.critical("Lock file lost during execution. Shutting down.")
                self._shutdown_requested = True
                return

            # 2. Write Heartbeat
            self._write_heartbeat()

            # 3. Check Resource Watchdog
            if self.resource_watchdog is not None:
                try:
                    if not self.resource_watchdog.is_safe_to_proceed():
                        logger.warning("Resource watchdog signaled unsafe conditions. Skipping ingestion cycle.")
                        return
                except Exception as e:
                    logger.warning("Resource watchdog check failed: %s", type(e).__name__)

            # 4. Check Backpressure
            try:
                stats = self.pipeline_flow.get_queue_stats()
                if stats.get("queued", 0) > self.queue_depth_threshold:
                    logger.warning("Backpressure active: Queue depth (%d) exceeds threshold (%d). Skipping ingestion.", 
                                   stats.get("queued", 0), self.queue_depth_threshold)
                    return
            except Exception as e:
                logger.warning("Failed to check backpressure: %s", type(e).__name__)

            # 5. Scan Inbox
            try:
                ticket_files = sorted(
                    [f for f in self.inbox_path.glob("ticket_*.json") if f.is_file()],
                    key=lambda x: x.stat().st_mtime
                )[:MAX_TICKETS_PER_CYCLE]
            except Exception as e:
                logger.error("Failed to scan inbox directory: %s", type(e).__name__)
                return

            # 6. Process Each Ticket
            for file_path in ticket_files:
                if self._shutdown_requested:
                    break
                self._process_ticket_file(file_path)

            # 7. Advance Pipeline
            if not self._shutdown_requested:
                try:
                    self.pipeline_flow.process_next_job()
                except Exception as e:
                    logger.error("Pipeline execution step failed: %s", type(e).__name__)
                    
        finally:
            self._in_cycle = False

    def start(self) -> None:
        """Starts the blocking daemon event loop. Handles graceful shutdown on SIGINT/SIGBREAK."""
        if not self.enabled:
            logger.info("Daemon loop is disabled via configuration. Exiting.")
            return
            
        if self._is_running:
            return

        self._lock_manager.acquire()
        
        self._is_running = True
        self._shutdown_requested = False
        self._start_time = time.time()
        logger.info("Daemon Loop started (PID: %d)", os.getpid())

        # Only register signals if we are in the main thread (e.g., not in a test thread)
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._handle_shutdown)
            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, self._handle_shutdown)

        try:
            while not self._shutdown_requested:
                self._ingest_cycle()
                if not self._shutdown_requested:
                    time.sleep(self.poll_interval_sec)
        except Exception as e:
            logger.critical("Fatal error in daemon loop: %s", type(e).__name__, exc_info=True)
        finally:
            self.stop()

    def stop(self, timeout_sec: float = SHUTDOWN_TIMEOUT_SEC) -> None:
        """Initiates graceful shutdown. Finishes current cycle within timeout."""
        if not self._is_running:
            return

        logger.info("Initiating graceful shutdown (timeout: %ds)...", timeout_sec)
        self._shutdown_requested = True
        self._is_running = False

        # Wait for current cycle to finish
        start_wait = time.time()
        while time.time() - start_wait < timeout_sec:
            if not self._in_cycle:
                break
            time.sleep(0.1)

        self._lock_manager.release()
        
        # Write final heartbeat to reflect stopped state
        self._write_heartbeat()
        
        logger.info("Daemon Loop stopped gracefully. Total ingested: %d", self._tickets_ingested)

    def is_running(self) -> bool:
        """Returns True if the daemon is currently running."""
        return self._is_running

    def get_stats(self) -> DaemonStats:
        """Returns current operational statistics."""
        uptime = int(time.time() - self._start_time) if self._start_time > 0 else 0
        
        queue_stats = {}
        try:
            queue_stats = self.pipeline_flow.get_queue_stats()
        except Exception:
            pass

        return DaemonStats(
            is_running=self._is_running,
            tickets_ingested=self._tickets_ingested,
            uptime_sec=uptime,
            last_heartbeat=datetime.now(timezone.utc).isoformat(),
            queue_stats=queue_stats,
        )