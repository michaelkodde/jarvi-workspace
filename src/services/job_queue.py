"""
Job Queue: SQLite-backed durable job queue with atomic claims and audit log.

Provides crash-safe job management for the PipelineFlow orchestrator.
Uses WAL mode for concurrent access and short-lived connections.
"""
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["JobQueue", "JobRecord", "QueueError"]


class QueueError(Exception):
    """Raised when queue operations fail."""
    pass


@dataclass(frozen=True)
class JobRecord:
    """Immutable record of a job."""
    job_id: str
    job_type: str
    status: str
    parameters: Dict[str, Any]
    priority: int
    description: str
    created_by: str
    created_at: str
    updated_at: str
    current_step: str
    completed_steps: Tuple[str, ...]
    artifacts: Dict[str, str]
    retry_count: int
    available_at: str
    error_message: Optional[str] = None


class JobQueue:
    """
    SQLite-backed job queue with atomic claims and audit log.
    
    Uses WAL mode for concurrent access and short-lived connections.
    
    Args:
        db_path: Path to the SQLite database file.
    
    Raises:
        QueueError: If database initialization fails.
    """

    def __init__(self, db_path: str = "dashboard/job_queue.db") -> None:
        self.db_path = Path(db_path)
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise QueueError(f"Failed to create database directory: {e}") from e
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the jobs and audit_log tables."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        job_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        parameters TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        description TEXT NOT NULL,
                        created_by TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        current_step TEXT NOT NULL,
                        completed_steps TEXT NOT NULL,
                        artifacts TEXT NOT NULL,
                        retry_count INTEGER NOT NULL,
                        available_at TEXT NOT NULL,
                        error_message TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        details TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_jobs_claim 
                    ON jobs(status, available_at, priority, created_at)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_job ON audit_log(job_id)
                """)
                conn.commit()
        except sqlite3.Error as e:
            raise QueueError(f"Failed to initialize queue database: {e}") from e

    def _now_iso(self) -> str:
        """Return current UTC time as ISO8601 string."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _row_to_record(self, row: Tuple) -> JobRecord:
        """Convert a DB row to a JobRecord."""
        return JobRecord(
            job_id=row[0],
            job_type=row[1],
            status=row[2],
            parameters=json.loads(row[3]),
            priority=row[4],
            description=row[5],
            created_by=row[6],
            created_at=row[7],
            updated_at=row[8],
            current_step=row[9],
            completed_steps=tuple(json.loads(row[10])),
            artifacts=json.loads(row[11]),
            retry_count=row[12],
            available_at=row[13],
            error_message=row[14],
        )

    def enqueue(
        self,
        job_id: str,
        job_type: str,
        parameters: Dict[str, Any],
        priority: int = 3,
        description: str = "",
        created_by: str = "user",
    ) -> None:
        """
        Enqueue a new job.
        
        Args:
            job_id: Unique job identifier.
            job_type: Pipeline type to execute.
            parameters: Job parameters.
            priority: Priority (higher = sooner).
            description: Human-readable description.
            created_by: Who created the job.
        
        Raises:
            QueueError: If insert fails.
            ValueError: If job_id is empty.
        """
        if not job_id:
            raise ValueError("job_id cannot be empty")
        
        now = self._now_iso()
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    INSERT INTO jobs
                    (job_id, job_type, status, parameters, priority, description, created_by,
                     created_at, updated_at, current_step, completed_steps, artifacts,
                     retry_count, available_at, error_message)
                    VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, '', '[]', '{}', 0, ?, NULL)
                """, (
                    job_id, job_type, json.dumps(parameters), priority, description,
                    created_by, now, now, now,
                ))
                conn.commit()
        except sqlite3.Error as e:
            raise QueueError(f"Failed to enqueue job: {e}") from e

    def claim_next(self) -> Optional[JobRecord]:
        """
        Atomically claim the highest-priority eligible queued job.
        
        Uses BEGIN IMMEDIATE to prevent concurrent claims across processes.
        Eligible: status='queued' AND available_at <= now.
        Order: priority DESC, created_at ASC.
        
        Returns:
            JobRecord if a job was claimed (with status='running'), None otherwise.
        
        Raises:
            QueueError: If claim fails.
        """
        now = self._now_iso()
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("BEGIN IMMEDIATE;")
                cursor = conn.execute("""
                    SELECT job_id, job_type, status, parameters, priority, description,
                           created_by, created_at, updated_at, current_step, completed_steps,
                           artifacts, retry_count, available_at, error_message
                    FROM jobs
                    WHERE status = 'queued' AND available_at <= ?
                    ORDER BY priority DESC, created_at ASC
                    LIMIT 1
                """, (now,))
                row = cursor.fetchone()
                if row is None:
                    conn.execute("COMMIT;")
                    return None
                job_id = row[0]
                conn.execute("""
                    UPDATE jobs SET status = 'running', updated_at = ?
                    WHERE job_id = ?
                """, (now, job_id))
                conn.execute("COMMIT;")
                
                # Re-fetch to get the updated status
                cursor = conn.execute("""
                    SELECT job_id, job_type, status, parameters, priority, description,
                           created_by, created_at, updated_at, current_step, completed_steps,
                           artifacts, retry_count, available_at, error_message
                    FROM jobs WHERE job_id = ?
                """, (job_id,))
                updated_row = cursor.fetchone()
                return self._row_to_record(updated_row)
        except sqlite3.Error as e:
            raise QueueError(f"Failed to claim next job: {e}") from e

    def complete_step(
        self,
        job_id: str,
        step_name: str,
        artifacts: Dict[str, str],
        completed_steps: Tuple[str, ...],
        elapsed_ms: int = 0,
    ) -> None:
        """
        Atomically persist artifact and completed_steps for a step.
        
        Both are written in one transaction to ensure crash consistency.
        
        Args:
            job_id: Job identifier.
            step_name: Name of the completed step.
            artifacts: Updated artifacts dict.
            completed_steps: Updated completed_steps tuple.
            elapsed_ms: Time taken for the step.
        
        Raises:
            QueueError: If update fails.
        """
        now = self._now_iso()
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    UPDATE jobs
                    SET artifacts = ?, completed_steps = ?, current_step = ?,
                        updated_at = ?
                    WHERE job_id = ?
                """, (
                    json.dumps(artifacts),
                    json.dumps(list(completed_steps)),
                    step_name,
                    now,
                    job_id,
                ))
                conn.commit()
                self.append_audit(job_id, "step_complete", {
                    "step": step_name,
                    "elapsed_ms": elapsed_ms,
                })
        except sqlite3.Error as e:
            raise QueueError(f"Failed to complete step: {e}") from e

    def update_status(
        self,
        job_id: str,
        status: str,
        error_message: Optional[str] = None,
        retry_count: Optional[int] = None,
        available_at: Optional[str] = None,
        current_step: Optional[str] = None,
        completed_steps: Optional[Tuple[str, ...]] = None,
    ) -> None:
        """
        Update job status and optional fields.
        
        Args:
            job_id: Job identifier.
            status: New status.
            error_message: Optional error message.
            retry_count: Optional retry count update.
            available_at: Optional available_at update.
            current_step: Optional current_step update.
            completed_steps: Optional completed_steps update.
        
        Raises:
            QueueError: If update fails.
        """
        now = self._now_iso()
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                updates = ["status = ?", "updated_at = ?"]
                values: List[Any] = [status, now]
                if error_message is not None:
                    updates.append("error_message = ?")
                    values.append(error_message)
                if retry_count is not None:
                    updates.append("retry_count = ?")
                    values.append(retry_count)
                if available_at is not None:
                    updates.append("available_at = ?")
                    values.append(available_at)
                if current_step is not None:
                    updates.append("current_step = ?")
                    values.append(current_step)
                if completed_steps is not None:
                    updates.append("completed_steps = ?")
                    values.append(json.dumps(list(completed_steps)))
                values.append(job_id)
                conn.execute(f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?", values)
                conn.commit()
                self.append_audit(job_id, "status_change", {
                    "to": status,
                    "error_message": error_message,
                })
        except sqlite3.Error as e:
            raise QueueError(f"Failed to update status: {e}") from e

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """
        Retrieve a job by ID.
        
        Args:
            job_id: Job identifier.
        
        Returns:
            JobRecord if found, None otherwise.
        
        Raises:
            QueueError: If query fails.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.execute("""
                    SELECT job_id, job_type, status, parameters, priority, description,
                           created_by, created_at, updated_at, current_step, completed_steps,
                           artifacts, retry_count, available_at, error_message
                    FROM jobs WHERE job_id = ?
                """, (job_id,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_record(row)
                return None
        except sqlite3.Error as e:
            raise QueueError(f"Failed to get job: {e}") from e

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[JobRecord]:
        """
        List jobs with optional status filter.
        
        Args:
            status: Optional status filter.
            limit: Max results.
            offset: Offset for pagination.
        
        Returns:
            List of JobRecords.
        
        Raises:
            QueueError: If query fails.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                if status:
                    cursor = conn.execute("""
                        SELECT job_id, job_type, status, parameters, priority, description,
                               created_by, created_at, updated_at, current_step, completed_steps,
                               artifacts, retry_count, available_at, error_message
                        FROM jobs WHERE status = ?
                        ORDER BY created_at DESC
                        LIMIT ? OFFSET ?
                    """, (status, limit, offset))
                else:
                    cursor = conn.execute("""
                        SELECT job_id, job_type, status, parameters, priority, description,
                               created_by, created_at, updated_at, current_step, completed_steps,
                               artifacts, retry_count, available_at, error_message
                        FROM jobs
                        ORDER BY created_at DESC
                        LIMIT ? OFFSET ?
                    """, (limit, offset))
                return [self._row_to_record(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise QueueError(f"Failed to list jobs: {e}") from e

    def get_stats(self) -> Dict[str, int]:
        """
        Get counts by status.
        
        Returns:
            Dict mapping status to count.
        
        Raises:
            QueueError: If query fails.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.execute("""
                    SELECT status, COUNT(*) FROM jobs GROUP BY status
                """)
                return {row[0]: row[1] for row in cursor.fetchall()}
        except sqlite3.Error as e:
            raise QueueError(f"Failed to get stats: {e}") from e

    def recover_orphans(self) -> int:
        """
        Reset any running jobs back to queued (crash recovery).
        
        Does NOT clear artifacts or completed_steps.
        
        **Limitation:** This assumes single-process execution. If multiple processes
        are running, a legitimately running job may be incorrectly reset. Phase 2
        should implement distributed locking.
        
        Returns:
            Number of jobs recovered.
        
        Raises:
            QueueError: If update fails.
        """
        now = self._now_iso()
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                cursor = conn.execute("SELECT job_id FROM jobs WHERE status = 'running'")
                job_ids = [row[0] for row in cursor.fetchall()]
                conn.execute("""
                    UPDATE jobs SET status = 'queued', updated_at = ?
                    WHERE status = 'running'
                """, (now,))
                conn.commit()
                for job_id in job_ids:
                    self.append_audit(job_id, "recovered", {})
                return len(job_ids)
        except sqlite3.Error as e:
            raise QueueError(f"Failed to recover orphans: {e}") from e

    def append_audit(
        self,
        job_id: str,
        event_type: str,
        details: Dict[str, Any],
    ) -> None:
        """
        Append an audit log entry.
        
        Args:
            job_id: Job identifier.
            event_type: Type of event.
            details: Event details (JSON-serializable).
        
        Raises:
            QueueError: If insert fails.
        """
        now = self._now_iso()
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    INSERT INTO audit_log (job_id, timestamp, event_type, details)
                    VALUES (?, ?, ?, ?)
                """, (job_id, now, event_type, json.dumps(details, default=str)))
                conn.commit()
        except sqlite3.Error as e:
            raise QueueError(f"Failed to append audit: {e}") from e

    def get_audit_log(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Get audit log for a job.
        
        Args:
            job_id: Job identifier.
        
        Returns:
            List of audit entries.
        
        Raises:
            QueueError: If query fails.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, event_type, details
                    FROM audit_log
                    WHERE job_id = ?
                    ORDER BY id ASC
                """, (job_id,))
                return [
                    {
                        "timestamp": row[0],
                        "event_type": row[1],
                        "details": json.loads(row[2]) if row[2] else {},
                    }
                    for row in cursor.fetchall()
                ]
        except sqlite3.Error as e:
            raise QueueError(f"Failed to get audit log: {e}") from e