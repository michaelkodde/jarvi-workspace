"""
PipelineFlow: Deterministic workflow engine for agent orchestration.

Manages job execution across agents with crash recovery, classified retries,
and HITL gates. Does NOT inherit BaseAgent and does NOT call the LLM directly.
"""
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.services.job_queue import JobQueue, JobRecord, QueueError
from src.services.step_adapters import (
    AdapterError,
    get_adapter,
    guard_creator_artifact,
    guard_scout_artifact,
    map_publisher_status,
)

logger = logging.getLogger(__name__)

__all__ = ["PipelineFlow", "PipelineError"]


class PipelineError(Exception):
    """Raised for orchestration failures."""
    pass


# Error classification
PERMANENT_ERRORS = (
    PipelineError,
    AdapterError,
    KeyError,
    ValueError,
    json.JSONDecodeError,
)

_TRANSIENT_ERROR_NAMES = ("ConnectionError", "TimeoutError", "OSError")


def _is_transient_error(exc: Exception) -> bool:
    """Check if an exception is transient (retryable)."""
    exc_type_name = type(exc).__name__
    if exc_type_name in _TRANSIENT_ERROR_NAMES:
        return True
    msg = str(exc).lower()
    if any(kw in msg for kw in ("5xx", "429", "timeout", "connection")):
        return True
    return False


def _import_agent_exceptions() -> tuple:
    """Lazily import agent-specific exceptions."""
    exceptions = []
    try:
        from src.agents.scout import ScoutError
        exceptions.append(ScoutError)
    except (ImportError, AttributeError):
        pass
    try:
        from src.agents.creator import ContentRequestError
        exceptions.append(ContentRequestError)
    except (ImportError, AttributeError):
        pass
    try:
        from src.agents.publisher import PublicationError
        exceptions.append(PublicationError)
    except (ImportError, AttributeError):
        pass
    try:
        from src.agents.coder import CoderError
        exceptions.append(CoderError)
    except (ImportError, AttributeError):
        pass
    return tuple(exceptions)


class PipelineFlow:
    """
    Deterministic workflow engine for agent orchestration.
    
    Manages job execution with crash recovery, classified retries, and HITL gates.
    Does NOT inherit BaseAgent and does NOT call the LLM directly.
    
    Args:
        agents: Dict mapping agent names to BaseAgent instances.
        job_queue_path: Path to the job queue SQLite database.
        pipeline_definitions_path: Path to the pipeline definitions JSON file.
    
    Raises:
        PipelineError: If pipeline definitions are invalid.
        QueueError: If queue initialization fails.
    """

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        job_queue_path: str = "dashboard/job_queue.db",
        pipeline_definitions_path: str = "config/pipelines.json",
    ) -> None:
        self._agents = agents
        self._definitions_path = Path(pipeline_definitions_path)
        self._pipelines: Dict[str, Dict[str, Any]] = {}
        self._defaults: Dict[str, Any] = {}
        self._load_definitions()
        
        self._queue = JobQueue(db_path=job_queue_path)
        self._queue.recover_orphans()
        
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._agent_exceptions = _import_agent_exceptions()

    def _load_definitions(self) -> None:
        """Load and validate pipeline definitions."""
        if not self._definitions_path.exists():
            raise PipelineError(f"Pipeline definitions not found: {self._definitions_path}")
        
        try:
            with open(self._definitions_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise PipelineError(f"Failed to load pipeline definitions: {e}") from e
        
        if not isinstance(data, dict):
            raise PipelineError("Pipeline definitions must be a JSON object")
        
        self._defaults = data.get("defaults", {})
        pipelines = data.get("pipelines", {})
        
        if not isinstance(pipelines, dict):
            raise PipelineError("'pipelines' must be a dict")
        
        for pipeline_name, pipeline_def in pipelines.items():
            if not isinstance(pipeline_def, dict):
                raise PipelineError(f"Pipeline '{pipeline_name}' must be a dict")
            steps = pipeline_def.get("steps", [])
            if not isinstance(steps, list) or len(steps) == 0:
                raise PipelineError(f"Pipeline '{pipeline_name}' must have at least one step")
            
            for step in steps:
                if not isinstance(step, dict):
                    raise PipelineError(f"Step in '{pipeline_name}' must be a dict")
                step_type = step.get("type")
                if step_type not in ("agent", "hitl"):
                    raise PipelineError(
                        f"Step type '{step_type}' in '{pipeline_name}' is invalid; must be 'agent' or 'hitl'"
                    )
                if step_type == "agent":
                    agent_name = step.get("agent")
                    if agent_name not in self._agents:
                        raise PipelineError(
                            f"Agent '{agent_name}' referenced in '{pipeline_name}' not found in agents dict"
                        )
        
        self._pipelines = pipelines

    def submit_job(
        self,
        job_type: str,
        parameters: Dict[str, Any],
        priority: int = 3,
        description: str = "",
    ) -> str:
        """
        Submit a new job to the queue.
        
        Args:
            job_type: Pipeline type to execute.
            parameters: Job parameters.
            priority: Priority (higher = sooner).
            description: Human-readable description.
        
        Returns:
            Job ID.
        
        Raises:
            PipelineError: If job_type is not defined.
        """
        if job_type not in self._pipelines:
            raise PipelineError(f"Unknown job_type: {job_type}")
        
        job_id = uuid.uuid4().hex
        self._queue.enqueue(
            job_id=job_id,
            job_type=job_type,
            parameters=parameters,
            priority=priority,
            description=description,
        )
        logger.info("Job submitted: %s (type=%s)", job_id, job_type)
        return job_id

    def process_next_job(self) -> Optional[str]:
        """
        Atomically claim and process the next eligible job.
        
        Returns:
            Job ID if a job was processed, None if queue is empty.
        """
        job = self._queue.claim_next()
        if job is None:
            return None
        
        logger.info("Processing job: %s", job.job_id)
        try:
            self._execute_pipeline(job)
        except Exception as e:
            logger.error("Job %s failed: %s", job.job_id, type(e).__name__)
        return job.job_id

    def _execute_pipeline(self, job: JobRecord) -> None:
        """Execute the pipeline for a job."""
        pipeline = self._pipelines[job.job_type]
        steps = pipeline["steps"]
        artifacts = dict(job.artifacts)
        completed_steps = list(job.completed_steps)
        
        for step in steps:
            step_name = step["name"]
            
            if step_name in completed_steps:
                continue
            
            self._queue.update_status(job.job_id, "running", current_step=step_name)
            self._queue.append_audit(job.job_id, "step_start", {"step": step_name})
            
            if step["type"] == "hitl":
                self._queue.update_status(job.job_id, "pending_review", current_step=step_name)
                self._queue.append_audit(job.job_id, "hitl_requested", {"step": step_name})
                logger.info("Job %s awaiting HITL at step %s", job.job_id, step_name)
                return
            
            # Agent step
            agent_name = step["agent"]
            agent = self._agents[agent_name]
            adapter = get_adapter(agent_name)
            
            try:
                task_prompt, context = adapter(artifacts, job.parameters, self._defaults)
            except AdapterError as e:
                error_msg = str(e)
                self._queue.update_status(job.job_id, "failed", error_message=error_msg)
                self._queue.append_audit(job.job_id, "step_failed", {
                    "step": step_name,
                    "error": type(e).__name__,
                })
                return
            
            start_time = time.time()
            try:
                result = agent.execute(task_prompt, context)
            except tuple(PERMANENT_ERRORS + self._agent_exceptions) as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                error_msg = f"{type(e).__name__}"
                self._queue.update_status(job.job_id, "failed", error_message=error_msg)
                self._queue.append_audit(job.job_id, "step_failed", {
                    "step": step_name,
                    "error": type(e).__name__,
                    "elapsed_ms": elapsed_ms,
                })
                logger.warning("Job %s permanent failure at %s: %s", job.job_id, step_name, type(e).__name__)
                return
            except Exception as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                if _is_transient_error(e):
                    retry_count = job.retry_count + 1
                    if retry_count >= 3:
                        self._queue.update_status(
                            job.job_id, "failed",
                            error_message=f"{type(e).__name__} (max retries)",
                            retry_count=retry_count,
                        )
                        self._queue.append_audit(job.job_id, "step_failed", {
                            "step": step_name,
                            "error": type(e).__name__,
                            "elapsed_ms": elapsed_ms,
                            "retry_count": retry_count,
                        })
                        logger.error("Job %s failed after %d retries", job.job_id, retry_count)
                    else:
                        backoff_sec = 2 ** retry_count
                        now = datetime.now(timezone.utc)
                        available_at = (now + timedelta(seconds=backoff_sec)).strftime("%Y-%m-%dT%H:%M:%SZ")
                        self._queue.update_status(
                            job.job_id, "queued",
                            retry_count=retry_count,
                            available_at=available_at,
                        )
                        self._queue.append_audit(job.job_id, "retry_scheduled", {
                            "step": step_name,
                            "error": type(e).__name__,
                            "retry_count": retry_count,
                            "backoff_sec": backoff_sec,
                            "elapsed_ms": elapsed_ms,
                        })
                        logger.warning(
                            "Job %s transient failure at %s, retry %d in %ds",
                            job.job_id, step_name, retry_count, backoff_sec,
                        )
                    return
                else:
                    error_msg = f"{type(e).__name__}"
                    self._queue.update_status(job.job_id, "failed", error_message=error_msg)
                    self._queue.append_audit(job.job_id, "step_failed", {
                        "step": step_name,
                        "error": type(e).__name__,
                        "elapsed_ms": elapsed_ms,
                    })
                    return
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            # Check if result is None
            if result is None:
                error_msg = "agent_returned_none"
                self._queue.update_status(job.job_id, "failed", error_message=error_msg)
                self._queue.append_audit(job.job_id, "step_failed", {
                    "step": step_name,
                    "error": error_msg,
                    "elapsed_ms": elapsed_ms,
                })
                return
            
            # Run guard
            try:
                if agent_name == "scout":
                    guard_scout_artifact(result.artifact)
                elif agent_name == "creator":
                    guard_creator_artifact(result.artifact)
            except AdapterError as e:
                error_msg = str(e)
                self._queue.update_status(job.job_id, "failed", error_message=error_msg)
                self._queue.append_audit(job.job_id, "step_failed", {
                    "step": step_name,
                    "error": type(e).__name__,
                    "elapsed_ms": elapsed_ms,
                })
                return
            
            # Publisher status mapping
            if agent_name == "publisher":
                try:
                    job_status = map_publisher_status(result.artifact)
                except AdapterError as e:
                    error_msg = str(e)
                    self._queue.update_status(job.job_id, "failed", error_message=error_msg)
                    self._queue.append_audit(job.job_id, "step_failed", {
                        "step": step_name,
                        "error": type(e).__name__,
                        "elapsed_ms": elapsed_ms,
                    })
                    return
                
                artifacts[agent_name] = result.artifact
                completed_steps.append(step_name)
                self._queue.complete_step(
                    job.job_id, step_name, artifacts, tuple(completed_steps), elapsed_ms
                )
                
                if job_status == "published":
                    self._queue.update_status(job.job_id, "published")
                    logger.info("Job %s published successfully", job.job_id)
                elif job_status == "failed":
                    self._queue.update_status(job.job_id, "failed", error_message="publisher_reported_failure")
                    logger.warning("Job %s failed: publisher reported failure", job.job_id)
                elif job_status == "deferred":
                    self._queue.update_status(job.job_id, "deferred")
                    logger.info("Job %s deferred", job.job_id)
                return
            
            # Persist step to SQLite
            artifacts[agent_name] = result.artifact
            completed_steps.append(step_name)
            self._queue.complete_step(
                job.job_id, step_name, artifacts, tuple(completed_steps), elapsed_ms
            )
            
            # --- ICM Enhancement: Save artifact to filesystem for human inspection ---
            from pathlib import Path
            run_dir = Path("dashboard/runs") / job.job_id / step_name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "output.md").write_text(result.artifact, encoding="utf-8")
            # ------------------------------------------------------------------------
            
            logger.info("Job %s completed step %s (%dms)", job.job_id, step_name, elapsed_ms)
        
        # All steps completed (shouldn't happen with default pipeline, but handle it)
        logger.warning("Job %s completed all steps without Publisher", job.job_id)
        self._queue.update_status(job.job_id, "completed")

    def start_worker(self) -> threading.Thread:
        """
        Start a background worker thread.
        
        Idempotent: calling twice returns the same thread.
        
        Returns:
            The worker thread.
        """
        if self._worker is not None and self._worker.is_alive():
            return self._worker
        
        self._stop_event.clear()
        
        def worker_loop() -> None:
            while not self._stop_event.is_set():
                try:
                    self.process_next_job()
                except Exception as e:
                    logger.error("Worker error: %s", type(e).__name__)
                self._stop_event.wait(1.0)
        
        self._worker = threading.Thread(target=worker_loop, daemon=True, name="pipeline-worker")
        self._worker.start()
        logger.info("Worker started")
        return self._worker

    def stop_worker(self) -> None:
        """Stop the background worker thread."""
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            logger.info("Worker stopped")

    def approve_job(self, job_id: str) -> bool:
        """
        Approve a job at a HITL gate.
        
        Args:
            job_id: Job identifier.
        
        Returns:
            True if approved, False if validation failed.
        """
        job = self._queue.get_job(job_id)
        if job is None:
            return False
        if job.status != "pending_review":
            return False
        
        pipeline = self._pipelines[job.job_type]
        current_step_def = None
        for step in pipeline["steps"]:
            if step["name"] == job.current_step:
                current_step_def = step
                break
        
        if current_step_def is None or current_step_def.get("type") != "hitl":
            return False
        
        completed_steps = list(job.completed_steps) + [job.current_step]
        self._queue.update_status(
            job_id, "queued",
            completed_steps=tuple(completed_steps),
            current_step="",
        )
        self._queue.append_audit(job_id, "hitl_approved", {"step": job.current_step})
        logger.info("Job %s approved at %s", job_id, job.current_step)
        return True

    def reject_job(self, job_id: str, reason: str) -> bool:
        """
        Reject a job at a HITL gate.
        
        Args:
            job_id: Job identifier.
            reason: Rejection reason.
        
        Returns:
            True if rejected, False if validation failed.
        """
        job = self._queue.get_job(job_id)
        if job is None:
            return False
        if job.status != "pending_review":
            return False
        
        pipeline = self._pipelines[job.job_type]
        current_step_def = None
        for step in pipeline["steps"]:
            if step["name"] == job.current_step:
                current_step_def = step
                break
        
        if current_step_def is None or current_step_def.get("type") != "hitl":
            return False
        
        self._queue.update_status(job_id, "rejected", error_message=reason)
        self._queue.append_audit(job_id, "hitl_rejected", {
            "step": job.current_step,
            "reason": reason,
        })
        logger.info("Job %s rejected: %s", job_id, reason)
        return True

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: Job identifier.
        
        Returns:
            True if cancelled, False if status is invalid.
        """
        job = self._queue.get_job(job_id)
        if job is None:
            return False
        if job.status not in ("queued", "pending_review"):
            return False
        
        self._queue.update_status(job_id, "cancelled")
        logger.info("Job %s cancelled", job_id)
        return True

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status with audit log.
        
        Args:
            job_id: Job identifier.
        
        Returns:
            Job status dict.
        """
        job = self._queue.get_job(job_id)
        if job is None:
            return {}
        
        from dataclasses import asdict
        status_dict = asdict(job)
        status_dict["audit_log"] = self._queue.get_audit_log(job_id)
        return status_dict

    def get_job_artifacts(self, job_id: str) -> Dict[str, str]:
        """
        Get job artifacts.
        
        Args:
            job_id: Job identifier.
        
        Returns:
            Artifacts dict.
        """
        job = self._queue.get_job(job_id)
        if job is None:
            return {}
        return dict(job.artifacts)

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List jobs with optional filter.
        
        Args:
            status: Optional status filter.
            limit: Max results.
            offset: Offset for pagination.
        
        Returns:
            List of job dicts.
        """
        from dataclasses import asdict
        jobs = self._queue.list_jobs(status=status, limit=limit, offset=offset)
        return [asdict(j) for j in jobs]

    def get_queue_stats(self) -> Dict[str, int]:
        """
        Get queue statistics.
        
        Returns:
            Dict mapping status to count.
        """
        return self._queue.get_stats()