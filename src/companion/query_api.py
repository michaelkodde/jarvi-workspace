# src/companion/query_api.py
"""
Query API for Jarvi Companion.

Provides interfaces for checking job status, handling Human-in-the-Loop (HITL) 
approvals, and exposing operational state. Acts as a safe abstraction layer 
over the PipelineFlow.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.daemon.exceptions import ErrorCode

__all__ = [
    "ApprovalItem",
    "ApprovalResult",
    "ApprovalsResult",
    "DaemonStats",
    "JobStatusResult",
    "QueryAPI",
    "QueryAPIError",
]

logger = logging.getLogger(__name__)


class QueryAPIError(Exception):
    """Raised when a QueryAPI operation fails. Chains from the original exception."""

    def __init__(self, message: str, error_code: str = ErrorCode.INTERNAL) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class JobStatusResult:
    """Result of a job status query."""
    status: str
    job: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None


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
    error: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class ApprovalResult:
    """Result of an approve_job or reject_job operation."""
    status: str
    job_id: str | None = None
    error: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class DaemonStats:
    """Operational statistics for the daemon/queue."""
    is_running: bool
    tickets_ingested: int
    uptime_sec: int
    last_heartbeat: str
    queue_stats: dict[str, int]


class QueryAPI:
    """
    API for querying job status, managing HITL approvals, and retrieving stats.
    """

    def __init__(self, pipeline_flow: Any) -> None:
        # Contract validation: ensure the injected dependency meets requirements
        required_methods = ["get_job_status", "approve_job", "reject_job", "list_jobs", "get_queue_stats"]
        for method in required_methods:
            if not hasattr(pipeline_flow, method):
                raise ValueError(f"pipeline_flow must have a '{method}' method")

        self.pipeline_flow = pipeline_flow

    def get_job_status(self, ticket_id: str) -> JobStatusResult:
        """Get the current status of a job."""
        try:
            status_data = self.pipeline_flow.get_job_status(ticket_id)
            
            # Handle both dict and dataclass/object returns gracefully
            if isinstance(status_data, dict):
                return JobStatusResult(
                    status=status_data.get("status", "unknown"),
                    job=status_data.get("job"),
                    error=status_data.get("error"),
                    error_code=status_data.get("error_code"),
                )
            
            return JobStatusResult(
                status=getattr(status_data, "status", "unknown"),
                job=getattr(status_data, "job", None),
                error=getattr(status_data, "error", None),
                error_code=getattr(status_data, "error_code", None),
            )
        except Exception as e:
            logger.error("Failed to get job status for %s: %s", ticket_id, type(e).__name__)
            raise QueryAPIError(
                f"Failed to query status for ticket {ticket_id}", 
                error_code="QUERY_FAILED"
            ) from e

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
        try:
            self.pipeline_flow.approve_job(job_id, feedback=feedback)
            return ApprovalResult(status="success", job_id=job_id, error=None, error_code=None)
        except Exception as e:
            logger.error("Failed to approve job %s: %s", job_id, type(e).__name__)
            error_code = ErrorCode.JOB_NOT_PENDING if "pending" in str(e).lower() else ErrorCode.INTERNAL
            return ApprovalResult(
                status="error",
                job_id=job_id,
                error=str(e),
                error_code=error_code,
            )

    def reject_job(self, job_id: str, reason: str) -> ApprovalResult:
        """Reject a pending job, canceling the pipeline."""
        try:
            self.pipeline_flow.reject_job(job_id, reason=reason)
            return ApprovalResult(status="success", job_id=job_id, error=None, error_code=None)
        except Exception as e:
            logger.error("Failed to reject job %s: %s", job_id, type(e).__name__)
            error_code = ErrorCode.JOB_NOT_PENDING if "pending" in str(e).lower() else ErrorCode.INTERNAL
            return ApprovalResult(
                status="error",
                job_id=job_id,
                error=str(e),
                error_code=error_code,
            )

    def get_stats(self) -> DaemonStats:
        """Retrieve current daemon/queue operational statistics."""
        try:
            queue_stats = self.pipeline_flow.get_queue_stats()
            return DaemonStats(
                is_running=False,  # QueryAPI is stateless; DaemonLoop manages the true running state
                tickets_ingested=0,
                uptime_sec=0,
                last_heartbeat="",
                queue_stats=queue_stats,
            )
        except Exception as e:
            logger.error("Failed to get stats: %s", type(e).__name__)
            raise QueryAPIError(f"Failed to get stats: {type(e).__name__}", ErrorCode.INTERNAL) from e