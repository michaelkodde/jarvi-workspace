# src/companion/query_api.py
"""
Query API for Jarvi Companion.

Provides interfaces for checking job status and handling Human-in-the-Loop (HITL) 
approvals. Acts as a safe abstraction layer over the PipelineFlow.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

__all__ = [
    "JobStatusResult",
    "QueryAPI",
    "QueryAPIError",
]

logger = logging.getLogger(__name__)


class QueryAPIError(Exception):
    """Raised when a QueryAPI operation fails. Chains from the original exception."""

    def __init__(self, message: str, error_code: str = "INTERNAL") -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class JobStatusResult:
    """
    Result of a job status query (MCP-Ready Schema).

    Attributes:
        status: Current state ("queued", "running", "pending_review", "completed", "failed").
        job: Optional dict containing job details and artifacts.
        error: Optional error message if status is "failed".
        error_code: Optional machine-readable error code.
    """
    status: str
    job: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None


class QueryAPI:
    """
    API for querying job status and managing HITL approvals.

    Args:
        pipeline_flow: The PipelineFlow instance managing the jobs.

    Raises:
        ValueError: If the injected pipeline_flow lacks required methods.
    """

    def __init__(self, pipeline_flow: Any) -> None:
        # Contract validation: ensure the injected dependency meets requirements
        if not hasattr(pipeline_flow, "get_job_status"):
            raise ValueError("pipeline_flow must have a 'get_job_status' method")
        if not hasattr(pipeline_flow, "approve_job"):
            raise ValueError("pipeline_flow must have an 'approve_job' method")
        if not hasattr(pipeline_flow, "reject_job"):
            raise ValueError("pipeline_flow must have a 'reject_job' method")

        self.pipeline_flow = pipeline_flow

    def get_job_status(self, ticket_id: str) -> JobStatusResult:
        """
        Get the current status of a job.

        Args:
            ticket_id: The unique identifier of the job.

        Returns:
            JobStatusResult containing the current status and job details.
        """
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
            # Constitution #5: Swallow 3rd-party exception, raise custom with chaining
            raise QueryAPIError(
                f"Failed to query status for ticket {ticket_id}", 
                error_code="QUERY_FAILED"
            ) from e

    def approve_job(self, ticket_id: str, feedback: str | None = None) -> None:
        """
        Approve a job pending HITL review.

        Args:
            ticket_id: The unique identifier of the job.
            feedback: Optional user feedback for the approval.

        Raises:
            QueryAPIError: If the approval fails, chained from the original exception.
        """
        try:
            self.pipeline_flow.approve_job(ticket_id, feedback=feedback)
            logger.info("Job %s approved.", ticket_id)
        except Exception as e:
            logger.error("Failed to approve job %s: %s", ticket_id, type(e).__name__)
            raise QueryAPIError(
                f"Failed to approve job {ticket_id}", 
                error_code="APPROVAL_FAILED"
            ) from e

    def reject_job(self, ticket_id: str, reason: str | None = None) -> None:
        """
        Reject a job pending HITL review.

        Args:
            ticket_id: The unique identifier of the job.
            reason: Optional reason for rejection.

        Raises:
            QueryAPIError: If the rejection fails, chained from the original exception.
        """
        try:
            self.pipeline_flow.reject_job(ticket_id, reason=reason)
            logger.info("Job %s rejected.", ticket_id)
        except Exception as e:
            logger.error("Failed to reject job %s: %s", ticket_id, type(e).__name__)
            raise QueryAPIError(
                f"Failed to reject job {ticket_id}", 
                error_code="REJECTION_FAILED"
            ) from e