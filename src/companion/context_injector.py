# src/companion/context_injector.py
"""
Context-aware execution module for Jarvi Companion.

Retrieves relevant past executions from MemoryEngine and injects
context into job parameters before submission to PipelineFlow.
"""
from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, List, Protocol

__all__ = [
    "ContextError",
    "ContextInjector",
    "ContextResult",
    "ErrorCode",
    "MemoryProtocol",
]

logger = logging.getLogger(__name__)

_JOB_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_MAX_PARAMETERS_SIZE_BYTES = 100 * 1024  # 100 KB


class MemoryProtocol(Protocol):
    """Protocol defining the expected interface for the MemoryEngine."""

    def recall(
        self, category: str, tags: List[str], limit: int
    ) -> Any:  # Returns a RecallResult-like object
        ...


class ErrorCode:
    """Machine-readable error codes for context injection operations."""

    INVALID_JOB_TYPE = "INVALID_JOB_TYPE"
    PARAMETERS_TOO_LARGE = "PARAMETERS_TOO_LARGE"
    MEMORY_QUERY_FAILED = "MEMORY_QUERY_FAILED"
    INTERNAL = "INTERNAL"


class ContextError(Exception):
    """Raised when context injection fails. Chains from the original exception."""

    def __init__(self, message: str, error_code: str = ErrorCode.INTERNAL) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class ContextResult:
    """Result of a context injection operation."""

    status: str
    enriched_parameters: dict[str, Any] | None
    memories_used: int
    error: str | None
    error_code: str | None


class ContextInjector:
    """
    Pre-execution memory query module that enriches job parameters with relevant context.

    Args:
        memory_engine: Existing MemoryEngine instance (must implement MemoryProtocol).
        max_context_memories: Max memories to retrieve (1-10, default 3).
        max_context_chars: Max characters for past_context (100-10000, default 2000).

    Raises:
        ContextError: If memory_engine is invalid or parameters are invalid.
    """

    def __init__(
        self,
        memory_engine: MemoryProtocol,
        max_context_memories: int = 3,
        max_context_chars: int = 2000,
    ) -> None:
        if not hasattr(memory_engine, "recall"):
            raise ContextError(
                "memory_engine must implement 'recall' method",
                ErrorCode.INTERNAL,
            )
        if not (1 <= max_context_memories <= 10):
            raise ContextError(
                "max_context_memories must be between 1 and 10",
                ErrorCode.INTERNAL,
            )
        if not (100 <= max_context_chars <= 10000):
            raise ContextError(
                "max_context_chars must be between 100 and 10000",
                ErrorCode.INTERNAL,
            )

        self.memory_engine = memory_engine
        self.max_context_memories = max_context_memories
        self.max_context_chars = max_context_chars

    def inject_context(
        self,
        job_type: str,
        parameters: dict[str, Any],
    ) -> ContextResult:
        """
        Query memory for relevant past executions and inject context into parameters.

        Args:
            job_type: Pipeline type (must match ^[a-z][a-z0-9_]{0,31}$).
            parameters: Job parameters dict.

        Returns:
            ContextResult with enriched parameters or error details.

        Raises:
            ContextError: If job_type is invalid or parameters are not serializable.
        """
        # 1. Validate job_type
        if not isinstance(job_type, str) or not _JOB_TYPE_RE.match(job_type):
            raise ContextError(
                f"Invalid job_type: must match ^[a-z][a-z0-9_]{{0,31}}$",
                ErrorCode.INVALID_JOB_TYPE,
            )

        # 2. Validate parameters size and serializability
        try:
            params_json = json.dumps(parameters)
            if len(params_json.encode("utf-8")) > _MAX_PARAMETERS_SIZE_BYTES:
                raise ContextError(
                    "Parameters exceed 100KB limit",
                    ErrorCode.PARAMETERS_TOO_LARGE,
                )
        except (TypeError, ValueError) as e:
            raise ContextError(
                f"Parameters not JSON-serializable: {type(e).__name__}",
                ErrorCode.PARAMETERS_TOO_LARGE,
            ) from e

        # 3. Query memory for relevant past executions
        try:
            # Request 2x limit to account for post-filtering dropouts
            recall_result = self.memory_engine.recall(
                category="execution_history",
                tags=[job_type],
                limit=self.max_context_memories * 2,
            )
        except Exception as e:
            logger.warning(
                "MemoryEngine query failed for job_type=%s: %s",
                job_type,
                type(e).__name__,
            )
            # Graceful degradation: inject empty past_context
            enriched = copy.deepcopy(parameters)
            enriched["past_context"] = ""
            return ContextResult(
                status="success",
                enriched_parameters=enriched,
                memories_used=0,
                error=None,
                error_code=None,
            )

        # 4. Filter memories robustly (check tags OR case-insensitive content)
        memories = getattr(recall_result, "memories", [])
        filtered_memories = [
            m
            for m in memories
            if hasattr(m, "content")
            and (
                job_type in getattr(m, "tags", [])
                or job_type.lower() in getattr(m, "content", "").lower()
            )
        ]

        # 5. Sort by timestamp descending (newest first)
        # ISO 8601 strings are lexicographically sortable
        sorted_memories = sorted(
            filtered_memories,
            key=lambda m: getattr(m, "timestamp", ""),
            reverse=True,
        )

        # 6. Truncate to max_context_memories
        truncated_memories = sorted_memories[: self.max_context_memories]

        # 7. Concatenate and truncate content
        if not truncated_memories:
            past_context = ""
        else:
            contents = [str(getattr(m, "content", "")) for m in truncated_memories]
            past_context = "\n\n".join(contents)

            if len(past_context) > self.max_context_chars:
                past_context = past_context[: self.max_context_chars - 3] + "..."

        # 8. Inject into a deep copy of parameters to prevent downstream mutation
        enriched = copy.deepcopy(parameters)
        enriched["past_context"] = past_context

        memories_used = len(truncated_memories)
        logger.info(
            "Context injected: job_type=%s, memories_used=%d",
            job_type,
            memories_used,
        )

        return ContextResult(
            status="success",
            enriched_parameters=enriched,
            memories_used=memories_used,
            error=None,
            error_code=None,
        )