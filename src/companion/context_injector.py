# src/companion/context_injector.py
"""
Context Injector (Module 2).
Pre-execution memory query module that retrieves relevant past executions and injects 
past_context into job parameters with half-life decay.
"""
from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
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

# Half-life decay constants (in days)
_SHORT_TERM_DAYS = 30
_MEDIUM_TERM_DAYS = 90

class MemoryProtocol(Protocol):
    """Protocol defining the expected interface for the MemoryEngine."""
    def recall(self, category: str, tags: List[str], limit: int) -> Any:
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
    Implements half-life decay to prioritize recent, relevant executions.
    """

    def __init__(
        self,
        memory_engine: MemoryProtocol,
        max_context_memories: int = 5,
        max_context_chars: int = 2000,
    ) -> None:
        if not hasattr(memory_engine, "recall"):
            raise ContextError("memory_engine must implement 'recall' method", ErrorCode.INTERNAL)
        if not (1 <= max_context_memories <= 20):
            raise ContextError("max_context_memories must be between 1 and 20", ErrorCode.INTERNAL)
        if not (100 <= max_context_chars <= 10000):
            raise ContextError("max_context_chars must be between 100 and 10000", ErrorCode.INTERNAL)

        self.memory_engine = memory_engine
        self.max_context_memories = max_context_memories
        self.max_context_chars = max_context_chars

    def _calculate_weight(self, age_days: float) -> float:
        """Calculates the half-life weight of a memory based on its age."""
        if age_days <= _SHORT_TERM_DAYS:
            return 1.0
        elif age_days <= _MEDIUM_TERM_DAYS:
            return 0.5
        else:
            return 0.1

    def _parse_timestamp(self, ts: Any) -> datetime:
        """Safely parses various timestamp formats into a timezone-aware datetime."""
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    def inject_context(
        self,
        job_type: str,
        parameters: dict[str, Any],
    ) -> ContextResult:
        """
        Query memory for relevant past executions and inject context into parameters.
        """
        # 1. Validate job_type
        if not isinstance(job_type, str) or not _JOB_TYPE_RE.match(job_type):
            raise ContextError(f"Invalid job_type: must match ^[a-z][a-z0-9_]{{0,31}}$", ErrorCode.INVALID_JOB_TYPE)

        # 2. Validate parameters size
        try:
            params_json = json.dumps(parameters)
            if len(params_json.encode("utf-8")) > _MAX_PARAMETERS_SIZE_BYTES:
                raise ContextError("Parameters exceed 100KB limit", ErrorCode.PARAMETERS_TOO_LARGE)
        except (TypeError, ValueError) as e:
            raise ContextError(f"Parameters not JSON-serializable: {type(e).__name__}", ErrorCode.PARAMETERS_TOO_LARGE) from e

        # 3. Query memory (Request 3x limit to account for filtering dropouts)
        try:
            recall_result = self.memory_engine.recall(
                category="execution_history",
                tags=[job_type],
                limit=self.max_context_memories * 3,
            )
        except Exception as e:
            logger.warning("MemoryEngine query failed for job_type=%s: %s. Degrading gracefully.", job_type, type(e).__name__)
            enriched = copy.deepcopy(parameters)
            enriched["past_context"] = ""
            return ContextResult(status="success", enriched_parameters=enriched, memories_used=0, error=None, error_code=None)

        # 4. Filter and Score Memories
        memories = getattr(recall_result, "memories", [])
        now = datetime.now(timezone.utc)
        
        scored_memories = []
        for m in memories:
            content = getattr(m, "content", "")
            tags = getattr(m, "tags", [])
            ts = self._parse_timestamp(getattr(m, "timestamp", now))
            
            # Strict filtering: Must match job_type in tags OR content
            if job_type not in tags and job_type.lower() not in content.lower():
                continue
                
            age_days = (now - ts).total_seconds() / 86400.0
            weight = self._calculate_weight(age_days)
            
            scored_memories.append({
                "memory": m,
                "weight": weight,
                "timestamp": ts,
                "content": str(content)
            })

        # 5. Sort by Weight (Half-life) DESC, then Timestamp DESC
        scored_memories.sort(key=lambda x: (x["weight"], x["timestamp"]), reverse=True)

        # 6. Priority-Based Context Filling (Respects Half-Life)
        truncated_memories = scored_memories[:self.max_context_memories]
        
        context_parts = []
        current_chars = 0
        
        for item in truncated_memories:
            content = item["content"]
            # If adding this memory exceeds the limit, truncate it or stop
            if current_chars + len(content) > self.max_context_chars:
                remaining = self.max_context_chars - current_chars
                if remaining > 50:  # Minimum useful chunk size
                    context_parts.append(content[:remaining - 3] + "...")
                break
            
            context_parts.append(content)
            current_chars += len(content)

        past_context = "\n\n---\n\n".join(context_parts) if context_parts else ""

        # 7. Inject into a deep copy
        enriched = copy.deepcopy(parameters)
        enriched["past_context"] = past_context
        memories_used = len(context_parts)

        logger.info("Context injected: job_type=%s, memories_used=%d, total_chars=%d", job_type, memories_used, len(past_context))

        return ContextResult(
            status="success",
            enriched_parameters=enriched,
            memories_used=memories_used,
            error=None,
            error_code=None,
        )