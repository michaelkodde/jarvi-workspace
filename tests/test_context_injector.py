# tests/test_context_injector.py
"""Tests for the Context Injector module."""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from src.companion.context_injector import (
    ContextError,
    ContextInjector,
    ContextResult,
    ErrorCode,
)


@dataclass
class MockMemory:
    """Mock Memory object for testing."""

    content: str
    timestamp: str
    tags: tuple[str, ...]


@dataclass
class MockRecallResult:
    """Mock RecallResult for testing."""

    memories: list[MockMemory]
    count: int
    error: str | None = None
    error_code: str | None = None


@pytest.fixture
def mock_memory_engine() -> MagicMock:
    """Create a mock MemoryEngine with recall method."""
    mock = MagicMock()
    mock.recall.return_value = MockRecallResult(memories=[], count=0)
    return mock


def test_inject_context_retrieves_and_enriches(
    mock_memory_engine: MagicMock,
) -> None:
    """Test that inject_context retrieves memories and enriches parameters."""
    memories = [
        MockMemory(
            content="Previous research task completed successfully",
            timestamp="2026-01-19T10:00:00Z",
            tags=("research",),
        ),
        MockMemory(
            content="Another research task with different results",
            timestamp="2026-01-18T10:00:00Z",
            tags=("research",),
        ),
    ]
    mock_memory_engine.recall.return_value = MockRecallResult(
        memories=memories, count=2
    )

    injector = ContextInjector(mock_memory_engine)
    result = injector.inject_context("research", {"query": "test"})

    assert result.status == "success"
    assert result.memories_used == 2
    assert "past_context" in result.enriched_parameters
    assert "Previous research task" in result.enriched_parameters["past_context"]


def test_memories_filtered_by_tag_or_content(
    mock_memory_engine: MagicMock,
) -> None:
    """Test that memories are filtered by job_type tag OR case-insensitive content."""
    memories = [
        MockMemory(
            content="Task A about market analysis",  # Has tag, but not the word "research"
            timestamp="2026-01-19T10:00:00Z",
            tags=("research",),
        ),
        MockMemory(
            content="Task B mentions RESEARCH in passing",  # Has word, but wrong tag
            timestamp="2026-01-18T10:00:00Z",
            tags=("analysis",),
        ),
        MockMemory(
            content="Task C is completely unrelated",
            timestamp="2026-01-17T10:00:00Z",
            tags=("other",),
        ),
    ]
    mock_memory_engine.recall.return_value = MockRecallResult(
        memories=memories, count=3
    )

    injector = ContextInjector(mock_memory_engine)
    result = injector.inject_context("research", {"query": "test"})

    # Both Task A (matches tag) and Task B (matches content case-insensitive) should be included
    assert result.memories_used == 2
    assert "Task A about market analysis" in result.enriched_parameters["past_context"]
    assert "Task B mentions RESEARCH in passing" in result.enriched_parameters["past_context"]
    assert "Task C is completely unrelated" not in result.enriched_parameters["past_context"]


def test_memories_sorted_by_timestamp_descending(
    mock_memory_engine: MagicMock,
) -> None:
    """Test that memories are sorted by timestamp descending (newest first)."""
    memories = [
        MockMemory(
            content="Older research task",
            timestamp="2026-01-17T10:00:00Z",
            tags=("research",),
        ),
        MockMemory(
            content="Newer research task",
            timestamp="2026-01-19T10:00:00Z",
            tags=("research",),
        ),
    ]
    mock_memory_engine.recall.return_value = MockRecallResult(
        memories=memories, count=2
    )

    injector = ContextInjector(mock_memory_engine)
    result = injector.inject_context("research", {"query": "test"})

    past_context = result.enriched_parameters["past_context"]
    assert past_context.index("Newer research task") < past_context.index(
        "Older research task"
    )


def test_max_context_memories_enforced(
    mock_memory_engine: MagicMock,
) -> None:
    """Test that only max_context_memories are injected."""
    memories = [
        MockMemory(
            content=f"Research task {i}",
            timestamp=f"2026-01-{19-i:02d}T10:00:00Z",
            tags=("research",),
        )
        for i in range(10)
    ]
    mock_memory_engine.recall.return_value = MockRecallResult(
        memories=memories, count=10
    )

    injector = ContextInjector(mock_memory_engine, max_context_memories=3)
    result = injector.inject_context("research", {"query": "test"})

    assert result.memories_used == 3


def test_max_context_chars_enforced_with_truncation(
    mock_memory_engine: MagicMock,
) -> None:
    """Test that past_context is truncated with '...' if exceeds max_context_chars."""
    long_content = "A" * 3000
    memories = [
        MockMemory(
            content=long_content,
            timestamp="2026-01-19T10:00:00Z",
            tags=("research",),
        ),
    ]
    mock_memory_engine.recall.return_value = MockRecallResult(
        memories=memories, count=1
    )

    injector = ContextInjector(mock_memory_engine, max_context_chars=1000)
    result = injector.inject_context("research", {"query": "test"})

    past_context = result.enriched_parameters["past_context"]
    assert len(past_context) == 1000
    assert past_context.endswith("...")


def test_empty_memory_result_injects_empty_past_context(
    mock_memory_engine: MagicMock,
) -> None:
    """Test that empty memory result injects empty past_context (not an error)."""
    mock_memory_engine.recall.return_value = MockRecallResult(
        memories=[], count=0
    )

    injector = ContextInjector(mock_memory_engine)
    result = injector.inject_context("research", {"query": "test"})

    assert result.status == "success"
    assert result.memories_used == 0
    assert result.enriched_parameters["past_context"] == ""


def test_memory_engine_failure_graceful_degradation(
    mock_memory_engine: MagicMock,
) -> None:
    """Test that MemoryEngine failure results in graceful degradation."""
    mock_memory_engine.recall.side_effect = Exception("Database error")

    injector = ContextInjector(mock_memory_engine)
    result = injector.inject_context("research", {"query": "test"})

    assert result.status == "success"
    assert result.memories_used == 0
    assert result.enriched_parameters["past_context"] == ""


def test_invalid_job_type_rejected() -> None:
    """Test that invalid job_type raises ContextError with INVALID_JOB_TYPE."""
    mock_engine = MagicMock()
    injector = ContextInjector(mock_engine)

    with pytest.raises(ContextError) as exc_info:
        injector.inject_context("INVALID_JOB_TYPE", {"query": "test"})

    assert exc_info.value.error_code == ErrorCode.INVALID_JOB_TYPE


def test_parameters_too_large_rejected() -> None:
    """Test that parameters >100KB raises ContextError with PARAMETERS_TOO_LARGE."""
    mock_engine = MagicMock()
    injector = ContextInjector(mock_engine)

    large_params = {"data": "x" * (100 * 1024 + 1)}

    with pytest.raises(ContextError) as exc_info:
        injector.inject_context("research", large_params)

    assert exc_info.value.error_code == ErrorCode.PARAMETERS_TOO_LARGE


def test_parameters_not_serializable_rejected() -> None:
    """Test that non-serializable parameters raise ContextError."""
    mock_engine = MagicMock()
    injector = ContextInjector(mock_engine)

    bad_params = {"data": object()}

    with pytest.raises(ContextError) as exc_info:
        injector.inject_context("research", bad_params)

    assert exc_info.value.error_code == ErrorCode.PARAMETERS_TOO_LARGE


def test_pii_not_logged(mock_memory_engine: MagicMock, caplog) -> None:
    """Test that memory content is never logged (PII protection)."""
    memories = [
        MockMemory(
            content="SENSITIVE_USER_DATA_SHOULD_NOT_BE_LOGGED",
            timestamp="2026-01-19T10:00:00Z",
            tags=("research",),
        ),
    ]
    mock_memory_engine.recall.return_value = MockRecallResult(
        memories=memories, count=1
    )

    injector = ContextInjector(mock_memory_engine)

    with caplog.at_level("INFO"):
        injector.inject_context("research", {"query": "test"})

    for record in caplog.records:
        assert "SENSITIVE_USER_DATA_SHOULD_NOT_BE_LOGGED" not in record.message


def test_invalid_memory_engine_raises_context_error() -> None:
    """Test that invalid memory_engine raises ContextError at startup."""
    with pytest.raises(ContextError) as exc_info:
        ContextInjector("not_a_memory_engine")  # type: ignore

    assert exc_info.value.error_code == ErrorCode.INTERNAL


def test_exception_chaining() -> None:
    """Test that ContextError chains from original exception."""
    mock_engine = MagicMock()
    injector = ContextInjector(mock_engine)

    bad_params = {"data": object()}

    with pytest.raises(ContextError) as exc_info:
        injector.inject_context("research", bad_params)

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, TypeError)


def test_deep_copy_prevents_mutation(mock_memory_engine: MagicMock) -> None:
    """Test that enriched_parameters is a deep copy, preventing downstream mutation."""
    memories = [
        MockMemory(
            content="Previous task",
            timestamp="2026-01-19T10:00:00Z",
            tags=("research",),
        ),
    ]
    mock_memory_engine.recall.return_value = MockRecallResult(
        memories=memories, count=1
    )

    injector = ContextInjector(mock_memory_engine)
    original_params = {"nested": {"key": "value"}}
    
    result = injector.inject_context("research", original_params)

    # Mutate the result
    result.enriched_parameters["nested"]["key"] = "mutated"  # type: ignore

    # Original must remain unchanged
    assert original_params["nested"]["key"] == "value"
    injector = ContextInjector(mock_memory_engine)
    original_params = {"nested": {"key": "value"}}
    
    result = injector.inject_context("research", original_params)

    # Mutate the result
    result.enriched_parameters["nested"]["key"] = "mutated"  # type: ignore

    # Original must remain unchanged
    assert original_params["nested"]["key"] == "value"