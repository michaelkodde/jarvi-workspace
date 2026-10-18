# tests/test_memory.py
"""Tests for the scope-isolated Memory Layer."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import pytest

from src.companion.memory import (
    ErrorCode,
    Memory,
    MemoryEngine,
    MemoryError,
    MemoryResult,
    RecallResult,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------


def test_remember_and_recall_round_trip(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")

    result = engine.remember(
        content="User prefers concise responses",
        category="preferences",
        tags=["communication", "style"],
        source="user",
    )

    assert result.status == "success"
    assert result.memory is not None
    assert result.memory.id.startswith("mem_")
    assert result.memory.scope == "business"
    assert result.memory.category == "preferences"
    assert result.memory.content == "User prefers concise responses"
    assert result.memory.tags == ("communication", "style")
    assert result.memory.source == "user"
    assert result.memory.timestamp.endswith("Z")
    assert "." in result.memory.timestamp  # Microsecond precision

    recall = engine.recall(category="preferences")
    assert recall.status == "success"
    assert recall.count == 1
    assert recall.memories[0].content == "User prefers concise responses"


def test_scope_isolation(workspace: Path) -> None:
    biz_engine = MemoryEngine(workspace, scope="business")
    per_engine = MemoryEngine(workspace, scope="personal")

    biz_engine.remember("Business secret", category="work")
    per_engine.remember("Personal secret", category="home")

    biz_recall = biz_engine.recall()
    per_recall = per_engine.recall()

    assert biz_recall.count == 1
    assert "Business" in biz_recall.memories[0].content
    assert per_recall.count == 1
    assert "Personal" in per_recall.memories[0].content

    # Verify physical directory separation
    assert (workspace / "brain" / "memories" / "business" / "work.json").exists()
    assert (workspace / "brain" / "memories" / "personal" / "home.json").exists()
    assert not (workspace / "brain" / "memories" / "business" / "home.json").exists()


def test_keyword_search_case_insensitive(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    engine.remember("Python is great for AI", category="tech")
    engine.remember("Java is verbose", category="tech")

    res = engine.recall(query="python")
    assert res.count == 1
    assert "Python" in res.memories[0].content

    res = engine.recall(query="PYTHON")
    assert res.count == 1


def test_tag_filtering_and_logic(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    engine.remember("A", category="tech", tags=["python", "ai"])
    engine.remember("B", category="tech", tags=["python"])
    engine.remember("C", category="tech", tags=["java"])

    # Single tag
    res = engine.recall(tags=["python"])
    assert res.count == 2

    # AND logic: must have ALL tags
    res = engine.recall(tags=["python", "ai"])
    assert res.count == 1
    assert res.memories[0].content == "A"


def test_source_filtering(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    engine.remember("From user", category="notes", source="user")
    engine.remember("From system", category="notes", source="system")

    res = engine.recall(source="user")
    assert res.count == 1
    assert res.memories[0].content == "From user"


def test_category_filtering(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    engine.remember("Tech note", category="tech")
    engine.remember("Food note", category="food")

    res = engine.recall(category="tech")
    assert res.count == 1
    assert res.memories[0].content == "Tech note"


def test_limit_enforcement(workspace: Path) -> None:
    """Test that limit parameter correctly restricts results."""
    engine = MemoryEngine(workspace, scope="business")
    for i in range(15):
        engine.remember(f"Note {i:02d}", category="notes")

    res = engine.recall(limit=5)
    assert res.count == 5


def test_sorting_by_timestamp_descending(workspace: Path) -> None:
    """Test that memories are sorted by timestamp descending (newest first)."""
    engine = MemoryEngine(workspace, scope="business")
    
    # Write memories with explicit delays to ensure unique timestamps
    engine.remember("First", category="notes")
    import time
    time.sleep(0.001)  # 1ms delay
    engine.remember("Second", category="notes")
    time.sleep(0.001)
    engine.remember("Third", category="notes")

    res = engine.recall(limit=10)
    assert res.count == 3
    
    # Verify timestamps are in descending order (newest first)
    timestamps = [m.timestamp for m in res.memories]
    assert timestamps == sorted(timestamps, reverse=True)
    
    # Verify content order matches timestamp order
    assert res.memories[0].content == "Third"
    assert res.memories[1].content == "Second"
    assert res.memories[2].content == "First"


def test_forget_success(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    res = engine.remember("Temporary", category="temp")
    assert res.memory is not None
    mem_id = res.memory.id

    forget_res = engine.forget(mem_id)
    assert forget_res.status == "success"
    assert forget_res.memory is None

    assert engine.recall().count == 0


def test_forget_not_found(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    res = engine.forget("nonexistent_id")
    assert res.status == "error"
    assert res.error_code == ErrorCode.MEMORY_NOT_FOUND


def test_get_memory(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    res = engine.remember("Specific data", category="data")
    assert res.memory is not None

    get_res = engine.get_memory(res.memory.id)
    assert get_res.status == "success"
    assert get_res.memory is not None
    assert get_res.memory.content == "Specific data"
    assert get_res.memory.id == res.memory.id


def test_get_memory_not_found(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    res = engine.get_memory("nonexistent")
    assert res.status == "error"
    assert res.error_code == ErrorCode.MEMORY_NOT_FOUND


def test_list_categories(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    engine.remember("A", category="cat1")
    engine.remember("B", category="cat2")
    engine.remember("C", category="cat1")  # Duplicate category

    cats = engine.list_categories()
    assert cats.status == "success"
    assert cats.categories == ("cat1", "cat2")


def test_list_categories_empty(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    cats = engine.list_categories()
    assert cats.status == "success"
    assert cats.categories == ()


def test_get_stats(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    engine.remember("A", category="cat1", source="src1")
    engine.remember("B", category="cat1", source="src2")
    engine.remember("C", category="cat2", source="src1")

    stats = engine.get_stats()
    assert stats.status == "success"
    assert stats.total_memories == 3
    assert stats.by_category == {"cat1": 2, "cat2": 1}
    assert stats.by_source == {"src1": 2, "src2": 1}


def test_get_stats_empty(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    stats = engine.get_stats()
    assert stats.status == "success"
    assert stats.total_memories == 0
    assert stats.by_category == {}
    assert stats.by_source == {}


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_invalid_scope_at_init(workspace: Path) -> None:
    with pytest.raises(MemoryError) as exc_info:
        MemoryEngine(workspace, scope="invalid")  # type: ignore[arg-type]
    assert exc_info.value.error_code == ErrorCode.INVALID_SCOPE


def test_invalid_workspace_type() -> None:
    with pytest.raises(MemoryError) as exc_info:
        MemoryEngine(123, scope="business")  # type: ignore[arg-type]
    assert exc_info.value.error_code == ErrorCode.INTERNAL


def test_empty_content_rejected(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    res = engine.remember("", category="test")
    assert res.status == "error"
    assert res.error_code == ErrorCode.INVALID_CONTENT

    res = engine.remember("   ", category="test")
    assert res.status == "error"
    assert res.error_code == ErrorCode.INVALID_CONTENT


def test_content_too_long_rejected(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    res = engine.remember("x" * 10_001, category="test")
    assert res.status == "error"
    assert res.error_code == ErrorCode.INVALID_CONTENT


def test_invalid_category_rejected(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")

    for bad_cat in ("", "INVALID", "has-dash", "has space", "1starts_digit", "a" * 33):
        res = engine.remember("valid", category=bad_cat)
        assert res.status == "error"
        assert res.error_code == ErrorCode.INVALID_CATEGORY


def test_invalid_source_rejected(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    res = engine.remember("valid", category="test", source="INVALID")
    assert res.status == "error"
    assert res.error_code == ErrorCode.INVALID_SOURCE


def test_invalid_tags_rejected(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")

    # Too many tags
    res = engine.remember("valid", category="test", tags=["t"] * 11)
    assert res.status == "error"
    assert res.error_code == ErrorCode.INVALID_TAGS

    # Invalid tag format
    res = engine.remember("valid", category="test", tags=["bad tag!"])
    assert res.status == "error"
    assert res.error_code == ErrorCode.INVALID_TAGS

    # Not a list
    res = engine.remember("valid", category="test", tags="not_a_list")  # type: ignore[arg-type]
    assert res.status == "error"
    assert res.error_code == ErrorCode.INVALID_TAGS


def test_invalid_limit_rejected(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")

    res = engine.recall(limit=0)
    assert res.status == "error"
    assert res.error_code == ErrorCode.INVALID_LIMIT

    res = engine.recall(limit=101)
    assert res.status == "error"
    assert res.error_code == ErrorCode.INVALID_LIMIT

    res = engine.recall(limit=-1)
    assert res.status == "error"
    assert res.error_code == ErrorCode.INVALID_LIMIT


def test_recall_invalid_category_filter(workspace: Path) -> None:
    """Test that invalid category filter in recall returns error."""
    engine = MemoryEngine(workspace, scope="business")
    engine.remember("Test", category="valid")

    res = engine.recall(category="INVALID")
    assert res.status == "error"
    assert res.error_code == ErrorCode.INVALID_CATEGORY


# ---------------------------------------------------------------------------
# Atomic write safety
# ---------------------------------------------------------------------------


def test_corrupt_file_graceful_degradation(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    engine.remember("Safe data", category="safe")

    file_path = engine.memories_dir / "safe.json"

    # Simulate corruption
    file_path.write_text("{ invalid json", encoding="utf-8")

    # Recall should gracefully handle corruption
    res = engine.recall(category="safe")
    assert res.status == "success"
    assert res.count == 0  # Corrupt file treated as empty

    # New write should succeed and overwrite the corrupt file
    engine.remember("New safe data", category="safe")
    content = file_path.read_text(encoding="utf-8")
    assert "New safe data" in content
    assert "{ invalid json" not in content


def test_corrupt_file_not_list_treated_as_empty(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")

    file_path = engine.memories_dir / "weird.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text('{"not": "a list"}', encoding="utf-8")

    res = engine.recall(category="weird")
    assert res.status == "success"
    assert res.count == 0


# ---------------------------------------------------------------------------
# PII protection
# ---------------------------------------------------------------------------


def test_pii_never_logged(workspace: Path, caplog: pytest.LogCaptureFixture) -> None:
    engine = MemoryEngine(workspace, scope="personal")
    secret = "super-secret-credit-card-1234-5678-9012"

    with caplog.at_level(logging.DEBUG):
        engine.remember(secret, category="finance")
        engine.recall(query="credit")
        engine.recall(category="finance")
        engine.get_stats()

    for record in caplog.records:
        assert secret not in record.getMessage()


def test_pii_in_query_never_logged(workspace: Path, caplog: pytest.LogCaptureFixture) -> None:
    engine = MemoryEngine(workspace, scope="personal")
    secret_query = "my-password-is-hunter2"

    with caplog.at_level(logging.DEBUG):
        engine.recall(query=secret_query)

    for record in caplog.records:
        assert secret_query not in record.getMessage()


# ---------------------------------------------------------------------------
# Memory dataclass
# ---------------------------------------------------------------------------


def test_memory_to_dict_and_from_dict() -> None:
    memory = Memory(
        id="mem_123_abc",
        timestamp="2026-01-01T00:00:00.000000Z",
        scope="business",
        category="test",
        content="Hello",
        tags=("a", "b"),
        source="user",
    )

    d = memory.to_dict()
    assert d["id"] == "mem_123_abc"
    assert d["tags"] == ["a", "b"]  # Serialized as list

    restored = Memory.from_dict(d)
    assert restored == memory
    assert restored.tags == ("a", "b")  # Restored as tuple


def test_memory_is_immutable() -> None:
    memory = Memory(
        id="mem_123",
        timestamp="2026-01-01T00:00:00.000000Z",
        scope="business",
        category="test",
        content="Hello",
        tags=(),
        source="user",
    )

    with pytest.raises(AttributeError):
        memory.content = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_recall_with_no_filters_returns_all(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    engine.remember("A", category="cat1")
    engine.remember("B", category="cat2")

    res = engine.recall()
    assert res.count == 2


def test_recall_nonexistent_category_returns_empty(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    res = engine.recall(category="nonexistent")
    assert res.status == "success"
    assert res.count == 0


def test_default_source_is_companion(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    res = engine.remember("Test", category="test")
    assert res.memory is not None
    assert res.memory.source == "companion"


def test_default_tags_is_empty_tuple(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    res = engine.remember("Test", category="test")
    assert res.memory is not None
    assert res.memory.tags == ()


def test_remember_with_empty_tags_list(workspace: Path) -> None:
    """Test that empty tags list creates memory with empty tuple."""
    engine = MemoryEngine(workspace, scope="business")
    res = engine.remember("Test", category="test", tags=[])
    assert res.status == "success"
    assert res.memory is not None
    assert res.memory.tags == ()


def test_forget_empty_memory_id_rejected(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    res = engine.forget("")
    assert res.status == "error"
    assert res.error_code == ErrorCode.INTERNAL


def test_get_memory_empty_id_rejected(workspace: Path) -> None:
    engine = MemoryEngine(workspace, scope="business")
    res = engine.get_memory("")
    assert res.status == "error"
    assert res.error_code == ErrorCode.INTERNAL


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_writes_same_category(workspace: Path) -> None:
    """Test that concurrent writes to the same category don't lose data."""
    engine = MemoryEngine(workspace, scope="business")

    def write_memory(i: int) -> None:
        engine.remember(f"Memory {i}", category="concurrent")

    threads = [threading.Thread(target=write_memory, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    res = engine.recall(category="concurrent")
    assert res.count == 10, f"Expected 10 memories, got {res.count}"


def test_concurrent_writes_different_categories(workspace: Path) -> None:
    """Test that concurrent writes to different categories work correctly."""
    engine = MemoryEngine(workspace, scope="business")

    results = []
    results_lock = threading.Lock()

    def write_to_category(cat: str, count: int) -> None:
        for i in range(count):
            result = engine.remember(f"Memory {i}", category=cat)
            with results_lock:
                results.append(result)

    threads = [
        threading.Thread(target=write_to_category, args=(f"cat{i}", 5))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify all remember() calls succeeded
    for result in results:
        assert result.status == "success", f"Failed to write memory: {result.error}"

    # Use limit=100 to retrieve all memories (default limit is 10)
    res = engine.recall(limit=100)
    assert res.count == 20, f"Expected 20 memories, got {res.count}"