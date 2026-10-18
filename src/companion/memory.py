# src/companion/memory.py
"""
Persistent, scope-isolated episodic memory store for the Jarvi Companion.

Provides remember/recall/forget operations with keyword/tag search and atomic writes.
Designed for deterministic retrieval now, with a pluggable backend path for future
vector-search upgrades.

This module NEVER logs memory content, queries, or tags (PII protection).
Only memory_id, category, and scope are logged at INFO level.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "CategoriesResult",
    "ErrorCode",
    "Memory",
    "MemoryEngine",
    "MemoryError",
    "MemoryResult",
    "RecallResult",
    "ScopeType",
    "StatsResult",
]

logger = logging.getLogger(__name__)

_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_SOURCE_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_TAG_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_MAX_CONTENT_CHARS = 10_000
_MAX_TAGS = 10
_MAX_RECALL_LIMIT = 100
_WARN_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB soft warning threshold

ScopeType = Literal["business", "personal"]


class ErrorCode:
    """Machine-readable error codes for memory operations."""

    INVALID_SCOPE = "INVALID_SCOPE"
    INVALID_CATEGORY = "INVALID_CATEGORY"
    INVALID_CONTENT = "INVALID_CONTENT"
    INVALID_TAGS = "INVALID_TAGS"
    INVALID_SOURCE = "INVALID_SOURCE"
    INVALID_LIMIT = "INVALID_LIMIT"
    MEMORY_NOT_FOUND = "MEMORY_NOT_FOUND"
    IO_ERROR = "IO_ERROR"
    INTERNAL = "INTERNAL"


class MemoryError(Exception):
    """Raised when a memory operation fails. Chains from the original exception."""

    def __init__(self, message: str, error_code: str = ErrorCode.INTERNAL) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class Memory:
    """A single, immutable memory entry.

    Attributes:
        id: Unique identifier (format: mem_<unix_ms>_<8-char-hex>).
        timestamp: ISO 8601 UTC timestamp of creation (microsecond precision).
        scope: "business" or "personal".
        category: Category name (lowercase alphanumeric + underscore).
        content: The memory content (max 10,000 chars).
        tags: Immutable tuple of tags.
        source: Source identifier (lowercase alphanumeric + underscore).
    """

    id: str
    timestamp: str
    scope: ScopeType
    category: str
    content: str
    tags: tuple[str, ...]
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "scope": self.scope,
            "category": self.category,
            "content": self.content,
            "tags": list(self.tags),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        """Deserialize from a dict."""
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            scope=data["scope"],
            category=data["category"],
            content=data["content"],
            tags=tuple(data.get("tags", [])),
            source=data.get("source", "unknown"),
        )


@dataclass(frozen=True)
class MemoryResult:
    """Result of a remember/forget/get_memory operation."""

    status: Literal["success", "error"]
    memory: Memory | None
    error: str | None
    error_code: str | None


@dataclass(frozen=True)
class RecallResult:
    """Result of a recall operation."""

    status: Literal["success", "error"]
    memories: tuple[Memory, ...]
    count: int
    error: str | None
    error_code: str | None


@dataclass(frozen=True)
class CategoriesResult:
    """Result of a list_categories operation."""

    status: Literal["success", "error"]
    categories: tuple[str, ...]
    error: str | None
    error_code: str | None


@dataclass(frozen=True)
class StatsResult:
    """Result of a get_stats operation."""

    status: Literal["success", "error"]
    total_memories: int
    by_category: dict[str, int]
    by_source: dict[str, int]
    error: str | None
    error_code: str | None


class MemoryEngine:
    """Persistent memory storage for the Companion, strictly isolated by scope.

    Each instance is bound to a single scope at initialization. The scope
    determines the physical directory used for storage, enforcing isolation
    at the OS level.

    Args:
        workspace_path: Root of jarvis-workspace.
        scope: Strictly "business" or "personal". Locked at initialization.

    Raises:
        MemoryError: If workspace is invalid, unwritable, or scope is invalid.
    """

    def __init__(self, workspace_path: str | Path, scope: ScopeType) -> None:
        if scope not in ("business", "personal"):
            raise MemoryError(
                f"Invalid scope: {scope!r}. Must be 'business' or 'personal'.",
                ErrorCode.INVALID_SCOPE,
            )

        if not isinstance(workspace_path, (str, Path)):
            raise MemoryError(
                "workspace_path must be a str or Path",
                ErrorCode.INTERNAL,
            )

        self.workspace = Path(workspace_path).resolve()
        self.scope: ScopeType = scope
        self.memories_dir = self.workspace / "brain" / "memories" / self.scope

        try:
            self.memories_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise MemoryError(
                f"Failed to create memory directory: {type(e).__name__}",
                ErrorCode.IO_ERROR,
            ) from e

        # Thread-safe per-category locks with explicit locking pattern
        self._locks_lock = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def _get_lock(self, category: str) -> threading.Lock:
        """Get or create a lock for a category (thread-safe).

        Args:
            category: The category name.

        Returns:
            A threading.Lock for the category.
        """
        with self._locks_lock:
            if category not in self._locks:
                self._locks[category] = threading.Lock()
            return self._locks[category]

    def remember(
        self,
        content: str,
        category: str,
        tags: list[str] | None = None,
        source: str = "companion",
    ) -> MemoryResult:
        """Store a new memory atomically.

        Args:
            content: The memory content (max 10,000 chars, non-empty).
            category: Category name (max 32 chars, lowercase alphanumeric + underscore).
            tags: Optional list of tags (max 10, each max 32 chars, lowercase alphanumeric + underscore).
            source: Source identifier (max 32 chars, lowercase alphanumeric + underscore).

        Returns:
            MemoryResult with status="success" and the created Memory, or error details.

        Raises:
            No exceptions raised; errors are returned via MemoryResult.
        """
        # Validate content
        if not isinstance(content, str) or not content.strip():
            return self._error_memory(
                "Content must be a non-empty string",
                ErrorCode.INVALID_CONTENT,
            )
        if len(content) > _MAX_CONTENT_CHARS:
            return self._error_memory(
                f"Content exceeds {_MAX_CONTENT_CHARS} chars",
                ErrorCode.INVALID_CONTENT,
            )

        # Validate category
        if not isinstance(category, str) or not _CATEGORY_RE.match(category):
            return self._error_memory(
                f"Invalid category: must match {_CATEGORY_RE.pattern}",
                ErrorCode.INVALID_CATEGORY,
            )

        # Validate source
        if not isinstance(source, str) or not _SOURCE_RE.match(source):
            return self._error_memory(
                f"Invalid source: must match {_SOURCE_RE.pattern}",
                ErrorCode.INVALID_SOURCE,
            )

        # Validate tags
        validated_tags: list[str] = []
        if tags is not None:
            if not isinstance(tags, list) or len(tags) > _MAX_TAGS:
                return self._error_memory(
                    f"Tags must be a list of max {_MAX_TAGS} items",
                    ErrorCode.INVALID_TAGS,
                )
            for tag in tags:
                if not isinstance(tag, str) or not _TAG_RE.match(tag):
                    return self._error_memory(
                        f"Invalid tag: must match {_TAG_RE.pattern}",
                        ErrorCode.INVALID_TAGS,
                    )
                validated_tags.append(tag)

        # Create memory with microsecond-precision timestamp
        memory_id = f"mem_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        memory = Memory(
            id=memory_id,
            timestamp=timestamp,
            scope=self.scope,
            category=category,
            content=content,
            tags=tuple(validated_tags),
            source=source,
        )

        file_path = self.memories_dir / f"{category}.json"

        # Thread-safe read-modify-write
        with self._get_lock(category):
            try:
                memories = self._load_memories(file_path)
                memories.append(memory.to_dict())
                memories.sort(key=lambda x: x["timestamp"], reverse=True)
                self._save_memories_atomic(file_path, memories)
            except MemoryError as e:
                return self._error_memory(str(e), e.error_code)
            except (OSError, json.JSONDecodeError) as e:
                logger.error("Failed to save memory: %s", type(e).__name__)
                return self._error_memory(
                    f"IO error: {type(e).__name__}",
                    ErrorCode.IO_ERROR,
                )

        logger.info(
            "Memory saved: %s (category=%s, scope=%s)",
            memory_id,
            category,
            self.scope,
        )
        return MemoryResult(status="success", memory=memory, error=None, error_code=None)

    def recall(
        self,
        query: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        limit: int = 10,
    ) -> RecallResult:
        """Retrieve memories matching filters.

        Args:
            query: Optional keyword search (case-insensitive substring in content).
            category: Optional exact category match.
            tags: Optional list of tags (memory must have ALL specified tags).
            source: Optional exact source match.
            limit: Maximum number of memories to return (1-100, default 10).

        Returns:
            RecallResult with tuple of Memory objects, sorted by timestamp descending.
        """
        # Validate limit
        if not isinstance(limit, int) or not (1 <= limit <= _MAX_RECALL_LIMIT):
            return self._error_recall(
                f"Limit must be between 1 and {_MAX_RECALL_LIMIT}",
                ErrorCode.INVALID_LIMIT,
            )

        # Validate category filter
        if category is not None and (
            not isinstance(category, str) or not _CATEGORY_RE.match(category)
        ):
            return self._error_recall(
                f"Invalid category filter: must match {_CATEGORY_RE.pattern}",
                ErrorCode.INVALID_CATEGORY,
            )

        # Determine target files
        if category is not None:
            target_files = [self.memories_dir / f"{category}.json"]
        else:
            try:
                target_files = sorted(self.memories_dir.glob("*.json"))
            except OSError as e:
                logger.error("Failed to list memory files: %s", type(e).__name__)
                return self._error_recall(
                    f"IO error: {type(e).__name__}",
                    ErrorCode.IO_ERROR,
                )

        all_memories: list[Memory] = []

        try:
            for file_path in target_files:
                if not file_path.is_file():
                    continue
                memories = self._load_memories(file_path)
                for mem_dict in memories:
                    memory = Memory.from_dict(mem_dict)

                    # Apply filters
                    if source is not None and memory.source != source:
                        continue
                    if tags is not None and not all(t in memory.tags for t in tags):
                        continue
                    if query is not None and query.lower() not in memory.content.lower():
                        continue

                    all_memories.append(memory)
        except MemoryError as e:
            return self._error_recall(str(e), e.error_code)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load memories for recall: %s", type(e).__name__)
            return self._error_recall(
                f"IO error: {type(e).__name__}",
                ErrorCode.IO_ERROR,
            )

        # Sort by timestamp descending (newest first)
        all_memories.sort(key=lambda m: m.timestamp, reverse=True)
        limited = all_memories[:limit]

        return RecallResult(
            status="success",
            memories=tuple(limited),
            count=len(limited),
            error=None,
            error_code=None,
        )

    def forget(self, memory_id: str) -> MemoryResult:
        """Delete a memory by ID.

        Args:
            memory_id: The exact ID of the memory to delete.

        Returns:
            MemoryResult indicating success or MEMORY_NOT_FOUND.
        """
        if not isinstance(memory_id, str) or not memory_id:
            return self._error_memory(
                "memory_id must be a non-empty string",
                ErrorCode.INTERNAL,
            )

        try:
            target_files = sorted(self.memories_dir.glob("*.json"))
        except OSError as e:
            logger.error("Failed to list memory files: %s", type(e).__name__)
            return self._error_memory(
                f"IO error: {type(e).__name__}",
                ErrorCode.IO_ERROR,
            )

        for file_path in target_files:
            category = file_path.stem
            with self._get_lock(category):
                try:
                    memories = self._load_memories(file_path)
                except MemoryError as e:
                    return self._error_memory(str(e), e.error_code)
                except (OSError, json.JSONDecodeError) as e:
                    logger.error("Failed to load memory file: %s", type(e).__name__)
                    return self._error_memory(
                        f"IO error: {type(e).__name__}",
                        ErrorCode.IO_ERROR,
                    )

                original_count = len(memories)
                filtered = [m for m in memories if m["id"] != memory_id]

                if len(filtered) < original_count:
                    filtered.sort(key=lambda x: x["timestamp"], reverse=True)
                    try:
                        self._save_memories_atomic(file_path, filtered)
                    except MemoryError as e:
                        return self._error_memory(str(e), e.error_code)
                    except (OSError, json.JSONDecodeError) as e:
                        logger.error("Failed to save memory file: %s", type(e).__name__)
                        return self._error_memory(
                            f"IO error: {type(e).__name__}",
                            ErrorCode.IO_ERROR,
                        )

                    logger.info(
                        "Memory deleted: %s (category=%s, scope=%s)",
                        memory_id,
                        category,
                        self.scope,
                    )
                    return MemoryResult(
                        status="success", memory=None, error=None, error_code=None
                    )

        return self._error_memory(
            f"Memory {memory_id} not found",
            ErrorCode.MEMORY_NOT_FOUND,
        )

    def get_memory(self, memory_id: str) -> MemoryResult:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: The exact ID of the memory to retrieve.

        Returns:
            MemoryResult with the Memory if found, or MEMORY_NOT_FOUND.
        """
        if not isinstance(memory_id, str) or not memory_id:
            return self._error_memory(
                "memory_id must be a non-empty string",
                ErrorCode.INTERNAL,
            )

        try:
            target_files = sorted(self.memories_dir.glob("*.json"))
        except OSError as e:
            logger.error("Failed to list memory files: %s", type(e).__name__)
            return self._error_memory(
                f"IO error: {type(e).__name__}",
                ErrorCode.IO_ERROR,
            )

        try:
            for file_path in target_files:
                for mem_dict in self._load_memories(file_path):
                    if mem_dict["id"] == memory_id:
                        return MemoryResult(
                            status="success",
                            memory=Memory.from_dict(mem_dict),
                            error=None,
                            error_code=None,
                        )
        except MemoryError as e:
            return self._error_memory(str(e), e.error_code)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load memories: %s", type(e).__name__)
            return self._error_memory(
                f"IO error: {type(e).__name__}",
                ErrorCode.IO_ERROR,
            )

        return self._error_memory(
            f"Memory {memory_id} not found",
            ErrorCode.MEMORY_NOT_FOUND,
        )

    def list_categories(self) -> CategoriesResult:
        """Return a sorted tuple of all categories with at least one memory in this scope.

        Returns:
            CategoriesResult with sorted tuple of category names.
        """
        try:
            categories = {
                f.stem
                for f in self.memories_dir.glob("*.json")
                if f.is_file()
            }
            return CategoriesResult(
                status="success",
                categories=tuple(sorted(categories)),
                error=None,
                error_code=None,
            )
        except OSError as e:
            logger.error("Failed to list categories: %s", type(e).__name__)
            return self._error_categories(
                f"IO error: {type(e).__name__}",
                ErrorCode.IO_ERROR,
            )

    def get_stats(self) -> StatsResult:
        """Return aggregate statistics for this scope.

        Returns:
            StatsResult with total count and breakdowns by category and source.
        """
        total = 0
        by_category: dict[str, int] = {}
        by_source: dict[str, int] = {}

        try:
            target_files = sorted(self.memories_dir.glob("*.json"))
        except OSError as e:
            logger.error("Failed to list memory files: %s", type(e).__name__)
            return self._error_stats(
                f"IO error: {type(e).__name__}",
                ErrorCode.IO_ERROR,
            )

        try:
            for file_path in target_files:
                category = file_path.stem
                memories = self._load_memories(file_path)
                count = len(memories)
                total += count
                by_category[category] = by_category.get(category, 0) + count
                for mem in memories:
                    src = mem.get("source", "unknown")
                    by_source[src] = by_source.get(src, 0) + 1
        except MemoryError as e:
            return self._error_stats(str(e), e.error_code)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load memories for stats: %s", type(e).__name__)
            return self._error_stats(
                f"IO error: {type(e).__name__}",
                ErrorCode.IO_ERROR,
            )

        return StatsResult(
            status="success",
            total_memories=total,
            by_category=by_category,
            by_source=by_source,
            error=None,
            error_code=None,
        )

    def _load_memories(self, file_path: Path) -> list[dict[str, Any]]:
        """Load memories from a JSON file, gracefully handling corruption.

        Args:
            file_path: Path to the JSON file.

        Returns:
            List of memory dicts. Empty list if file missing or corrupt.

        Raises:
            MemoryError: Only on unrecoverable IO errors (not corrupt JSON).
        """
        if not file_path.exists():
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(
                "Corrupt memory file detected, treating as empty: %s (%s)",
                file_path.name,
                type(e).__name__,
            )
            return []
        except OSError as e:
            raise MemoryError(
                f"Failed to read memory file: {type(e).__name__}",
                ErrorCode.IO_ERROR,
            ) from e

        if not isinstance(data, list):
            logger.warning(
                "Corrupt memory file (not a list): %s, treating as empty",
                file_path.name,
            )
            return []

        return data

    def _save_memories_atomic(
        self, file_path: Path, memories: list[dict[str, Any]]
    ) -> None:
        """Atomically save memories via temp file + fsync + os.replace.

        Args:
            file_path: Target path for the JSON file.
            memories: List of memory dicts to save.

        Raises:
            MemoryError: If the write fails.
        """
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        fd = None
        temp_path: str | None = None

        try:
            fd, temp_path = tempfile.mkstemp(
                dir=str(self.memories_dir), suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = None  # os.fdopen took ownership
                json.dump(memories, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, file_path)

            # Soft warning for memory bloat
            try:
                size = file_path.stat().st_size
                if size > _WARN_FILE_SIZE_BYTES:
                    logger.warning(
                        "Memory file %s exceeds %d MB; consider splitting category",
                        file_path.name,
                        _WARN_FILE_SIZE_BYTES // (1024 * 1024),
                    )
            except OSError:
                pass

            temp_path = None  # Successfully replaced
        except OSError as e:
            raise MemoryError(
                f"Failed to write memory file: {type(e).__name__}",
                ErrorCode.IO_ERROR,
            ) from e
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    def _error_memory(self, message: str, code: str) -> MemoryResult:
        """Build an error MemoryResult."""
        logger.warning("Memory operation failed [%s]: %s", code, message)
        return MemoryResult(
            status="error", memory=None, error=message, error_code=code
        )

    def _error_recall(self, message: str, code: str) -> RecallResult:
        """Build an error RecallResult."""
        logger.warning("Recall operation failed [%s]: %s", code, message)
        return RecallResult(
            status="error", memories=(), count=0, error=message, error_code=code
        )

    def _error_categories(self, message: str, code: str) -> CategoriesResult:
        """Build an error CategoriesResult."""
        logger.warning("Categories operation failed [%s]: %s", code, message)
        return CategoriesResult(
            status="error", categories=(), error=message, error_code=code
        )

    def _error_stats(self, message: str, code: str) -> StatsResult:
        """Build an error StatsResult."""
        logger.warning("Stats operation failed [%s]: %s", code, message)
        return StatsResult(
            status="error",
            total_memories=0,
            by_category={},
            by_source={},
            error=message,
            error_code=code,
        )