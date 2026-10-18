# src/companion/__init__.py
"""Jarvi Companion: the brain that discovers and operates the toolchain.

Uses lazy imports via __getattr__ to avoid pulling in heavy dependencies
(agents, router, discovery) on package load.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.companion.discovery import (
        AgentContract,
        DiscoveryEngine,
        DiscoveryError,
        ParamSpec,
        PipelineContract,
        PlaybookResolution,
        ScanError,
        SystemRegistry,
        UtilContract,
    )
    from src.companion.tool_registry import (
        ErrorCode,
        ToolRegistry,
        ToolRegistryError,
        ToolResult,
        ToolSchema,
    )

__all__ = [
    # Discovery
    "AgentContract",
    "DiscoveryEngine",
    "DiscoveryError",
    "ParamSpec",
    "PipelineContract",
    "PlaybookResolution",
    "ScanError",
    "SystemRegistry",
    "UtilContract",
    # Tool Registry
    "ErrorCode",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolResult",
    "ToolSchema",
]


def __getattr__(name: str) -> Any:
    """Lazy import to avoid heavy dependencies on package load."""
    if name in {
        "AgentContract", "DiscoveryEngine", "DiscoveryError", "ParamSpec",
        "PipelineContract", "PlaybookResolution", "ScanError", "SystemRegistry",
        "UtilContract",
    }:
        from src.companion import discovery
        return getattr(discovery, name)
    if name in {
        "ErrorCode", "ToolRegistry", "ToolRegistryError", "ToolResult", "ToolSchema",
    }:
        from src.companion import tool_registry
        return getattr(tool_registry, name)
    raise AttributeError(f"module 'src.companion' has no attribute {name!r}")